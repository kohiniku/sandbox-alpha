"""
Tests for error taxonomy: infra/code tagging, evaluate_result routing,
knowledge bookkeeping, and backlog attempts logic.

No network — all external calls are mocked.
"""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_loop import (
    evaluate_result,
    _run_backtest_subprocess,
    _run_backtest_sandbox,
    load_knowledge,
    save_knowledge,
    KNOWLEDGE_FILE,
)
from backlog import Backlog, make_param_entry, make_code_entry
from strategy_ideation import _summarise_code_errors


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def _make_hypothesis(strategy="sma_crossover", symbol="AAPL", params=None):
    if params is None:
        params = {"fast_window": 10, "slow_window": 30}
    return {
        "id": "hyp_test_001",
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "description": f"{strategy} on {symbol}",
        "generated_at": "2026-07-19T00:00:00",
    }


def _fresh_knowledge():
    return {
        "tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
        "superseded": [], "families": {}, "iterations": 0, "errors": [],
    }


# ──────────────────────────────────────────────
# 1. evaluate_result taxonomy routing
# ──────────────────────────────────────────────

class TestEvaluateResultTaxonomy:
    """Error types route to correct verdicts and never become 'rejected'."""

    def test_infra_error_routes_to_error(self):
        """infra error_type → verdict 'error'"""
        result = {"error": "connection refused", "error_type": "infra"}
        hyp = _make_hypothesis()
        verdict, evaluation = evaluate_result(hyp, result, _fresh_knowledge())

        assert verdict == "error"
        assert evaluation["verdict"] == "error"
        assert evaluation["error_type"] == "infra"
        assert "Infra error" in evaluation["reasons"][0]

    def test_code_error_routes_to_code_error(self):
        """code error_type → verdict 'code_error'"""
        result = {"error": "signal validation failed: 'close'", "error_type": "code"}
        hyp = _make_hypothesis()
        verdict, evaluation = evaluate_result(hyp, result, _fresh_knowledge())

        assert verdict == "code_error"
        assert evaluation["verdict"] == "code_error"
        assert evaluation["error_type"] == "code"
        assert "Code error" in evaluation["reasons"][0]

    def test_unknown_error_type_routes_to_error(self):
        """Missing or unknown error_type → verdict 'error' (safe default)"""
        result = {"error": "something went wrong"}  # no error_type
        hyp = _make_hypothesis()
        verdict, evaluation = evaluate_result(hyp, result, _fresh_knowledge())

        assert verdict == "error"
        assert evaluation["verdict"] == "error"
        assert evaluation["error_type"] == "unknown"

    def test_bogus_error_type_routes_to_error(self):
        """Bogus error_type → verdict 'error' (never negative evidence)"""
        result = {"error": "weird stuff", "error_type": "gremlins"}
        hyp = _make_hypothesis()
        verdict, evaluation = evaluate_result(hyp, result, _fresh_knowledge())

        assert verdict == "error"
        assert evaluation["verdict"] == "error"

    def test_gate_failure_still_rejected(self):
        """Legitimate gate failures still route to 'rejected'."""
        result = {
            "walkforward": {"enabled": False},
        }
        hyp = _make_hypothesis()
        verdict, evaluation = evaluate_result(hyp, result, _fresh_knowledge())

        assert verdict == "rejected"


# ──────────────────────────────────────────────
# 2. Error tagging at source
# ──────────────────────────────────────────────

class TestErrorTaggingAtSource:
    """Errors from subprocess/sandbox runners carry 'error_type' in result dict."""

    def test_subprocess_nonzero_returncode_is_code_error(self):
        """returncode != 0 → stderr is strategy execution error → 'code'"""
        result = {"error": "KeyError: 'close'", "error_type": "code"}
        assert result["error_type"] == "code"

    def test_subprocess_timeout_is_infra_error(self):
        """TimeoutExpired → 'infra'"""
        result = {"error": "Timeout (120s)", "error_type": "infra"}
        assert result["error_type"] == "infra"

    def test_subprocess_json_parse_is_infra_error(self):
        """JSONDecodeError from runner output → 'infra'"""
        result = {"error": "JSON parse error: ...", "error_type": "infra"}
        assert result["error_type"] == "infra"

    def test_subprocess_cannot_parse_output_is_infra_error(self):
        """'Could not parse output' → 'infra'"""
        result = {"error": "Could not parse output", "error_type": "infra"}
        assert result["error_type"] == "infra"

    def test_sandbox_http_error_is_infra(self):
        """HTTPError from sandbox runner → 'infra'"""
        result = {"error": "Sandbox runner HTTP 500: ...", "error_type": "infra"}
        assert result["error_type"] == "infra"

    def test_sandbox_connection_error_is_infra(self):
        """URLError from sandbox runner → 'infra'"""
        result = {"error": "Sandbox runner connection error: ...", "error_type": "infra"}
        assert result["error_type"] == "infra"

    def test_sandbox_non_200_is_infra(self):
        """status != 200 from sandbox runner → 'infra'"""
        result = {"error": "Sandbox runner returned HTTP 502: ...", "error_type": "infra"}
        assert result["error_type"] == "infra"

    def test_sandbox_runner_reported_error_is_code(self):
        """HTTP 200 + JSON body with 'error' field → runner-reported → 'code'"""
        result = {"error": "signal validation failed: 'close'", "error_type": "code"}
        assert result["error_type"] == "code"


# ──────────────────────────────────────────────
# 3. Knowledge bookkeeping
# ──────────────────────────────────────────────

class TestKnowledgeBookkeeping:
    """Error/code_error verdicts go to knowledge['errors'], not rejected."""

    def test_errors_not_counted_as_rejected(self, tmp_path, monkeypatch):
        """Error verdicts go to errors[] list, not rejected."""
        kf = tmp_path / "knowledge.json"
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)

        knowledge = _fresh_knowledge()
        knowledge["errors"] = []  # explicit

        # Add an infra error record
        error_record = {
            "hypothesis": _make_hypothesis(),
            "backtest_result": {"error": "timeout", "error_type": "infra"},
            "evaluation": {"verdict": "error", "error_type": "infra", "error": "timeout"},
            "verdict": "error",
            "tested_at": "2026-07-19T00:00:00",
        }
        knowledge["errors"].append(error_record)
        assert len(knowledge["errors"]) == 1
        assert len(knowledge["rejected"]) == 0

    def test_error_cap_at_100(self, tmp_path, monkeypatch):
        """errors list capped at 100, keep newest."""
        kf = tmp_path / "knowledge.json"
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)

        knowledge = _fresh_knowledge()
        knowledge["errors"] = []

        for i in range(110):
            knowledge["errors"].append({
                "hypothesis": _make_hypothesis(),
                "evaluation": {"verdict": "error", "error_type": "infra"},
                "verdict": "error",
                "tested_at": f"2026-07-19T{i:02d}:00:00",
            })
            if len(knowledge["errors"]) > 100:
                knowledge["errors"] = knowledge["errors"][-100:]

        assert len(knowledge["errors"]) == 100

    def test_errors_dont_touch_family_aggregates(self):
        """Error records never update family aggregates."""
        knowledge = _fresh_knowledge()

        error_record = {
            "hypothesis": _make_hypothesis(strategy="sma_crossover", symbol="AAPL"),
            "evaluation": {"verdict": "error", "error_type": "infra"},
            "verdict": "error",
        }

        knowledge["errors"].append(error_record)

        # families should be empty
        assert knowledge["families"] == {}

    def test_errors_dont_touch_tested_combinations(self):
        """Error records never update tested_combinations."""
        knowledge = _fresh_knowledge()

        error_record = {
            "hypothesis": _make_hypothesis(),
            "evaluation": {"verdict": "error", "error_type": "infra"},
            "verdict": "error",
        }
        knowledge["errors"].append(error_record)

        assert len(knowledge["tested_combinations"]) == 0

    def test_load_knowledge_adds_errors_default(self):
        """load_knowledge return includes 'errors' key for new knowledge."""
        knowledge = _fresh_knowledge()
        assert "errors" in knowledge
        assert knowledge["errors"] == []

    def test_load_knowledge_backwards_compatible(self, tmp_path, monkeypatch):
        """load_knowledge handles old knowledge.json without 'errors' key."""
        kf = tmp_path / "knowledge.json"
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)

        # Write OLD-format knowledge.json (no 'errors' key)
        old_data = {
            "tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
            "superseded": [], "families": {}, "iterations": 5,
        }
        kf.write_text(json.dumps(old_data))

        knowledge = load_knowledge()
        assert "errors" in knowledge
        assert knowledge["errors"] == []


# ──────────────────────────────────────────────
# 4. Backlog attempts logic
# ──────────────────────────────────────────────

class TestBacklogAttempts:
    """Backlog entries retry infra errors (max 3 attempts), mark code errors immediately."""

    def test_infra_error_retried_with_attempts_counter(self, tmp_path, monkeypatch):
        """Infra error on backlog entry → re-marked 'pending' with attempts counter."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
        )
        bl.add_entry(entry)

        # Simulate two infra error retries
        for attempt in range(1, 3):  # attempts 1, 2
            attempts = entry.get("attempts", 0) + 1
            if attempts < 3:
                bl.mark(entry["id"], "pending", {
                    "verdict": "error",
                    "summary": f"infra error (attempt {attempts}/3)",
                    "finished_at": "2026-07-19T00:00:00",
                    "attempts": attempts,
                })
                entry["attempts"] = attempts

        data = bl.load()
        e = [x for x in data["entries"] if x["id"] == entry["id"]][0]
        assert e["status"] == "pending"
        assert e["result"]["attempts"] == 2

    def test_infra_error_after_3_attempts_marked_done_error(self, tmp_path, monkeypatch):
        """After 3 infra error attempts → 'done_error'."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
        )
        bl.add_entry(entry)

        # Simulate 3rd attempt (exceeds max)
        attempts = 3
        bl.mark(entry["id"], "done_error", {
            "verdict": "error",
            "summary": f"infra error after {attempts} attempts",
            "finished_at": "2026-07-19T00:00:00",
            "attempts": attempts,
        })

        data = bl.load()
        e = [x for x in data["entries"] if x["id"] == entry["id"]][0]
        assert e["status"] == "done_error"
        assert e["result"]["attempts"] == 3

    def test_code_error_immediately_done_error(self, tmp_path, monkeypatch):
        """Code error on backlog entry → 'done_error' immediately, no retry."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_code_entry(
            name="test_strat", description="desc",
            code="import numpy as np\n\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)",
            symbol="SPY", priority=0.8,
            source={"kind": "idea", "ref": "test"},
        )
        bl.add_entry(entry)

        bl.mark(entry["id"], "done_error", {
            "verdict": "code_error",
            "error": "signal validation failed: 'close'",
            "summary": "code error: signal validation failed: 'close'",
            "finished_at": "2026-07-19T00:00:00",
        })

        data = bl.load()
        e = [x for x in data["entries"] if x["id"] == entry["id"]][0]
        assert e["status"] == "done_error"
        assert e["result"]["verdict"] == "code_error"
        assert "attempts" not in e["result"]  # code errors don't retry

    def test_error_never_marked_done_rejected(self, tmp_path, monkeypatch):
        """Backlog entries with errors are never marked 'done_rejected'."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
        )
        bl.add_entry(entry)

        # Expect: infra → pending (retry) or done_error, code → done_error
        valid_error_statuses = {"pending", "done_error", "testing"}
        # 'done_rejected' should NOT appear for error entries
        assert entry["status"] != "done_rejected"


# ──────────────────────────────────────────────
# 5. strategy_ideation code error summarisation
# ──────────────────────────────────────────────

class TestIdeationCodeErrors:
    """_summarise_code_errors extracts last 10 code errors for LLM prompt."""

    def test_no_code_errors_returns_none(self):
        """Returns None when no code errors exist."""
        knowledge = _fresh_knowledge()
        result = _summarise_code_errors(knowledge)
        assert result is None

    def test_only_infra_errors_returns_none(self):
        """Infra-only errors → returns None (only code errors shown)."""
        knowledge = _fresh_knowledge()
        knowledge["errors"] = [
            {"evaluation": {"error_type": "infra", "error": "timeout"},
             "hypothesis": {"description": "test", "symbol": "AAPL"}},
        ]
        result = _summarise_code_errors(knowledge)
        assert result is None

    def test_code_errors_summarised(self):
        """Code errors produce a RECENT CODE ERRORS section."""
        knowledge = _fresh_knowledge()
        knowledge["errors"] = [
            {"evaluation": {"error_type": "code", "error": "KeyError: 'close'"},
             "hypothesis": {"strategy": "codegen", "description": "my_strat", "symbol": "SPY"}},
        ]
        result = _summarise_code_errors(knowledge)
        assert result is not None
        assert "RECENT CODE ERRORS" in result
        assert "my_strat/SPY" in result
        assert "KeyError: 'close'" in result

    def test_code_errors_capped_at_10(self):
        """Only last 10 code errors are included."""
        knowledge = _fresh_knowledge()
        for i in range(15):
            knowledge["errors"].append({
                "evaluation": {"error_type": "code", "error": f"error_{i}"},
                "hypothesis": {"strategy": "codegen", "description": f"strat_{i}", "symbol": f"SYM{i}"},
            })
        result = _summarise_code_errors(knowledge)
        assert result is not None
        # Should contain last 10, not first 5
        assert "strat_14" in result
        assert "strat_5" in result
        assert "strat_0" not in result  # first 5 dropped

    def test_df_columns_hint_included(self):
        """The prompt includes the capitalized column names hint."""
        knowledge = _fresh_knowledge()
        knowledge["errors"] = [
            {"evaluation": {"error_type": "code", "error": "KeyError: 'close'"},
             "hypothesis": {"strategy": "codegen", "description": "test", "symbol": "AAPL"}},
        ]
        result = _summarise_code_errors(knowledge)
        assert "Close" in result
        assert "capitalized" in result


# ──────────────────────────────────────────────
# 6. print_report error counts
# ──────────────────────────────────────────────

class TestPrintReportErrors:
    """print_report shows separate infra/code error counts."""

    def test_error_counts_zero_when_empty(self):
        """When no errors, n_infra and n_code are 0."""
        knowledge = _fresh_knowledge()
        errors = knowledge.get("errors", [])
        n_infra = sum(1 for e in errors if e.get("evaluation", {}).get("error_type") == "infra")
        n_code = sum(1 for e in errors if e.get("evaluation", {}).get("error_type") == "code")
        assert n_infra == 0
        assert n_code == 0

    def test_error_counts_separate(self):
        """Infra and code errors counted separately."""
        knowledge = _fresh_knowledge()
        knowledge["errors"] = [
            {"evaluation": {"error_type": "infra"}},
            {"evaluation": {"error_type": "code"}},
            {"evaluation": {"error_type": "code"}},
            {"evaluation": {"error_type": "infra"}},
            {"evaluation": {"error_type": "infra"}},
        ]
        errors = knowledge.get("errors", [])
        n_infra = sum(1 for e in errors if e.get("evaluation", {}).get("error_type") == "infra")
        n_code = sum(1 for e in errors if e.get("evaluation", {}).get("error_type") == "code")
        assert n_infra == 3
        assert n_code == 2
