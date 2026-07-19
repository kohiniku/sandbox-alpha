"""
Tests for backlog consumption in autonomous_loop.py.
No network — all external calls are mocked.
"""
import base64
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backlog import Backlog, make_param_entry, make_code_entry
from autonomous_loop import (
    load_knowledge,
    save_knowledge,
    evaluate_result,
    _params_within_cluster,
    STRATEGY_TEMPLATES,
    KNOWLEDGE_FILE,
)


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def _make_synthetic_backtest_result(val_sharpe, val_return, val_max_dd, val_days,
                                     holdout_sharpe, holdout_return, holdout_days,
                                     strategy="sma_crossover", symbol="AAPL",
                                     params=None):
    """Mirrors the helper in test_overfitting_guards.py."""
    if params is None:
        params = {"fast_window": 10, "slow_window": 30}
    return {
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "in_sample": {
            "total_return_pct": val_return + 2, "sharpe_ratio": val_sharpe + 0.1,
            "max_drawdown_pct": val_max_dd + 2, "num_trades": 5,
            "avg_daily_return_pct": 0.05, "cost_bps": 5.0, "num_days": int(val_days * 0.6),
        },
        "out_of_sample": {
            "total_return_pct": val_return, "sharpe_ratio": val_sharpe,
            "max_drawdown_pct": val_max_dd, "num_trades": 5,
            "avg_daily_return_pct": 0.04, "cost_bps": 5.0, "num_days": val_days,
        },
        "holdout": {
            "total_return_pct": holdout_return, "sharpe_ratio": holdout_sharpe,
            "max_drawdown_pct": -8.0, "num_trades": 3,
            "avg_daily_return_pct": 0.02, "cost_bps": 5.0, "num_days": holdout_days,
        },
        "walkforward": {"enabled": True, "train_ratio": 0.6, "val_ratio": 0.2, "holdout_ratio": 0.2},
    }


def _known_good_result(symbol="AAPL", params=None):
    """A backtest result that passes all global gates."""
    if params is None:
        params = {"fast_window": 10, "slow_window": 30}
    return _make_synthetic_backtest_result(
        val_sharpe=1.2, val_return=15.0, val_max_dd=-10.0, val_days=252,
        holdout_sharpe=0.8, holdout_return=10.0, holdout_days=80,
        symbol=symbol, params=params,
    )


# ──────────────────────────────────────────────
# param entry consumed and marked done with attribution
# ──────────────────────────────────────────────

class TestParamEntryConsumption:
    def test_param_entry_marked_done_adopted(self, tmp_path, monkeypatch):
        """A param entry that passes all gates → done_adopted with source attribution."""
        # Set up a temp backlog and knowledge
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))

        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
            eval_plan={"extra_criteria": []},
        )
        bl.add_entry(entry)

        # Patch run_backtest to return a known-good result
        mock_result = _known_good_result()

        with patch("autonomous_loop.run_backtest", return_value=mock_result) as mock_bt:
            from autonomous_loop import run_loop
            # Use a temp knowledge file
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

            # run_backtest should have been called
            mock_bt.assert_called_once()

        # Check the backlog entry was marked done_adopted
        data = bl.load()
        entry = data["entries"][0]
        assert entry["status"] == "done_adopted"
        assert entry["result"]["verdict"] == "adopted"
        # source attribution in knowledge
        knowledge = json.loads(kf.read_text())
        # Should have a record in adopted with source attribution
        found = False
        for rec in knowledge.get("adopted", []):
            if rec.get("source"):
                found = True
                break
        assert found, "Knowledge record should have source attribution"

    def test_param_entry_marked_done_rejected(self, tmp_path, monkeypatch):
        """A param entry that fails gates → done_rejected."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="mean_reversion", symbol="TSLA",
            params={"window": 20, "threshold": 2.0},
            priority=0.5, source={"kind": "idea", "ref": "my_idea"},
            eval_plan={"extra_criteria": []},
        )
        bl.add_entry(entry)

        # Result that fails (negative sharpe, negative return, bad drawdown)
        bad_result = _make_synthetic_backtest_result(
            val_sharpe=-0.3, val_return=-12.0, val_max_dd=-40.0, val_days=252,
            holdout_sharpe=-0.5, holdout_return=-8.0, holdout_days=80,
            strategy="mean_reversion", symbol="TSLA", params={"window": 20, "threshold": 2.0},
        )

        with patch("autonomous_loop.run_backtest", return_value=bad_result):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_rejected"
        assert data["entries"][0]["result"]["verdict"] == "rejected"


# ──────────────────────────────────────────────
# code entry → /run_code request body correct
# ──────────────────────────────────────────────

class TestCodeEntryRunCode:
    SAMPLE_CODE = "import numpy as np\nimport pandas as pd\n\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)\n"

    def test_run_code_request_body_correct(self, tmp_path, monkeypatch):
        """Code entry POSTs to /run_code with base64-encoded code and symbol."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = make_code_entry(
            name="test_strategy", description="a test",
            code=self.SAMPLE_CODE, symbol="SPY",
            priority=0.88, source={"kind": "idea", "ref": "test"},
            eval_plan={"extra_criteria": []},
        )
        bl.add_entry(entry)

        # Simulate a response from /run_code that passes gates
        mock_response = _known_good_result(symbol="SPY", params={})

        # We need to mock urllib.request.urlopen
        def mock_urlopen(req, timeout=180):
            # Verify the request body
            body = json.loads(req.data)
            assert "code_b64" in body
            assert body["symbol"] == "SPY"
            decoded = base64.b64decode(body["code_b64"]).decode("utf-8")
            assert self.SAMPLE_CODE in decoded

            # Return a mock response
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"

    def test_run_code_metrics_evaluated(self, tmp_path, monkeypatch):
        """Code entry result flows through evaluate_result gates."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = make_code_entry(
            name="test_strategy", description="a test",
            code=self.SAMPLE_CODE, symbol="GOOGL",
            priority=0.7, source={"kind": "idea", "ref": "test"},
            eval_plan={"extra_criteria": []},
        )
        bl.add_entry(entry)

        # Result that barely passes
        mock_response = _make_synthetic_backtest_result(
            val_sharpe=1.2, val_return=8.0, val_max_dd=-15.0, val_days=252,
            holdout_sharpe=0.3, holdout_return=5.0, holdout_days=80,
            symbol="GOOGL", params={},
        )

        def mock_urlopen(req, timeout=180):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"
        assert data["entries"][0]["result"]["verdict"] == "adopted"


# ──────────────────────────────────────────────
# extra_criteria enforcement
# ──────────────────────────────────────────────

class TestExtraCriteria:
    def test_extra_criteria_enforced_total_return(self, tmp_path, monkeypatch):
        """Candidate passes global gates but fails 'total_return_pct >= 30' → rejected."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
            eval_plan={"extra_criteria": ["total_return_pct >= 30"]},
        )
        bl.add_entry(entry)

        # Result that passes global gates but has total_return_pct = 15 (< 30)
        mock_result = _known_good_result()  # val_return=15.0, passes gates

        with patch("autonomous_loop.run_backtest", return_value=mock_result):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_rejected"
        assert "total_return_pct" in str(data["entries"][0]["result"]).lower()

    def test_extra_criteria_passes_when_met(self, tmp_path, monkeypatch):
        """Candidate passes global gates AND extra_criteria → adopted."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
            eval_plan={"extra_criteria": ["sharpe_ratio >= 1.0"]},
        )
        bl.add_entry(entry)

        mock_result = _known_good_result()  # val_sharpe=1.2 >= 1.0

        with patch("autonomous_loop.run_backtest", return_value=mock_result):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"

    def test_extra_criteria_enforced_max_drawdown(self, tmp_path, monkeypatch):
        """Extra criterion: max_drawdown_pct >= -5 with val_max_dd=-10 → rejected."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="momentum", symbol="NVDA",
            params={"lookback": 20, "hold_period": 5},
            priority=0.3, source={"kind": "idea", "ref": "test"},
            eval_plan={"extra_criteria": ["max_drawdown_pct >= -5"]},
        )
        bl.add_entry(entry)

        mock_result = _make_synthetic_backtest_result(
            val_sharpe=1.5, val_return=20.0, val_max_dd=-10.0, val_days=252,
            holdout_sharpe=1.0, holdout_return=15.0, holdout_days=80,
            strategy="momentum", symbol="NVDA", params={"lookback": 20, "hold_period": 5},
        )

        with patch("autonomous_loop.run_backtest", return_value=mock_result):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_rejected"


# ──────────────────────────────────────────────
# unparseable criterion ignored with warning
# ──────────────────────────────────────────────

class TestUnparseableCriteria:
    def test_unparseable_criteria_ignored(self, tmp_path, monkeypatch):
        """Unparseable criterion logged as warning, entry still evaluated normally."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
            eval_plan={"extra_criteria": ["garbage_nonsense", "also_broken > xyz"]},
        )
        bl.add_entry(entry)

        mock_result = _known_good_result()

        with patch("autonomous_loop.run_backtest", return_value=mock_result):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        # Should still be adopted since global gates pass and unparseable criteria are ignored
        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"

    def test_mixed_parseable_and_unparseable(self, tmp_path, monkeypatch):
        """One unparseable (ignored) + one parseable that passes → adopted."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
            eval_plan={"extra_criteria": ["broken stuff here", "sharpe_ratio >= 0.5"]},
        )
        bl.add_entry(entry)

        mock_result = _known_good_result()  # sharpe=1.2 >= 0.5

        with patch("autonomous_loop.run_backtest", return_value=mock_result):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"


# ──────────────────────────────────────────────
# empty backlog → fallback to hypothesis path
# ──────────────────────────────────────────────

class TestEmptyBacklogFallback:
    def test_empty_backlog_falls_back_to_hypothesis(self, tmp_path, monkeypatch):
        """When backlog is empty, existing LLM/random hypothesis path is used."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))

        # Create empty backlog
        bl = Backlog(str(backlog_path))
        # Ensure it's empty
        data = bl.load()
        data["entries"] = []
        bl.save(data)

        mock_result = _known_good_result()

        with patch("autonomous_loop.run_backtest", return_value=mock_result) as mock_bt:
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

            # run_backtest still called (from hypothesis fallback path)
            mock_bt.assert_called_once()

        # Background should still be empty (no entry consumed)
        data = bl.load()
        pending = [e for e in data["entries"] if e["status"] == "pending"]
        assert len(pending) == 0  # No backlog entries were added


# ──────────────────────────────────────────────
# duplicate code_hash skipped without HTTP call
# ──────────────────────────────────────────────

class TestDuplicateCodeHash:
    SAMPLE_CODE = "import numpy as np\nimport pandas as pd\n\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)\n"

    def test_duplicate_code_hash_skipped(self, tmp_path, monkeypatch):
        """Code entry with same code_hash+symbol in knowledge → done_rejected, no HTTP."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        monkeypatch.setenv("USE_LLM_HYPOTHESIS", "0")

        bl = Backlog(str(backlog_path))
        entry = make_code_entry(
            name="dup_test", description="duplicate",
            code=self.SAMPLE_CODE, symbol="SPY",
            priority=0.9, source={"kind": "idea", "ref": "test"},
            eval_plan={"extra_criteria": []},
        )
        bl.add_entry(entry)

        # Pre-populate knowledge with the same code_hash
        code_hash = hashlib.sha256(self.SAMPLE_CODE.encode("utf-8")).hexdigest()

        from autonomous_loop import run_loop
        kf = tmp_path / "knowledge.json"
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
        monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
        (tmp_path / "results").mkdir(exist_ok=True)

        # Write knowledge with existing code_hash in adopted
        knowledge = {
            "tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
            "superseded": [], "families": {}, "iterations": 0,
        }
        # Add a record with the same code_hash + symbol in tested history
        knowledge["adopted"].append({
            "hypothesis": {
                "strategy": "codegen", "symbol": "SPY", "params": {},
                "description": "pre-existing code strategy on SPY",
                "generated_at": "2026-01-01T00:00:00",
            },
            "evaluation": {
                "verdict": "adopted", "code_hash": code_hash,
                "sharpe_ratio": 0.8, "total_return_pct": 10.0,
            },
            "tested_at": "2026-01-01T00:00:00",
        })
        kf.write_text(json.dumps(knowledge, indent=2))

        # urllib should NOT be called (code entry skipped; fallback uses run_backtest mock)
        mock_result = _known_good_result(symbol="SPY")
        with patch("urllib.request.urlopen") as mock_urlopen:
            with patch("autonomous_loop.run_backtest", return_value=mock_result):
                run_loop(1)
            mock_urlopen.assert_not_called()

        # Check backlog entry status
        data = bl.load()
        entry = data["entries"][0]
        assert entry["status"] == "done_rejected"
        assert "duplicate_code" in str(entry.get("result", {}))
        assert "コードハッシュ重複" in str(entry.get("result", {}))


# ──────────────────────────────────────────────
# code entry with unset SANDBOX_RUNNER_URL → return to pending
# ──────────────────────────────────────────────

class TestCodeEntryNoSandboxUrl:
    SAMPLE_CODE = "import numpy as np\nimport pandas as pd\n\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)\n"

    def test_code_entry_no_sandbox_url_returned_to_pending(self, tmp_path, monkeypatch):
        """Code entry with SANDBOX_RUNNER_URL unset → entry returned to pending."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        # Explicitly unset SANDBOX_RUNNER_URL
        monkeypatch.delenv("SANDBOX_RUNNER_URL", raising=False)

        bl = Backlog(str(backlog_path))
        entry = make_code_entry(
            name="unsandboxed_test", description="no runner",
            code=self.SAMPLE_CODE, symbol="MSFT",
            priority=0.7, source={"kind": "idea", "ref": "test"},
            eval_plan={"extra_criteria": []},
        )
        bl.add_entry(entry)

        mock_result = _known_good_result()

        # Should fall back to hypothesis path since code entry can't run
        with patch("autonomous_loop.run_backtest", return_value=mock_result):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        # The code entry should be back to "pending" (not consumed)
        data = bl.load()
        entry = data["entries"][0]
        assert entry["status"] == "pending"
        # The iteration should have fallen back and generated a random hypothesis
