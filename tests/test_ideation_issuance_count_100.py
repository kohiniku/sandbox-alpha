"""
Regression tests for Issue #100 — ideation delivery '発行 N' counts pre-preflight proposals.

Root cause (strategy_ideation.py):
  * ``_save_ideation_log_v3`` persisted ``stage3_selection.n_proposed =
    len(manifest_dicts)`` where ``manifest_dicts`` is populated BEFORE the
    preflight loop drops manifests. The accepted count (``manifest_count``) was
    only printed to stdout, never persisted to the log the delivery agent reads.
  * Consequently the delivered headline "発行 N" keyed off the pre-preflight
    proposal count and contradicted the run's own ``manifest_count=0``.

These tests assert the post-fix contract:
  1. ``stage3_selection.n_accepted`` is persisted and equals the number of
     entries actually accepted into the backlog (0 when preflight drops all).
  2. Per-proposal ``preflight_status`` / ``accepted`` is persisted so the
     delivery can distinguish "proposed" from "actually entered the backlog".
  3. A machine-readable ``ISSUANCE_FULL_FAILURE`` marker is emitted when
     ``n_proposed > 0`` but ``n_accepted == 0`` (a full-issuance-failure cycle).

All LLM / runner / preflight calls are mocked — no network.
"""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_ideation import _run_ideation_v3
from autonomous_loop import STRATEGY_TEMPLATES
from manifest import StrategyManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _structured_manifest_dict(name="test_strategy"):
    """Minimal valid structured manifest dict (mirrors test_ideation_v3)."""
    code = b"def generate_signals(df):\n    import pandas as pd\n    return pd.Series(0, index=df.index)\n"
    return {
        "name": name,
        "code_b64": base64.b64encode(code).decode("ascii"),
        "data_sources": [
            {"type": "ohlcv", "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "start": "2020-01-01"}
        ],
        "model_artifacts": [],
        "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
        "evaluator": {"type": "portfolio", "metrics": ["sharpe", "ir", "turnover"], "benchmark": "SPY"},
        "execution_mode": "structured",
        "priority": 0.85,
        "source": {"kind": "paper", "ref": "test-paper.md"},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def issuance_env(monkeypatch):
    """Redirect ideation log + checkpoint dirs to a temp dir; dummy keys."""
    monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "DUMMY_KEY")
    monkeypatch.setenv("DUMMY_KEY", "noop")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        log_dir = td_path / "ideation_logs"
        log_dir.mkdir()
        monkeypatch.setattr("strategy_ideation._IDEATION_LOG_DIR", log_dir)
        monkeypatch.setattr("strategy_ideation._checkpoint_path", lambda: log_dir / "checkpoint.json")
        yield td_path, log_dir


@pytest.fixture
def v2_knowledge():
    return {
        "rejected": [],
        "adopted": [],
        "families": {
            "momentum|AAPL": {"n_trials": 5, "best_val_sharpe": 0.8},
        },
        "near_misses": [],
        "errors": [],
        "iterations": 10,
    }


def _run_v3_with_preflight(si, manifests, knowledge, preflight_result, backlog, capsys=None):
    """Drive _run_ideation_v3 with mocked stages; return the written v3 log dict."""
    brainstorm_ideas = [
        {"name": m["name"], "type": "code", "family": "momentum|AAPL",
         "one_line_rationale": "test"} for m in manifests
    ]
    debate_results = [
        {"index": i, "survive": True, "attack": "ok", "rebuttal": "ok",
         "variation": "", "reason": "ok"} for i in range(len(manifests))
    ]
    judge_report = [{"index": i, "survive": True} for i in range(len(manifests))]

    def mock_select_v3(*args, **kwargs):
        return [StrategyManifest.from_dict(m) for m in manifests], False

    with patch.object(si, "_stage_brainstorm", return_value=brainstorm_ideas), \
         patch.object(si, "_stage_debate", return_value=(debate_results, judge_report)), \
         patch.object(si, "_stage_select_v3", side_effect=mock_select_v3), \
         patch.object(si, "_syntax_preflight_with_fix", return_value=preflight_result), \
         patch.object(si, "_runtime_preflight_with_fix", return_value=(True, "", 0)), \
         patch.object(si, "_preflight_manifest", return_value=(None, "skipped")), \
         patch.object(si, "_get_killed_families", return_value={}):
        result = _run_ideation_v3(knowledge, STRATEGY_TEMPLATES, [], backlog, 5, dry_run=False)
    return result


def _load_v3_log(log_dir):
    logs = sorted(log_dir.glob("*_v3.json"))
    assert logs, "no v3 ideation log was written"
    return json.loads(logs[-1].read_text())


# ---------------------------------------------------------------------------
# Tests — full issuance failure (proposed > 0, accepted == 0)
# ---------------------------------------------------------------------------

class TestFullIssuanceFailure:
    """All manifests proposed but dropped by preflight → 0 accepted."""

    def test_log_persists_n_accepted_zero(self, issuance_env, v2_knowledge, capsys):
        """stage3_selection.n_accepted must be persisted and == 0."""
        import strategy_ideation as si
        _, log_dir = issuance_env

        manifests = [_structured_manifest_dict("strat_a"), _structured_manifest_dict("strat_b")]
        # Syntax preflight fails for every manifest → all dropped
        _run_v3_with_preflight(si, manifests, v2_knowledge,
                               preflight_result=(False, "empty code_b64", 0),
                               backlog=MagicMock())

        log = _load_v3_log(log_dir)
        sel = log["stage3_selection"]
        assert sel["n_proposed"] == 2
        # THE fix: accepted count persisted (was absent pre-fix)
        assert sel.get("n_accepted") == 0

    def test_marker_emitted_on_full_failure(self, issuance_env, v2_knowledge, capsys):
        """ISSUANCE_FULL_FAILURE marker emitted when proposed>0 and accepted==0."""
        import strategy_ideation as si
        _, log_dir = issuance_env

        manifests = [_structured_manifest_dict("strat_a"), _structured_manifest_dict("strat_b")]
        _run_v3_with_preflight(si, manifests, v2_knowledge,
                               preflight_result=(False, "empty code_b64", 0),
                               backlog=MagicMock())

        out = capsys.readouterr().out
        assert "ISSUANCE_FULL_FAILURE" in out

    def test_per_proposal_status_persisted(self, issuance_env, v2_knowledge, capsys):
        """Each proposal records preflight_status + accepted=False when dropped."""
        import strategy_ideation as si
        _, log_dir = issuance_env

        manifests = [_structured_manifest_dict("strat_a"), _structured_manifest_dict("strat_b")]
        _run_v3_with_preflight(si, manifests, v2_knowledge,
                               preflight_result=(False, "empty code_b64", 0),
                               backlog=MagicMock())

        log = _load_v3_log(log_dir)
        proposals = log["stage3_selection"].get("proposals")
        assert proposals is not None, "per-proposal outcomes not persisted"
        assert len(proposals) == 2
        assert all(p["accepted"] is False for p in proposals)


# ---------------------------------------------------------------------------
# Tests — happy path (proposed == accepted)
# ---------------------------------------------------------------------------

class TestHappyPathIssuance:
    """All manifests pass preflight and enter the backlog."""

    def test_log_persists_n_accepted_equals_proposed(self, issuance_env, v2_knowledge, capsys):
        import strategy_ideation as si
        _, log_dir = issuance_env

        manifests = [_structured_manifest_dict("strat_a"), _structured_manifest_dict("strat_b")]
        backlog = MagicMock()
        backlog.add_entry.return_value = (True, "entry_1")
        result = _run_v3_with_preflight(si, manifests, v2_knowledge,
                                        preflight_result=(True, "", 0),
                                        backlog=backlog)

        log = _load_v3_log(log_dir)
        sel = log["stage3_selection"]
        assert sel["n_proposed"] == 2
        assert sel.get("n_accepted") == 2
        # Both proposals flagged accepted
        assert all(p["accepted"] is True for p in sel["proposals"])

    def test_no_marker_on_partial_or_full_success(self, issuance_env, v2_knowledge, capsys):
        """No ISSUANCE_FULL_FAILURE when at least one manifest is accepted."""
        import strategy_ideation as si
        _, log_dir = issuance_env

        manifests = [_structured_manifest_dict("strat_a")]
        backlog = MagicMock()
        backlog.add_entry.return_value = (True, "entry_1")
        _run_v3_with_preflight(si, manifests, v2_knowledge,
                               preflight_result=(True, "", 0),
                               backlog=backlog)

        out = capsys.readouterr().out
        assert "ISSUANCE_FULL_FAILURE" not in out
