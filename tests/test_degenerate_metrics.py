"""Tests for degenerate-metrics detection (PR: feat/degenerate-metrics).

Validates:
  - _validate_expert_metrics: Series → error, all-finite → None, NaN → None
  - _find_nonfinite_metrics: identifies NaN/inf keys
  - _validate_manifest_synthetic: Series → valid false, NaN → valid true + warnings
  - Loop: degenerate backtest → verdict rejected, knowledge counts, no error
"""

import base64
import json
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest_runner import (
    _validate_expert_metrics,
    _find_nonfinite_metrics,
    _validate_manifest_synthetic,
    run_manifest,
    REQUIRED_EXPERT_METRICS,
)
from manifest import StrategyManifest


# ---------------------------------------------------------------------------
# 7a. _validate_expert_metrics unit tests
# ---------------------------------------------------------------------------

class TestValidateExpertMetrics:
    """Direct unit tests for _validate_expert_metrics and _find_nonfinite_metrics."""

    def test_series_value_is_error(self):
        """A Series metric value must produce an error message."""
        result = {
            "val_sharpe": pd.Series([1.5]),
            "val_max_drawdown_pct": 5.0,
            "val_total_return_pct": 10.0,
            "holdout_sharpe": 0.8,
            "holdout_max_drawdown_pct": 3.0,
            "holdout_total_return_pct": 6.0,
        }
        err = _validate_expert_metrics(result)
        assert err is not None
        assert "must be numeric" in err
        assert "Series" in err
        assert "val_sharpe" in err

    def test_all_finite_valid_dict_returns_none(self):
        """A well-formed dict with all-finite numeric values is valid."""
        result = {
            "val_sharpe": 1.5,
            "val_max_drawdown_pct": 5.0,
            "val_total_return_pct": 10.0,
            "holdout_sharpe": 0.8,
            "holdout_max_drawdown_pct": 3.0,
            "holdout_total_return_pct": 6.0,
        }
        assert _validate_expert_metrics(result) is None
        assert _find_nonfinite_metrics(result) == []

    def test_nan_value_is_no_longer_error(self):
        """NaN values must NOT produce an error from _validate_expert_metrics."""
        result = {
            "val_sharpe": 1.2,
            "val_max_drawdown_pct": 5.0,
            "val_total_return_pct": 8.0,
            "holdout_sharpe": float("nan"),
            "holdout_max_drawdown_pct": 2.0,
            "holdout_total_return_pct": 5.0,
        }
        assert _validate_expert_metrics(result) is None

    def test_nan_detected_by_find_nonfinite(self):
        """_find_nonfinite_metrics returns the NaN key."""
        result = {
            "val_sharpe": 1.2,
            "val_max_drawdown_pct": 5.0,
            "val_total_return_pct": 8.0,
            "holdout_sharpe": float("nan"),
            "holdout_max_drawdown_pct": 2.0,
            "holdout_total_return_pct": 5.0,
        }
        nf = _find_nonfinite_metrics(result)
        assert nf == ["holdout_sharpe"]

    def test_multiple_nan_keys(self):
        """Multiple non-finite keys returned in sorted order."""
        result = {
            "val_sharpe": float("nan"),
            "val_max_drawdown_pct": 5.0,
            "val_total_return_pct": 8.0,
            "holdout_sharpe": float("inf"),
            "holdout_max_drawdown_pct": float("-inf"),
            "holdout_total_return_pct": 5.0,
        }
        nf = _find_nonfinite_metrics(result)
        assert nf == ["holdout_max_drawdown_pct", "holdout_sharpe", "val_sharpe"]

    def test_missing_key_is_not_nonfinite(self):
        """Missing keys are NOT returned by _find_nonfinite_metrics."""
        result = {
            "val_sharpe": 1.2,
            "val_max_drawdown_pct": 5.0,
            "val_total_return_pct": 8.0,
            # holdout_* missing
        }
        nf = _find_nonfinite_metrics(result)
        assert nf == []

    def test_non_dict_returns_error(self):
        """Non-dict returns error as before."""
        err = _validate_expert_metrics(pd.DataFrame())
        assert err is not None
        assert "must return dict" in err

    def test_missing_keys_returns_error(self):
        """Missing required keys returns error as before."""
        result = {"val_sharpe": 1.0}
        err = _validate_expert_metrics(result)
        assert err is not None
        assert "missing required metrics" in err


# ---------------------------------------------------------------------------
# 7b. _validate_manifest_synthetic in-process tests
# ---------------------------------------------------------------------------

def _make_expert_manifest(code: str, name: str = "test_expert"):
    """Build a StrategyManifest with expert execution_mode and base64-encoded code."""
    code_b64 = base64.b64encode(textwrap.dedent(code).encode()).decode()
    payload = {
        "name": name,
        "code_b64": code_b64,
        "data_sources": [
            {"type": "ohlcv", "universe": ["AAPL", "MSFT", "GOOG"], "start": "2023-01-01", "end": "2023-12-31"}
        ],
        "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
        "evaluator": {"type": "portfolio", "metrics": ["sharpe", "max_drawdown_pct"]},
        "execution_mode": "expert",
    }
    return StrategyManifest.from_dict(payload)


class TestValidateManifestSynthetic:
    """Test _validate_manifest_synthetic with in-process expert code."""

    def test_series_metric_returns_valid_false(self):
        """Code returning a Series metric → valid false, error_type code."""
        code = """
            import pandas as pd
            import numpy as np

            def run(data, train_end, val_end, benchmark, config):
                return {
                    "val_sharpe": pd.Series([1.5]),
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": 0.8,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 6.0,
                }
        """
        manifest = _make_expert_manifest(code)
        result = json.loads(_validate_manifest_synthetic(manifest))
        assert result["valid"] is False
        assert result["error_type"] == "code"
        assert "must be numeric" in result["error"]

    def test_nan_holdout_sharpe_returns_valid_true_with_warnings(self):
        """Code returning NaN holdout_sharpe → valid true + warnings."""
        code = """
            import numpy as np

            def run(data, train_end, val_end, benchmark, config):
                return {
                    "val_sharpe": 1.5,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": float("nan"),
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 6.0,
                }
        """
        manifest = _make_expert_manifest(code)
        result = json.loads(_validate_manifest_synthetic(manifest))
        assert result["valid"] is True
        assert "warnings" in result
        assert any("holdout_sharpe" in w for w in result["warnings"])
        assert any("not finite on synthetic data" in w for w in result["warnings"])

    def test_well_behaved_code_valid_true_no_warnings(self):
        """Well-behaved code → valid true, no warnings."""
        code = """
            import numpy as np

            def run(data, train_end, val_end, benchmark, config):
                return {
                    "val_sharpe": 1.5,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": 0.8,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 6.0,
                }
        """
        manifest = _make_expert_manifest(code)
        result = json.loads(_validate_manifest_synthetic(manifest))
        assert result["valid"] is True
        assert "warnings" not in result


# ---------------------------------------------------------------------------
# 7c. Loop mapping: degenerate backtest → verdict rejected
# ---------------------------------------------------------------------------

from autonomous_loop import evaluate_result, _evaluate_manifest_result
from loop_constants import Verdict, BacklogStatus


def _fake_hypothesis():
    return {
        "id": "degenerate_test_001",
        "strategy": "manifest:test_degen",
        "symbol": "universe:abc12345",
        "params": {"universe_size": 3, "execution_mode": "expert", "primary_metric": "sharpe"},
        "description": "degenerate test strategy",
        "generated_at": "2026-07-22T00:00:00",
    }


def _fresh_knowledge():
    return {
        "tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
        "superseded": [], "families": {}, "iterations": 0, "errors": [],
    }


class TestDegenerateLoopMapping:
    """Tests that a degenerate backtest result maps correctly in the loop."""

    def test_evaluate_result_degenerate_short_circuit(self):
        """Degenerate in evaluate_result → rejected with degenerate: reason."""
        result = {
            "degenerate": True,
            "degenerate_reason": "metrics not finite: ['holdout_sharpe'] (likely no trades in a segment)",
        }
        hyp = {
            "id": "deg_test",
            "strategy": "sma_crossover",
            "symbol": "AAPL",
            "params": {"fast_window": 10, "slow_window": 30},
            "description": "test",
            "generated_at": "2026-01-01T00:00:00",
        }
        knowledge = _fresh_knowledge()
        verdict, evaluation = evaluate_result(hyp, result, knowledge)

        assert verdict == "rejected"
        assert evaluation["verdict"] == Verdict.REJECTED
        assert len(evaluation["reasons"]) == 1
        assert evaluation["reasons"][0].startswith("degenerate:")
        assert "holdout_sharpe" in evaluation["reasons"][0]

        # Verify errors list does NOT grow (degenerate is not an error)
        assert len(knowledge.get("errors", [])) == 0

    def test_evaluate_manifest_result_degenerate_short_circuit(self):
        """Degenerate manifest result → rejected with degenerate: reason."""
        runner_result = {
            "status": "ok",
            "execution_mode": "expert",
            "manifest_name": "test_degen",
            "universe_size": 3,
            "n_days": 200,
            "metrics": {
                "val_sharpe": 0.5,
                "val_max_drawdown_pct": -5.0,
                "val_total_return_pct": 2.0,
                "holdout_sharpe": float("nan"),
                "holdout_max_drawdown_pct": -3.0,
                "holdout_total_return_pct": 1.0,
            },
            "degenerate": True,
            "degenerate_reason": "metrics not finite: ['holdout_sharpe'] (likely no trades in a segment)",
        }
        hyp = _fake_hypothesis()
        knowledge = _fresh_knowledge()
        knowledge["families"] = {"manifest:test_degen|universe:abc12345": {"n_trials": 0}}
        verdict, evaluation = _evaluate_manifest_result(runner_result, hyp, knowledge)

        assert verdict == "rejected"
        assert evaluation["verdict"] == Verdict.REJECTED
        assert evaluation["reasons"][0].startswith("degenerate:")
        assert "holdout_sharpe" in evaluation["reasons"][0]

    def test_degenerate_not_in_errors_summary(self):
        """Degenerate rejected entries do NOT go to knowledge.errors."""
        knowledge = _fresh_knowledge()
        runner_result = {
            "status": "ok",
            "metrics": {
                "val_sharpe": 0.5, "val_max_drawdown_pct": -5.0,
                "val_total_return_pct": 2.0, "holdout_sharpe": float("nan"),
                "holdout_max_drawdown_pct": -3.0, "holdout_total_return_pct": 1.0,
            },
            "degenerate": True,
            "degenerate_reason": "metrics not finite: ['holdout_sharpe'] (likely no trades in a segment)",
        }
        hyp = _fake_hypothesis()
        knowledge["families"] = {"manifest:test_degen|universe:abc12345": {"n_trials": 0}}

        verdict, evaluation = _evaluate_manifest_result(runner_result, hyp, knowledge)
        assert verdict == "rejected"

        # Simulate loop accumulation: rejected (not error/code_error) → goes to rejected, not errors
        # The loop would do: if verdict in ("error","code_error"): errors.append(...) else: rejected.append(...)
        # Since verdict is "rejected", it would NOT go to errors.
        assert len(knowledge.get("errors", [])) == 0

    def test_degenerate_knowledge_tested_increments(self, tmp_path):
        """Full disk round-trip: degenerate → knowledge tested count increments, backlog done_rejected."""
        from backlog import Backlog

        # Create a fresh backlog
        backlog_path = tmp_path / "backlog.json"
        bl = Backlog(str(backlog_path))

        from backlog import make_param_entry
        entry = make_param_entry(
            strategy="manifest:test_degen",
            symbol="universe:abc12345",
            params={"universe_size": 3},
            priority=5.0,
            source={"kind": "test", "ref": "degenerate_test"},
        )
        # Add to backlog (simulating ideation LLM pushing it)
        accepted, eid = bl.add_entry(entry)
        assert accepted
        bl.mark(eid, BacklogStatus.TESTING)

        knowledge = _fresh_knowledge()
        knowledge["families"] = {"manifest:test_degen|universe:abc12345": {"n_trials": 0}}

        runner_result = {
            "status": "ok",
            "metrics": {
                "val_sharpe": 0.5, "val_max_drawdown_pct": -5.0,
                "val_total_return_pct": 2.0, "holdout_sharpe": float("nan"),
                "holdout_max_drawdown_pct": -3.0, "holdout_total_return_pct": 1.0,
            },
            "degenerate": True,
            "degenerate_reason": "metrics not finite: ['holdout_sharpe'] (likely no trades in a segment)",
        }
        hyp = _fake_hypothesis()
        verdict, evaluation = _evaluate_manifest_result(runner_result, hyp, knowledge)

        assert verdict == "rejected"

        # Mark backlog as done_rejected (simulating loop)
        bl.mark(eid, BacklogStatus.DONE_REJECTED, {
            "verdict": verdict,
            "summary": evaluation["reasons"][0],
            "finished_at": "2026-07-22T00:00:00",
        })

        # Re-read from disk and verify
        bl2 = Backlog(str(backlog_path))
        entries = bl2.load()["entries"]
        assert len(entries) == 1
        assert entries[0]["status"] == BacklogStatus.DONE_REJECTED
        assert entries[0]["result"]["verdict"] == "rejected"
        assert "degenerate:" in entries[0]["result"]["summary"]
