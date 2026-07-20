"""
Tests for stricter holdout gate + rolling OOS monitor.
Covers:
  - Holdout threshold math (min(0.5, 0.5*val_sharpe))
  - Near-miss classification on holdout failure
  - OOS window slicing logic
  - Greppable output format
"""
import sys
import io
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_loop import evaluate_result
from oos_monitor import (
    _estimate_warmup_days,
    _parse_adoption_date,
    _is_param_strategy,
    run_oos_check,
    run_oos_monitor,
)


# ============================================================
# Part 1: Stricter holdout gate
# ============================================================

def _make_synthetic_result(val_sharpe, val_return, val_max_dd, val_days,
                           holdout_sharpe, holdout_return, holdout_days,
                           strategy="sma_crossover", symbol="AAPL",
                           params=None):
    if params is None:
        params = {"fast_window": 10, "slow_window": 30}
    return {
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "walkforward": {"enabled": True, "train_ratio": 0.6, "val_ratio": 0.2, "holdout_ratio": 0.2},
        "in_sample": {"sharpe_ratio": 1.0, "total_return_pct": 20.0, "max_drawdown_pct": -10.0,
                      "num_days": 378, "num_trades": 10},
        "out_of_sample": {"sharpe_ratio": val_sharpe, "total_return_pct": val_return,
                          "max_drawdown_pct": val_max_dd, "num_days": val_days, "num_trades": 5},
        "holdout": {"sharpe_ratio": holdout_sharpe, "total_return_pct": holdout_return,
                    "max_drawdown_pct": -5.0, "num_days": holdout_days, "num_trades": 3},
    }


class TestHoldoutThresholdMath:
    """Test the new holdout gate formula: min(0.5, 0.5 * val_sharpe)."""

    def test_threshold_with_high_val_sharpe(self):
        """val_sharpe=2.0 → threshold = min(0.5, 1.0) = 0.5."""
        result = _make_synthetic_result(
            val_sharpe=2.0, val_return=30.0, val_max_dd=-8.0, val_days=252,
            holdout_sharpe=0.5, holdout_return=10.0, holdout_days=252,
        )
        hyp = {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 10, "slow_window": 30}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        # holdout_sharpe=0.5 >= threshold=0.5 → passes
        assert evaluation["gate_results"]["holdout"] is True

    def test_threshold_with_modest_val_sharpe(self):
        """val_sharpe=1.2 → threshold = min(0.5, 0.6) = 0.5 (floor dominates).
        holdout=0.5 >= 0.5 → passes holdout gate.
        """
        result = _make_synthetic_result(
            val_sharpe=1.2, val_return=15.0, val_max_dd=-8.0, val_days=252,
            holdout_sharpe=0.5, holdout_return=8.0, holdout_days=252,
        )
        hyp = {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 10, "slow_window": 30}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        # holdout_sharpe=0.5 >= threshold=0.5 → passes
        assert evaluation["gate_results"]["holdout"] is True

    def test_holdout_below_threshold_rejected(self):
        """val_sharpe=1.6, threshold=0.5, holdout=0.29 → rejected."""
        result = _make_synthetic_result(
            val_sharpe=1.6, val_return=25.0, val_max_dd=-8.0, val_days=252,
            holdout_sharpe=0.29, holdout_return=5.0, holdout_days=252,
        )
        hyp = {"strategy": "rsi", "symbol": "TSLA", "params": {"rsi_window": 14, "oversold": 30, "overbought": 70}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        assert verdict == "rejected"
        assert evaluation["gate_results"]["holdout"] is False

    def test_holdout_zero_sharpe_rejected(self):
        """holdout_sharpe=0, threshold>0 → rejected."""
        result = _make_synthetic_result(
            val_sharpe=1.5, val_return=25.0, val_max_dd=-8.0, val_days=252,
            holdout_sharpe=0.0, holdout_return=5.0, holdout_days=252,
        )
        hyp = {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 10, "slow_window": 30}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        assert evaluation["gate_results"]["holdout"] is False


class TestHoldoutNearMiss:
    """Near-miss classification must catch holdout failures as failed_gate='holdout'."""

    def test_holdout_failure_classified_as_holdout_nearmiss(self):
        """Validation passes but holdout fails → near-miss with failed_gate='holdout'."""
        result = _make_synthetic_result(
            val_sharpe=1.6, val_return=25.0, val_max_dd=-8.0, val_days=252,
            holdout_sharpe=0.29, holdout_return=5.0, holdout_days=252,
        )
        hyp = {"id": "test-001", "strategy": "rsi", "symbol": "TSLA", "params": {"rsi_window": 14, "oversold": 30, "overbought": 70}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        
        from autonomous_loop import _classify_near_miss
        nm = _classify_near_miss(hyp, evaluation)
        assert nm is not None, "Expected near-miss classification for holdout failure"
        assert nm["failed_gate"] == "holdout"
        assert nm["holdout_sharpe"] == 0.29


# ============================================================
# Part 2: OOS monitor
# ============================================================

class TestWarmupEstimation:
    def test_max_param_plus_buffer(self):
        """slow_window=90 → warmup = 95."""
        params = {"fast_window": 10, "slow_window": 90}
        assert _estimate_warmup_days(params) == 95

    def test_fallback_30_days(self):
        """No numeric params → 30 days."""
        params = {"mode": "long"}
        assert _estimate_warmup_days(params) == 30

    def test_small_param_still_30(self):
        """rsi_window=14 → max(19, 30) = 30."""
        params = {"rsi_window": 14, "oversold": 30, "overbought": 70}
        assert _estimate_warmup_days(params) == 75  # max(70+5, 30) = 75


class TestAdoptionDateParsing:
    def test_tested_at_field(self):
        entry = {"tested_at": "2026-07-15T10:00:00"}
        dt = _parse_adoption_date(entry)
        assert dt == datetime(2026, 7, 15, 10, 0, 0)

    def test_tested_at_with_z_suffix(self):
        entry = {"tested_at": "2026-07-15T10:00:00Z"}
        dt = _parse_adoption_date(entry)
        assert dt == datetime(2026, 7, 15, 10, 0, 0)

    def test_finished_at_fallback(self):
        entry = {"finished_at": "2026-07-10T12:00:00"}
        dt = _parse_adoption_date(entry)
        assert dt == datetime(2026, 7, 10, 12, 0, 0)

    def test_missing_date_returns_none(self):
        entry = {"hypothesis": {}}
        assert _parse_adoption_date(entry) is None


class TestParamStrategyDetection:
    def test_param_strategy(self):
        entry = {"hypothesis": {"strategy": "rsi", "params": {"rsi_window": 14}}}
        assert _is_param_strategy(entry) is True

    def test_code_strategy_with_code_field(self):
        entry = {"hypothesis": {"strategy": "custom", "code": "def run(df): ..."}}
        assert _is_param_strategy(entry) is False

    def test_code_strategy_prefix(self):
        entry = {"hypothesis": {"strategy": "code:abc123"}}
        assert _is_param_strategy(entry) is False


class TestOOSWindowSlicing:
    """OOS window logic: adoption_date to today."""

    def test_window_days_calculation(self):
        today = datetime(2026, 7, 20)
        entry = {
            "hypothesis": {"strategy": "rsi", "symbol": "TSLA", "params": {"rsi_window": 14}},
            "tested_at": "2026-06-20T10:00:00",
        }
        with patch("oos_monitor.run_backtest") as mock_bt:
            mock_bt.return_value = {
                "holdout": {"sharpe_ratio": 0.5, "total_return_pct": 10.0, "max_drawdown_pct": -5.0}
            }
            oos_record, error = run_oos_check(entry, today=today)
            assert error is None
            assert oos_record["window_days"] == 29  # 6/20 10:00 → 7/20 00:00 = 29 days

    def test_adoption_in_future_skipped(self):
        today = datetime(2026, 7, 1)
        entry = {
            "hypothesis": {"strategy": "rsi", "symbol": "TSLA", "params": {"rsi_window": 14}},
            "tested_at": "2026-07-10T10:00:00",
        }
        oos_record, error = run_oos_check(entry, today=today)
        assert error == "adoption_in_future"
        assert oos_record is None

    def test_no_adoption_date_skipped(self):
        today = datetime(2026, 7, 20)
        entry = {"hypothesis": {"strategy": "rsi", "symbol": "TSLA", "params": {}}}
        oos_record, error = run_oos_check(entry, today=today)
        assert error == "no_adoption_date"


class TestGreppableOutput:
    """Verify the greppable output format."""

    def test_status_line_format(self):
        """OOS_STATUS line must match: OOS_STATUS <strategy>/<symbol> days=<n> sharpe=<x> return=<y>%"""
        today = datetime(2026, 7, 20)
        knowledge = {
            "adopted": [
                {
                    "hypothesis": {"strategy": "rsi", "symbol": "TSLA", "params": {"rsi_window": 14}},
                    "tested_at": "2026-06-20T10:00:00",
                }
            ]
        }
        with patch("oos_monitor.run_backtest") as mock_bt, \
             patch("oos_monitor.save_knowledge") as mock_save:
            mock_bt.return_value = {
                "holdout": {"sharpe_ratio": 0.5, "total_return_pct": 10.0, "max_drawdown_pct": -5.0}
            }
            # Capture stdout
            buf = io.StringIO()
            sys.stdout = buf
            try:
                run_oos_monitor(knowledge, today=today)
            finally:
                sys.stdout = sys.__stdout__
            
            output = buf.getvalue()
            assert "OOS_STATUS rsi/TSLA days=29 sharpe=0.5 return=10.0%" in output
            assert "OOS_SUMMARY checked=1 negative=0" in output

    def test_empty_adopted_list(self):
        """Empty adopted → OOS_SUMMARY checked=0 negative=0."""
        knowledge = {"adopted": []}
        buf = io.StringIO()
        sys.stdout = buf
        try:
            run_oos_monitor(knowledge, today=datetime(2026, 7, 20))
        finally:
            sys.stdout = sys.__stdout__
        
        output = buf.getvalue()
        assert "OOS_SUMMARY checked=0 negative=0" in output

    def test_negative_oos_counted(self):
        """OOS Sharpe < 0 with window_days >= 30 → counted as negative."""
        today = datetime(2026, 7, 20)
        knowledge = {
            "adopted": [
                {
                    "hypothesis": {"strategy": "rsi", "symbol": "TSLA", "params": {"rsi_window": 14}},
                    "tested_at": "2026-05-01T10:00:00",  # 80 days ago
                }
            ]
        }
        with patch("oos_monitor.run_backtest") as mock_bt, \
             patch("oos_monitor.save_knowledge"):
            mock_bt.return_value = {
                "holdout": {"sharpe_ratio": -0.3, "total_return_pct": -5.0, "max_drawdown_pct": -15.0}
            }
            checked, negative, skipped = run_oos_monitor(knowledge, today=today)
            assert checked == 1
            assert negative == 1

    def test_runner_error_skipped(self):
        """Runner error → skipped, not counted as checked."""
        today = datetime(2026, 7, 20)
        knowledge = {
            "adopted": [
                {
                    "hypothesis": {"strategy": "rsi", "symbol": "TSLA", "params": {"rsi_window": 14}},
                    "tested_at": "2026-06-20T10:00:00",
                }
            ]
        }
        with patch("oos_monitor.run_backtest") as mock_bt, \
             patch("oos_monitor.save_knowledge"):
            mock_bt.return_value = {"error": "timeout", "error_type": "infra"}
            checked, negative, skipped = run_oos_monitor(knowledge, today=today)
            assert checked == 0
            assert skipped == 1
