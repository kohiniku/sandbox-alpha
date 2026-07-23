"""
Tests for cost_bps override and turnover metric (PR-B1).
"""
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.metrics import (
    COST_BPS,
    apply_trading_cost,
    calculate_metrics,
    _compute_pos_changes,
)


# ---------------------------------------------------------------------------
# Cost override
# ---------------------------------------------------------------------------

class TestCostOverride:
    def test_cost_bps_zero_equals_gross_returns(self):
        """With cost_bps=0, net returns equal gross returns."""
        returns = pd.Series(
            [0.01, 0.02, -0.005, 0.01, -0.01],
            index=pd.date_range("2024-01-02", periods=5, freq="B"),
        )
        signal = pd.Series([0, 0, 1, 1, -1], index=returns.index)
        result = apply_trading_cost(returns, signal, cost_bps=0.0)
        pd.testing.assert_series_equal(result, returns)

    def test_cost_bps_default_unchanged(self):
        """cost_bps=None uses module default COST_BPS."""
        returns = pd.Series(
            [0.01, 0.02, -0.005, 0.01],
            index=pd.date_range("2024-01-02", periods=4, freq="B"),
        )
        signal = pd.Series([0, 0, 1, 1], index=returns.index)

        result_none = apply_trading_cost(returns, signal, cost_bps=None)
        result_default = apply_trading_cost(returns, signal, cost_bps=COST_BPS)
        pd.testing.assert_series_equal(result_none, result_default)

    def test_cost_bps_override_reduces_returns_more(self):
        """Higher cost_bps reduces net returns more than default."""
        n = 20
        returns = pd.Series(
            [0.01] * n,
            index=pd.date_range("2024-01-02", periods=n, freq="B"),
        )
        signal = pd.Series(
            [0] * 5 + [1] * 10 + [0] * 5,
            index=returns.index,
        )

        result_default = apply_trading_cost(returns, signal)
        result_high = apply_trading_cost(returns, signal, cost_bps=10.0)

        # Higher cost → lower total returns
        assert result_high.sum() < result_default.sum()

    def test_metrics_reflect_effective_cost_bps(self):
        """calculate_metrics cost_bps field reflects override, not module default."""
        returns = pd.Series(
            [0.01, -0.005],
            index=pd.date_range("2024-01-02", periods=2, freq="B"),
        )
        m_default = calculate_metrics(returns, cost_bps=None)
        m_override = calculate_metrics(returns, cost_bps=7.5)

        assert m_default["cost_bps"] == COST_BPS
        assert m_override["cost_bps"] == 7.5


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------

class TestTurnover:
    def test_turnover_four_changes_over_252_days(self):
        """4 position changes over 252 days → annualized turnover 4.0."""
        n = 252
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        returns = pd.Series([0.001] * n, index=dates)
        # Signal: 4 unit-magnitude position changes (0→1→0→-1→0)
        signal = pd.Series(
            [0] * 50 + [1] * 50 + [0] * 50 + [-1] * 50 + [0] * 52,
            index=dates,
        )

        result = calculate_metrics(returns, signal)
        assert result["turnover"] == pytest.approx(4.0, abs=0.05)

    def test_turnover_no_trades_is_zero(self):
        """No position changes → turnover 0.0."""
        n = 100
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        returns = pd.Series([0.001] * n, index=dates)
        signal = pd.Series([0] * n, index=dates)

        result = calculate_metrics(returns, signal)
        assert result["turnover"] == 0.0

    def test_turnover_no_signal_is_zero(self):
        """No signal → turnover 0.0."""
        n = 100
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        returns = pd.Series([0.001] * n, index=dates)

        result = calculate_metrics(returns, signal=None)
        assert result["turnover"] == 0.0

    def test_turnover_empty_returns(self):
        """Empty returns → error dict, no turnover key needed (handled by error path)."""
        result = calculate_metrics(pd.Series([], dtype=float))
        assert "error" in result

    def test_turnover_cost_consistency(self):
        """Zero turnover implies zero cost drag (net == gross)."""
        n = 20
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        returns = pd.Series([0.01] * n, index=dates)
        signal = pd.Series([1] * n, index=dates)  # Fixed position, no changes

        result = calculate_metrics(returns, signal)
        assert result["turnover"] == 0.0

        # With zero turnover, cost should be zero regardless of cost_bps
        net_zero = apply_trading_cost(returns, signal, cost_bps=0.0)
        net_high = apply_trading_cost(returns, signal, cost_bps=100.0)
        pd.testing.assert_series_equal(returns, net_zero)
        pd.testing.assert_series_equal(returns, net_high)

    def test_turnover_key_in_metrics(self):
        """Turnover key is present in metrics dict."""
        returns = pd.Series(
            [0.01, -0.005],
            index=pd.date_range("2024-01-02", periods=2, freq="B"),
        )
        result = calculate_metrics(returns)
        assert "turnover" in result
        assert isinstance(result["turnover"], float)


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

ENGINE_SCRIPT = Path(__file__).resolve().parent.parent / "backtests" / "backtest_engine.py"


class TestCLICostBps:
    def test_rejects_negative(self):
        proc = subprocess.run(
            [sys.executable, str(ENGINE_SCRIPT), "--cost-bps", "-1", "--symbol", "AAPL"],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0

    def test_rejects_over_100(self):
        proc = subprocess.run(
            [sys.executable, str(ENGINE_SCRIPT), "--cost-bps", "101", "--symbol", "AAPL"],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0

    def test_rejects_non_numeric(self):
        proc = subprocess.run(
            [sys.executable, str(ENGINE_SCRIPT), "--cost-bps", "abc", "--symbol", "AAPL"],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0

    def test_accepts_zero(self):
        """0.0 is a valid cost_bps."""
        # This will fail at data-load stage, not at arg validation.
        proc = subprocess.run(
            [sys.executable, str(ENGINE_SCRIPT), "--cost-bps", "0", "--strategy",
             "sma_crossover", "--symbol", "AAPL"],
            capture_output=True, text=True,
        )
        # Should NOT fail on argument validation
        stderr = proc.stderr
        assert "cost-bps" not in stderr.lower() or "invalid" not in stderr.lower()

    def test_accepts_100(self):
        """100.0 is a valid cost_bps."""
        proc = subprocess.run(
            [sys.executable, str(ENGINE_SCRIPT), "--cost-bps", "100", "--strategy",
             "sma_crossover", "--symbol", "AAPL"],
            capture_output=True, text=True,
        )
        stderr = proc.stderr
        assert "cost-bps" not in stderr.lower() or "invalid" not in stderr.lower()


# ---------------------------------------------------------------------------
# pos_changes helper (factored for cost/turnover consistency)
# ---------------------------------------------------------------------------

class TestPosChanges:
    def test_shared_helper_no_divergence(self):
        """_compute_pos_changes used by both cost model and turnover."""
        n = 10
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        signal = pd.Series([0, 0, 1, 1, -1, -1, 0, 0, 1, 1], index=dates)
        returns = pd.Series([0.01] * n, index=dates)

        # Cost model path
        cost_frac = COST_BPS / 10000.0
        pos_changes_cost = _compute_pos_changes(signal, returns_index=returns.index)
        cost_drag = (pos_changes_cost * cost_frac).sum()

        # Turnover path
        pos_changes_turnover = _compute_pos_changes(signal, returns_index=returns.index)
        turnover = float(pos_changes_turnover.sum()) / len(returns) * 252

        # Both use the same pos_changes — cost drag is proportional to turnover
        assert pos_changes_cost.sum() == pos_changes_turnover.sum()
        assert cost_drag == pytest.approx(pos_changes_cost.sum() * cost_frac)
        assert turnover > 0  # There are trades
