"""Tests for cross-sectional portfolio metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Dual-import for container flat-layout compatibility
try:
    from backtests.cross_sectional.cross_metrics import (
        portfolio_sharpe,
        portfolio_ir,
        portfolio_turnover,
        portfolio_max_drawdown,
        portfolio_hit_rate,
    )
except ImportError:
    from cross_sectional.cross_metrics import (  # type: ignore[no-redef]
        portfolio_sharpe,
        portfolio_ir,
        portfolio_turnover,
        portfolio_max_drawdown,
        portfolio_hit_rate,
    )


# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioSharpe:
    def test_portfolio_sharpe_matches_manual_reference(self):
        """Synthetic returns → Sharpe = mean/std * sqrt(252)."""
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0.001, 0.02, 500))
        expected = rets.mean() / rets.std() * np.sqrt(252)
        result = portfolio_sharpe(rets)
        assert abs(result - expected) < 1e-10, (
            f"Expected {expected:.10f}, got {result:.10f}"
        )


class TestPortfolioIR:
    def test_portfolio_ir_zero_when_portfolio_equals_benchmark(self):
        """Active returns = 0 → IR returns 0.0."""
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0.001, 0.02, 500))
        ir = portfolio_ir(rets, rets.copy())
        assert ir == 0.0, f"IR should be 0.0 when portfolio == benchmark, got {ir}"


class TestPortfolioMaxDrawdown:
    def test_portfolio_max_drawdown_synthetic_case(self):
        """Known return path with ~30% drawdown → verify MDD < -0.25."""
        # Build a series: steady gain → sharp drop → partial recovery
        np.random.seed(1)
        n = 252
        base = pd.Series(np.random.normal(0.001, 0.01, n))
        # Inject a large drop at day 150-170
        base.iloc[150] = -0.05
        base.iloc[151] = -0.08
        base.iloc[155] = -0.10
        base.iloc[160] = -0.12
        # Some recovery
        base.iloc[170] = 0.02
        base.iloc[180] = 0.03

        mdd = portfolio_max_drawdown(base)
        assert mdd < -0.25, f"Expected MDD < -0.25 (~30%), got {mdd:.4f}"
        assert mdd > -1.0, f"MDD should be > -100%, got {mdd:.4f}"

    def test_portfolio_max_drawdown_no_drawdown(self):
        """Purely increasing returns → MDD = 0.0."""
        rets = pd.Series([0.01] * 100)
        mdd = portfolio_max_drawdown(rets)
        assert mdd <= 0.0


class TestPortfolioHitRate:
    def test_portfolio_hit_rate_half_positive_half_negative(self):
        """50/50 series → 0.5."""
        rets = pd.Series([0.01, -0.01] * 250)
        hr = portfolio_hit_rate(rets)
        assert abs(hr - 0.5) < 1e-10, f"Expected 0.5, got {hr}"


class TestPortfolioTurnover:
    def test_portfolio_turnover_zero_for_static_weights(self):
        """Constant weights → 0 turnover."""
        dates = pd.bdate_range("2024-01-02", periods=100)
        w = pd.DataFrame(
            [[0.5, 0.5]] * 100,
            index=dates,
            columns=["A", "B"],
        )
        to = portfolio_turnover(w)
        assert to == 0.0, f"Expected 0.0 turnover for static weights, got {to}"
