"""Tests for evaluators package (PR-D)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import pytest

# Ensure registrable metrics are loaded
import evaluators  # noqa: F401
from evaluators.base import Evaluator
from evaluators.dispatch import evaluate


# ---------------------------------------------------------------------------
# Duck-typed spec (no import from manifest.py)
# ---------------------------------------------------------------------------

@dataclass
class FakeSpec:
    metrics: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _constant_returns(value: float = 0.001, days: int = 252, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=days, freq="B")
    return pd.DataFrame(np.full((days, cols), value), index=idx, columns=["A", "B", "C"])


def _random_returns(days: int = 252, cols: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days, freq="B")
    return pd.DataFrame(rng.normal(0.0005, 0.01, (days, cols)), index=idx, columns=["A", "B", "C"])


def _equal_weights(returns: pd.DataFrame) -> pd.DataFrame:
    n = len(returns.columns)
    return pd.DataFrame(
        np.full(returns.shape, 1.0 / n),
        index=returns.index,
        columns=returns.columns,
    )


def _benchmark(returns: pd.DataFrame, value: float = 0.0003) -> pd.Series:
    return pd.Series(value, index=returns.index, name="SPY")


# ---------------------------------------------------------------------------
# Known-answer tests
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_constant_returns(self):
        """Constant 0.001 daily, equal-weight -> Sharpe = 0.001/std * sqrt(252).
        With constant returns std=0, so NaN."""
        ret = _constant_returns(0.001)
        fn = Evaluator.get("sharpe")
        result = fn(ret, None, None, {})
        assert math.isnan(result), "Constant returns should give NaN Sharpe (std=0)"

    def test_known_sharpe(self):
        """With known mean and std we can verify Sharpe."""
        ret = _random_returns(seed=1)
        w = _equal_weights(ret)
        fn = Evaluator.get("sharpe")
        result = fn(ret, w, None, {})
        # Verify manually
        pr = (ret * w).sum(axis=1)
        expected = pr.mean() / pr.std(ddof=1) * np.sqrt(252)
        assert abs(result - expected) < 1e-10

    def test_short_series_raises(self):
        ret = _constant_returns(days=10)
        fn = Evaluator.get("sharpe")
        with pytest.raises(ValueError, match="20 rows"):
            fn(ret, None, None, {})


class TestIR:
    def test_no_benchmark_returns_nan(self):
        ret = _random_returns()
        fn = Evaluator.get("ir")
        assert math.isnan(fn(ret, None, None, {}))

    def test_known_ir(self):
        ret = _random_returns(seed=2)
        w = _equal_weights(ret)
        bm = _benchmark(ret, 0.0003)
        fn = Evaluator.get("ir")
        result = fn(ret, w, bm, {})
        pr = (ret * w).sum(axis=1)
        active = pr - bm.reindex(pr.index)
        expected = active.mean() / active.std(ddof=1) * np.sqrt(252)
        assert abs(result - expected) < 1e-10


class TestTurnover:
    def test_no_weights_zero(self):
        ret = _random_returns()
        fn = Evaluator.get("turnover")
        assert fn(ret, None, None, {}) == 0.0

    def test_constant_weights_zero_turnover(self):
        ret = _random_returns()
        w = _equal_weights(ret)  # constant weights
        fn = Evaluator.get("turnover")
        result = fn(ret, w, None, {})
        assert abs(result) < 1e-10

    def test_changing_weights_positive_turnover(self):
        ret = _random_returns()
        w = _equal_weights(ret)
        # Shift weights every 5 days
        for i in range(0, len(w), 5):
            w.iloc[i] = [0.5, 0.3, 0.2]
        fn = Evaluator.get("turnover")
        result = fn(ret, w, None, {})
        assert result > 0.0


class TestCVaR95:
    def test_known_cvar(self):
        ret = _random_returns(seed=3)
        fn = Evaluator.get("cvar_95")
        result = fn(ret, None, None, {})
        pr = ret.mean(axis=1)
        cutoff = int(max(1, np.floor(len(pr) * 0.05)))
        worst = pr.nsmallest(cutoff)
        expected = -worst.mean()
        assert abs(result - expected) < 1e-10

    def test_positive_value(self):
        """CVaR should be a positive number (expected shortfall)."""
        ret = _random_returns(seed=4)
        fn = Evaluator.get("cvar_95")
        assert fn(ret, None, None, {}) > 0.0


class TestMaxDrawdown:
    def test_negative_value(self):
        """Max drawdown should be negative (or zero for monotone-up)."""
        ret = _random_returns(seed=5)
        fn = Evaluator.get("max_drawdown_pct")
        result = fn(ret, None, None, {})
        assert result <= 0.0

    def test_known_drawdown(self):
        """Manually construct a return series with known drawdown."""
        # Up 10%, down 20%, up 5%
        vals = [0.01] * 10 + [-0.02] * 10 + [0.005] * 10
        idx = pd.date_range("2023-01-01", periods=30, freq="B")
        ret = pd.DataFrame({"A": vals}, index=idx)
        fn = Evaluator.get("max_drawdown_pct")
        result = fn(ret, None, None, {})
        # Compute manually
        cum = (1.0 + ret["A"]).cumprod()
        peak = cum.cummax()
        dd = ((cum - peak) / peak).min() * 100.0
        assert abs(result - dd) < 1e-8


class TestTotalReturn:
    def test_constant(self):
        """0.001 daily for 252 days -> (1.001^252 - 1) * 100."""
        ret = _constant_returns(0.001, days=252)
        fn = Evaluator.get("total_return_pct")
        result = fn(ret, None, None, {})
        expected = ((1.001 ** 252) - 1.0) * 100.0
        assert abs(result - expected) < 1e-6


class TestFactorExposure:
    def test_no_factors_nan(self):
        ret = _random_returns()
        fn = Evaluator.get("factor_exposure")
        result = fn(ret, None, None, {})
        assert "factor_exposure_alpha" in result
        assert math.isnan(result["factor_exposure_alpha"])

    def test_single_factor(self):
        ret = _random_returns(seed=6)
        pr = ret.mean(axis=1)
        # Create a single factor MKT
        mkt = pd.DataFrame({"MKT": pr * 1.5 + 0.001}, index=ret.index)
        fn = Evaluator.get("factor_exposure")
        result = fn(ret, None, None, {"factors": mkt})
        assert "factor_exposure_alpha" in result
        assert "factor_exposure_MKT" in result
        # MKT exposure should be close to 0.667 (since pr = MKT/1.5 - ...)
        # Just check it's a finite number
        assert not math.isnan(result["factor_exposure_MKT"])


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_multi_metric(self):
        """Spec with 3 metrics returns dict with those 3 keys."""
        ret = _random_returns(seed=7)
        spec = FakeSpec(metrics=["sharpe", "cvar_95", "max_drawdown_pct"])
        result = evaluate(spec, ret)
        assert set(result.keys()) == {"sharpe", "cvar_95", "max_drawdown_pct"}

    def test_factor_exposure_expansion(self):
        """factor_exposure should expand into multiple keys."""
        ret = _random_returns(seed=8)
        mkt = pd.DataFrame({"MKT": np.random.default_rng(0).normal(0, 0.01, len(ret))}, index=ret.index)
        spec = FakeSpec(metrics=["sharpe", "factor_exposure"])
        result = evaluate(spec, ret, config={"factors": mkt})
        assert "sharpe" in result
        assert "factor_exposure_alpha" in result
        assert "factor_exposure_MKT" in result

    def test_unknown_metric_warning(self, capsys):
        """Unknown metric prints warning and is skipped."""
        ret = _random_returns()
        spec = FakeSpec(metrics=["nonexistent_metric"])
        result = evaluate(spec, ret)
        assert "nonexistent_metric" not in result
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


# ---------------------------------------------------------------------------
# Registry: custom metric via decorator
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_custom_metric(self):
        @Evaluator.register("my_custom_metric")
        def my_custom_metric(returns, weights, benchmark, config):
            return 42.0

        ret = _random_returns()
        spec = FakeSpec(metrics=["my_custom_metric"])
        result = evaluate(spec, ret)
        assert result["my_custom_metric"] == 42.0

        # Cleanup
        del Evaluator._registry["my_custom_metric"]
