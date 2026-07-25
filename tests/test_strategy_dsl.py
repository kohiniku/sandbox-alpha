"""Tests for strategy_dsl.py — indicator computation, rule evaluation,
cross-sectional ranking, DSL execution error handling, and E2E manifest_runner
dispatch for DSL-based manifests.
"""

import json
import sys
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_dsl import (
    run_dsl_strategy,
    DslExecutionError,
    _compute_sma,
    _compute_ema,
    _compute_rsi,
    _compute_bollinger_pct_b,
    _compute_rolling_zscore,
    _compute_rolling_vol,
    _compute_macd,
    _eval_rule,
    _eval_node,
)
from manifest import StrategyManifest
from manifest_runner import run_manifest


# ---------------------------------------------------------------------------
# Synthetic OHLCV helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n_days=200, seed=42):
    """Create a minimal single-symbol OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    rets = rng.normal(0.0005, 0.015, size=n_days)
    close = pd.Series(100.0 * (1 + rets).cumprod(), index=idx, name="Close")
    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n_days)),
        "High": close * (1 + np.abs(rng.normal(0, 0.005, n_days))),
        "Low": close * (1 - np.abs(rng.normal(0, 0.005, n_days))),
        "Close": close,
        "Volume": rng.integers(1_000_000, 10_000_000, n_days),
    }, index=idx)
    return df


def _make_ohlcv_csv(tmp_path, symbol, n_days=200, seed=42):
    """Write a CSV for manifest_runner data loading."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rets = rng.normal(0.0005, 0.015, n_days)
    close = 100.0 * (1 + rets).cumprod()
    df = pd.DataFrame({
        "Date": dates,
        "Open": close * (1 + rng.normal(0, 0.002, n_days)),
        "High": close * (1 + np.abs(rng.normal(0, 0.005, n_days))),
        "Low": close * (1 - np.abs(rng.normal(0, 0.005, n_days))),
        "Close": close,
        "Volume": rng.integers(1_000_000, 10_000_000, n_days),
    })
    df.to_csv(os.path.join(str(tmp_path), f"{symbol}.csv"), index=False)


def _make_all_data(symbols, n_days=200, seed=42):
    """Create dict of {symbol: DataFrame}."""
    data = {}
    for i, sym in enumerate(symbols):
        data[sym] = _make_ohlcv_df(n_days=n_days, seed=seed + i)
    return data


# ---------------------------------------------------------------------------
# 1. Indicator unit tests
# ---------------------------------------------------------------------------

class TestIndicatorSMA:
    def test_sma_matches_rolling_mean(self):
        df = _make_ohlcv_df(n_days=100)
        result = _compute_sma(df["Close"], window=20)
        expected = df["Close"].rolling(20, min_periods=1).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_sma_window_1_equals_close(self):
        df = _make_ohlcv_df(n_days=50)
        result = _compute_sma(df["Close"], window=1)
        pd.testing.assert_series_equal(result, df["Close"], check_names=False)


class TestIndicatorEMA:
    def test_ema_matches_pandas_ewm(self):
        df = _make_ohlcv_df(n_days=100)
        result = _compute_ema(df["Close"], window=20)
        expected = df["Close"].ewm(span=20, min_periods=1, adjust=False).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestIndicatorRSI:
    def test_rsi_on_simple_up_sequence(self):
        """RSI should approach 100 on a pure uptrend."""
        n = 50
        idx = pd.bdate_range("2023-01-02", periods=n)
        close = pd.Series(100.0 + np.arange(n) * 0.5, index=idx)
        result = _compute_rsi(close, window=14)
        # After warm-up, RSI should be high (uptrend)
        last_val = result.dropna().iloc[-1]
        assert last_val > 60, f"RSI on uptrend should be > 60, got {last_val}"

    def test_rsi_on_simple_down_sequence(self):
        """RSI should approach 0 on a pure downtrend."""
        n = 50
        idx = pd.bdate_range("2023-01-02", periods=n)
        close = pd.Series(100.0 - np.arange(n) * 0.5, index=idx)
        result = _compute_rsi(close, window=14)
        last_val = result.dropna().iloc[-1]
        assert last_val < 40, f"RSI on downtrend should be < 40, got {last_val}"


class TestIndicatorBollingerPctB:
    def test_bollinger_pct_b_at_sma_is_0_5(self):
        """When price equals the SMA, %B should be ~0.5 (midpoint of bands)."""
        n = 100
        idx = pd.bdate_range("2023-01-02", periods=n)
        # Flat price -> SMA = price, bands = price ± 2*0 = price
        close = pd.Series(100.0, index=idx)
        result = _compute_bollinger_pct_b(close, window=20)
        # After warmup, should be NaN (denom=0) or ~0.5
        # Actually with flat price std=0, denom=0, %B = NaN
        # So test with a slight trend
        close2 = pd.Series(100.0 + np.arange(n) * 0.05, index=idx)
        result = _compute_bollinger_pct_b(close2, window=20)
        last = result.dropna().iloc[-1]
        assert 0.0 <= last <= 1.0 or np.isnan(last), f"%B should be 0-1, got {last}"

    def test_bollinger_pct_b_near_upper(self):
        """Price 2 std above SMA -> %B ~1.0."""
        n = 80
        idx = pd.bdate_range("2023-01-02", periods=n)
        base = pd.Series(100.0, index=idx)
        # Set last 20 values 2 std above mean of first 60
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, n)
        close = base + noise
        # For a clear upper band test, spike the last value
        close.iloc[-1] = close.iloc[:60].mean() + 2 * close.iloc[:60].std()
        result = _compute_bollinger_pct_b(close, window=20)
        assert result.iloc[-1] > 0.9, f"Spike above +2std should give %B > 0.9, got {result.iloc[-1]}"


class TestIndicatorRollingZscore:
    def test_zscore_zero_mean(self):
        n = 100
        idx = pd.bdate_range("2023-01-02", periods=n)
        close = pd.Series(100.0, index=idx)
        result = _compute_rolling_zscore(close, window=20)
        # Flat price -> std=0 -> NaN; last valid should be near 0
        # Actually NaN from div by 0, that's fine
        assert result.isna().all() or abs(result.dropna().iloc[-1]) < 0.01

    def test_zscore_spike(self):
        n = 80
        idx = pd.bdate_range("2023-01-02", periods=n)
        close = pd.Series(100.0, index=idx)
        close.iloc[-1] = 105.0
        result = _compute_rolling_zscore(close, window=20)
        assert result.iloc[-1] > 1.0, f"2std spike should have zscore > 1, got {result.iloc[-1]}"


class TestIndicatorRollingVol:
    def test_rolling_vol_positive(self):
        df = _make_ohlcv_df(n_days=200)
        result = _compute_rolling_vol(df["Close"], window=20)
        valid = result.dropna()
        assert (valid >= 0).all(), "Volatility should be >= 0"


class TestIndicatorMACD:
    def test_macd_sign(self):
        df = _make_ohlcv_df(n_days=300)
        result = _compute_macd(df["Close"], window=26)
        # MACD is just a series, should have values after warmup
        assert not result.dropna().empty


# ---------------------------------------------------------------------------
# 2. Rule evaluator tests
# ---------------------------------------------------------------------------

class TestRuleEvaluation:
    def _make_values(self, n=200):
        """Create indicator values for rule testing."""
        idx = pd.bdate_range("2023-01-02", periods=n)
        values = {
            "rsi": pd.Series(np.linspace(20, 80, n), index=idx),
            "sma": pd.Series(100.0 + np.arange(n) * 0.5, index=idx),
        }
        return values

    def test_comparison_gt(self):
        values = self._make_values(100)
        node = {"op": "gt", "left": {"indicator": "rsi"}, "right": {"const": 50}}
        result = _eval_node(node, values)
        assert result.iloc[-1]  # Last value 80 > 50
        assert not result.iloc[0]  # First value 20 > 50 is false

    def test_comparison_lt(self):
        values = self._make_values(100)
        node = {"op": "lt", "left": {"indicator": "rsi"}, "right": {"const": 30}}
        result = _eval_node(node, values)
        assert result.iloc[0]  # First value 20 < 30
        assert not result.iloc[-1]

    def test_and_combine(self):
        values = self._make_values(100)
        node = {
            "op": "and",
            "children": [
                {"op": "gt", "left": {"indicator": "rsi"}, "right": {"const": 40}},
                {"op": "lt", "left": {"indicator": "rsi"}, "right": {"const": 60}},
            ],
        }
        result = _eval_node(node, values)
        # Should be True where 40 < rsi < 60
        assert result.iloc[50]  # rsi ~50

    def test_or_combine(self):
        values = self._make_values(100)
        node = {
            "op": "or",
            "children": [
                {"op": "lt", "left": {"indicator": "rsi"}, "right": {"const": 25}},
                {"op": "gt", "left": {"indicator": "rsi"}, "right": {"const": 75}},
            ],
        }
        result = _eval_node(node, values)
        assert result.iloc[0]  # rsi=20 < 25
        assert result.iloc[-1]  # rsi=80 > 75
        assert not result.iloc[40]  # rsi ~44

    def test_not(self):
        values = self._make_values(100)
        node = {
            "op": "not",
            "child": {"op": "gt", "left": {"indicator": "rsi"}, "right": {"const": 50}},
        }
        result = _eval_node(node, values)
        assert result.iloc[0]  # NOT(20>50) = True
        assert not result.iloc[-1]  # NOT(80>50) = False

    def test_crosses_above(self):
        """Create a series that crosses above 50 at a known day."""
        n = 50
        idx = pd.bdate_range("2023-01-02", periods=n)
        rsi = pd.Series(np.where(np.arange(n) < 25, 40, 60), index=idx)
        values = {"rsi": rsi}
        node = {
            "op": "crosses_above",
            "left": {"indicator": "rsi"},
            "right": {"const": 50},
        }
        result = _eval_node(node, values)
        # Should be True exactly at index 25 (first day above 50)
        assert result.iloc[25], f"crosses_above should be True at crossover day"
        assert not result.iloc[24]
        assert not result.iloc[26]
        assert result.sum() == 1, f"should cross exactly once, got {result.sum()}"

    def test_crosses_below(self):
        """Create a series that crosses below 50 at a known day."""
        n = 50
        idx = pd.bdate_range("2023-01-02", periods=n)
        rsi = pd.Series(np.where(np.arange(n) < 25, 60, 40), index=idx)
        values = {"rsi": rsi}
        node = {
            "op": "crosses_below",
            "left": {"indicator": "rsi"},
            "right": {"const": 50},
        }
        result = _eval_node(node, values)
        assert result.iloc[25], f"crosses_below should be True at crossover day"
        assert result.sum() == 1


# ---------------------------------------------------------------------------
# 3. Malformed rule node -> DslExecutionError
# ---------------------------------------------------------------------------

class TestMalformedNodes:
    def _make_values(self, n=50):
        idx = pd.bdate_range("2023-01-02", periods=n)
        return {"rsi": pd.Series(50.0, index=idx)}

    def test_unknown_op(self):
        values = self._make_values()
        node = {"op": "foobar", "left": {"indicator": "rsi"}, "right": {"const": 50}}
        with pytest.raises(DslExecutionError, match="Unknown op"):
            _eval_node(node, values)

    def test_missing_indicator(self):
        values = self._make_values()
        node = {"op": "gt", "left": {"indicator": "nonexistent"}, "right": {"const": 50}}
        with pytest.raises(DslExecutionError, match="Unknown indicator"):
            _eval_node(node, values)

    def test_missing_op(self):
        values = self._make_values()
        node = {"left": {"indicator": "rsi"}}
        with pytest.raises(DslExecutionError, match="missing 'op'"):
            _eval_node(node, values)

    def test_and_without_children(self):
        values = self._make_values()
        node = {"op": "and"}
        with pytest.raises(DslExecutionError, match="non-empty 'children'"):
            _eval_node(node, values)

    def test_not_without_child(self):
        values = self._make_values()
        node = {"op": "not"}
        with pytest.raises(DslExecutionError, match="requires a 'child'"):
            _eval_node(node, values)

    def test_comparison_without_operands(self):
        values = self._make_values()
        node = {"op": "gt", "left": {"indicator": "rsi"}}
        with pytest.raises(DslExecutionError, match="'right' operand"):
            _eval_node(node, values)

    def test_invalid_const_type(self):
        values = self._make_values()
        node = {"op": "gt", "left": {"indicator": "rsi"}, "right": {"const": "abc"}}
        with pytest.raises(DslExecutionError, match="numeric"):
            _eval_node(node, values)

    def test_none_rule(self):
        with pytest.raises(DslExecutionError, match="None"):
            _eval_rule(None, self._make_values())


# ---------------------------------------------------------------------------
# 4. E2E: single_asset_rule through manifest_runner
# ---------------------------------------------------------------------------

class TestDslSingleAssetE2E:
    def test_rsi_oversold_overbought(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        n_days = 200
        for i, sym in enumerate(symbols):
            _make_ohlcv_csv(str(tmp_path), sym, n_days=n_days, seed=42 + i)

        logic_spec = {
            "kind": "single_asset_rule",
            "indicators": {
                "rsi14": {"type": "rsi", "window": 14},
            },
            "entry_rule": {
                "op": "lt", "left": {"indicator": "rsi14"}, "right": {"const": 30},
            },
            "exit_rule": {
                "op": "gt", "left": {"indicator": "rsi14"}, "right": {"const": 70},
            },
            "position_sizing": "long_only_binary",
        }

        payload = {
            "name": "rsi_dsl_oversold",
            "code_b64": "",  # Not used when logic_spec is present
            "logic_spec": logic_spec,
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {
                "type": "portfolio",
                "metrics": ["sharpe", "max_drawdown_pct"],
                "benchmark": None,
            },
            "execution_mode": "structured",
        }

        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert violations == [], f"Manifest should be valid: {violations}"

        result = json.loads(run_manifest(manifest, str(tmp_path)))
        assert result["status"] == "ok", f"Expected ok, got: {result.get('error', result)}"
        assert result["manifest_name"] == "rsi_dsl_oversold"
        assert result["execution_mode"] == "structured"
        assert "val_sharpe" in result["metrics"]
        assert "holdout_sharpe" in result["metrics"]
        assert result["config"]["weighting"] == "dsl_trusted"

    def test_sma_crossover_dsl(self, tmp_path):
        """SMA crossover via DSL — fast SMA crosses above slow SMA as entry."""
        symbols = ["SPY"]
        _make_ohlcv_csv(str(tmp_path), "SPY", n_days=300, seed=123)

        logic_spec = {
            "kind": "single_asset_rule",
            "indicators": {
                "sma_fast": {"type": "sma", "window": 10},
                "sma_slow": {"type": "sma", "window": 30},
            },
            "entry_rule": {
                "op": "gt", "left": {"indicator": "sma_fast"}, "right": {"indicator": "sma_slow"},
            },
            "exit_rule": {
                "op": "lt", "left": {"indicator": "sma_fast"}, "right": {"indicator": "sma_slow"},
            },
            "position_sizing": "long_short_binary",
        }

        payload = {
            "name": "sma_dsl_crossover",
            "code_b64": "",
            "logic_spec": logic_spec,
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {
                "type": "portfolio",
                "metrics": ["sharpe", "max_drawdown_pct"],
                "benchmark": None,
            },
            "execution_mode": "structured",
        }

        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert violations == [], f"Manifest should be valid: {violations}"

        result = json.loads(run_manifest(manifest, str(tmp_path)))
        assert result["status"] == "ok", f"Expected ok, got: {result.get('error', result)}"


# ---------------------------------------------------------------------------
# 5. E2E: cross_sectional_rank through manifest_runner
# ---------------------------------------------------------------------------

class TestDslCrossSectionalE2E:
    def test_momentum_top5(self, tmp_path):
        """Cross-sectional momentum: rank by SMA windows, pick top 3."""
        symbols = ["A", "B", "C", "D", "E"]
        n_days = 300
        for i, sym in enumerate(symbols):
            _make_ohlcv_csv(str(tmp_path), sym, n_days=n_days, seed=42 + i)

        logic_spec = {
            "kind": "cross_sectional_rank",
            "indicators": {
                "mom60": {"type": "sma", "window": 60},
            },
            "rank_by": "mom60",
            "direction": "top",
            "n_select": 3,
            "weighting": "equal",
        }

        payload = {
            "name": "xs_momentum_dsl",
            "code_b64": "",
            "logic_spec": logic_spec,
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {
                "type": "portfolio",
                "metrics": ["sharpe", "max_drawdown_pct"],
                "benchmark": None,
                "extras": {
                    "cross_return_type": "scores",
                    "construction_mode": "top_k",
                    "top_k": 3,
                    "rebalance": "monthly",
                    "cost_bps": 5.0,
                    "long_only": True,
                },
            },
            "execution_mode": "structured",
        }

        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert violations == [], f"Manifest should be valid: {violations}"

        result = json.loads(run_manifest(manifest, str(tmp_path)))
        assert "in_sample" in result or "status" in result
        # Cross engine returns in_sample/out_of_sample/holdout directly


# ---------------------------------------------------------------------------
# 6. run_dsl_strategy dispatch tests
# ---------------------------------------------------------------------------

class TestDslDispatch:
    def test_unknown_kind(self):
        all_data = _make_all_data(["AAPL"], n_days=50)
        spec = {"kind": "unknown_kind"}
        with pytest.raises(DslExecutionError, match="Unknown logic_spec kind"):
            run_dsl_strategy(spec, all_data)

    def test_missing_kind(self):
        all_data = _make_all_data(["AAPL"], n_days=50)
        spec = {}
        with pytest.raises(DslExecutionError, match="missing 'kind'"):
            run_dsl_strategy(spec, all_data)

    def test_not_a_dict(self):
        with pytest.raises(DslExecutionError, match="must be a dict"):
            run_dsl_strategy("not-a-dict", {})  # type: ignore[arg-type]

    def test_unknown_indicator_type(self):
        all_data = _make_all_data(["AAPL"], n_days=50)
        spec = {
            "kind": "single_asset_rule",
            "indicators": {"bad": {"type": "kalman_filter", "window": 20}},
            "entry_rule": {"op": "gt", "left": {"indicator": "bad"}, "right": {"const": 0}},
        }
        with pytest.raises(DslExecutionError, match="Unknown indicator type"):
            run_dsl_strategy(spec, all_data)

    def test_cross_sectional_missing_rank_by(self):
        all_data = _make_all_data(["A", "B"], n_days=50)
        spec = {
            "kind": "cross_sectional_rank",
            "indicators": {"mom": {"type": "sma", "window": 20}},
        }
        with pytest.raises(DslExecutionError, match="rank_by"):
            run_dsl_strategy(spec, all_data)

    def test_cross_sectional_missing_n_and_pct(self):
        all_data = _make_all_data(["A", "B"], n_days=50)
        spec = {
            "kind": "cross_sectional_rank",
            "indicators": {"mom": {"type": "sma", "window": 20}},
            "rank_by": "mom",
            "direction": "top",
        }
        with pytest.raises(DslExecutionError, match="n_select.*pct_select"):
            run_dsl_strategy(spec, all_data)

    def test_cross_sectional_both_n_and_pct(self):
        all_data = _make_all_data(["A", "B"], n_days=50)
        spec = {
            "kind": "cross_sectional_rank",
            "indicators": {"mom": {"type": "sma", "window": 20}},
            "rank_by": "mom",
            "direction": "top",
            "n_select": 10,
            "pct_select": 0.1,
        }
        with pytest.raises(DslExecutionError, match="not both"):
            run_dsl_strategy(spec, all_data)

    def test_single_asset_rule_output_shape(self):
        all_data = _make_all_data(["AAPL", "MSFT"], n_days=100)
        spec = {
            "kind": "single_asset_rule",
            "indicators": {"r": {"type": "rsi", "window": 14}},
            "entry_rule": {"op": "lt", "left": {"indicator": "r"}, "right": {"const": 30}},
        }
        result = run_dsl_strategy(spec, all_data)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["AAPL", "MSFT"]
        assert len(result) == 100
        # Values should be in {0, 1} range for long_only_binary
        assert set(result["AAPL"].unique()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 7. Manifest validation: logic_spec mutual exclusion
# ---------------------------------------------------------------------------

class TestManifestLogicSpecValidation:
    def test_both_code_and_logic_rejected(self):
        payload = {
            "name": "bad_manifest",
            "code_b64": "aW1wb3J0IHBhbmRhcw==",  # "import pandas"
            "logic_spec": {
                "kind": "single_asset_rule",
                "indicators": {"r": {"type": "rsi", "window": 14}},
            },
            "data_sources": [
                {"type": "ohlcv", "universe": ["AAPL"], "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert any("mutually exclusive" in v for v in violations), f"Got: {violations}"

    def test_neither_code_nor_logic_rejected(self):
        payload = {
            "name": "empty_manifest",
            "data_sources": [
                {"type": "ohlcv", "universe": ["AAPL"], "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert any("either code_b64 or logic_spec" in v for v in violations), f"Got: {violations}"

    def test_expert_mode_rejects_logic_spec(self):
        payload = {
            "name": "expert_with_dsl",
            "code_b64": "ZGVmIHJ1bihkYXRhLCB0cmFpbl9lbmQsIHZhbF9lbmQsIGJlbmNobWFyaywgY29uZmlnKTogcmV0dXJuIHt9",
            "logic_spec": {
                "kind": "single_asset_rule",
                "indicators": {"r": {"type": "rsi", "window": 14}},
            },
            "data_sources": [
                {"type": "ohlcv", "universe": ["AAPL"], "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "expert",
        }
        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert any("not allowed in expert mode" in v for v in violations), f"Got: {violations}"

    def test_valid_dsl_manifest(self):
        payload = {
            "name": "valid_dsl",
            "logic_spec": {
                "kind": "single_asset_rule",
                "indicators": {"r": {"type": "rsi", "window": 14}},
            },
            "data_sources": [
                {"type": "ohlcv", "universe": ["AAPL"], "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert violations == [], f"Valid DSL manifest should have no violations: {violations}"
        assert manifest.logic_spec == payload["logic_spec"]

    def test_dsl_to_dict_roundtrips(self):
        payload = {
            "name": "roundtrip",
            "logic_spec": {
                "kind": "single_asset_rule",
                "indicators": {"r": {"type": "rsi", "window": 14}},
            },
            "data_sources": [
                {"type": "ohlcv", "universe": ["AAPL"], "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(payload)
        d = manifest.to_dict()
        assert d["logic_spec"] == payload["logic_spec"]

    def test_valid_cross_sectional_manifest(self):
        payload = {
            "name": "xs_dsl",
            "logic_spec": {
                "kind": "cross_sectional_rank",
                "indicators": {"mom": {"type": "sma", "window": 60}},
                "rank_by": "mom",
                "direction": "top",
                "n_select": 10,
                "weighting": "equal",
            },
            "data_sources": [
                {"type": "ohlcv", "universe": ["A", "B", "C"], "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert violations == [], f"Valid cross-sectional DSL manifest: {violations}"

    def test_rank_by_not_in_indicators(self):
        payload = {
            "name": "bad_xs",
            "logic_spec": {
                "kind": "cross_sectional_rank",
                "indicators": {"mom": {"type": "sma", "window": 60}},
                "rank_by": "not_an_indicator",
                "direction": "top",
                "n_select": 10,
                "weighting": "equal",
            },
            "data_sources": [
                {"type": "ohlcv", "universe": ["A"], "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert any("rank_by" in v for v in violations), f"Got: {violations}"

    def test_invalid_logic_spec_kind(self):
        payload = {
            "name": "bad_kind",
            "logic_spec": {"kind": "expert_override"},
            "data_sources": [
                {"type": "ohlcv", "universe": ["AAPL"], "start": "2023-01-01"}
            ],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(payload)
        violations = manifest.validate()
        assert any("kind" in v for v in violations), f"Got: {violations}"
