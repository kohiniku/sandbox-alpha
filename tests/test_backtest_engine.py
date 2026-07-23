"""
Tests for backtest_engine.py — synthetic data only, no yfinance dependency.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.backtest_engine import (
    COST_BPS,
    apply_trading_cost,
    calculate_metrics,
    run_mean_reversion_strategy,
    run_momentum_strategy,
    run_sma_crossover_strategy,
    split_walkforward,
)


# -- helpers --


def make_ohlc(n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLC data: random walk Close with date index."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    daily_ret = rng.normal(0.0005, 0.015, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_ret)
    df = pd.DataFrame(
        {
            "Open": close * (1 - 0.0005),
            "High": close * (1 + 0.005),
            "Low": close * (1 - 0.005),
            "Close": close,
            "Volume": np.full(n_days, 1_000_000),
        },
        index=dates,
    )
    return df


# -- calculate_metrics --


class TestCalculateMetrics:
    def test_total_return_compound(self):
        """Compound (geometric) total return matches manual calculation."""
        returns = pd.Series(
            [0.01, -0.005, 0.02, 0.015, -0.01],
            index=pd.date_range("2024-01-02", periods=5, freq="B"),
        )
        result = calculate_metrics(returns)
        expected = (1.01 * 0.995 * 1.02 * 1.015 * 0.99) - 1.0
        assert result["total_return_pct"] == pytest.approx(round(expected * 100, 2))

    def test_max_drawdown_compound(self):
        """Max drawdown computed from equity curve (compound), not raw returns."""
        returns = pd.Series(
            [0.02, -0.05, 0.01, -0.03, 0.04, 0.01, -0.02],
            index=pd.date_range("2024-01-02", periods=7, freq="B"),
        )
        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        expected_dd = ((equity - running_max) / running_max).min()
        result = calculate_metrics(returns)
        assert result["max_drawdown_pct"] == pytest.approx(round(expected_dd * 100, 2))

    def test_num_trades_counts_position_changes(self):
        """num_trades counts Signal position changes, not days."""
        returns = pd.Series(
            np.array([0.001] * 100),
            index=pd.date_range("2024-01-02", periods=100, freq="B"),
        )
        signal = pd.Series(
            [0] * 20 + [1] * 30 + [0] * 10 + [-1] * 20 + [0] * 20,
            index=returns.index,
        )
        result = calculate_metrics(returns, signal)
        assert result["num_trades"] == 4

    def test_num_trades_no_signal_defaults_to_rows(self):
        """Without signal, num_trades falls back to len(returns)."""
        returns = pd.Series(
            [0.01, -0.01, 0.02],
            index=pd.date_range("2024-01-02", periods=3, freq="B"),
        )
        result = calculate_metrics(returns)
        assert result["num_trades"] == 3

    def test_num_trades_empty_signal_falls_back(self):
        """With empty signal array, falls back to len(returns)."""
        returns = pd.Series(
            [0.01, -0.01, 0.02],
            index=pd.date_range("2024-01-02", periods=3, freq="B"),
        )
        empty_signal = pd.Series([], dtype=float)
        result = calculate_metrics(returns, empty_signal)
        assert result["num_trades"] == 3

    def test_empty_returns_returns_error(self):
        result = calculate_metrics(pd.Series([], dtype=float))
        assert "error" in result

    def test_sharpe_zero_std(self):
        """Zero-variance returns -> Sharpe = 0."""
        returns = pd.Series(
            [0.01, 0.01, 0.01],
            index=pd.date_range("2024-01-02", periods=3, freq="B"),
        )
        result = calculate_metrics(returns)
        assert result["sharpe_ratio"] == 0.0

    def test_result_keys(self):
        """All expected keys present in metrics dict."""
        returns = pd.Series(
            [0.01, -0.005],
            index=pd.date_range("2024-01-02", periods=2, freq="B"),
        )
        result = calculate_metrics(returns)
        for key in [
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "num_trades",
            "avg_daily_return_pct",
            "cost_bps",
            "turnover",
        ]:
            assert key in result


# -- apply_trading_cost --


class TestApplyTradingCost:
    def test_no_cost_when_static_signal(self):
        """Flat position -> no cost deducted."""
        returns = pd.Series(
            [0.01, 0.02, -0.005, 0.01],
            index=pd.date_range("2024-01-02", periods=4, freq="B"),
        )
        signal = pd.Series([0, 0, 0, 0], index=returns.index)
        result = apply_trading_cost(returns, signal)
        pd.testing.assert_series_equal(result, returns)

    def test_single_entry_cost_one_side(self):
        """One position change (0->1) costs exactly COST_BPS bps."""
        returns = pd.Series(
            [0.01, 0.02, -0.01, 0.005],
            index=pd.date_range("2024-01-02", periods=4, freq="B"),
        )
        signal = pd.Series([0, 0, 1, 1], index=returns.index)

        cost_one_side = COST_BPS / 10000.0
        result = apply_trading_cost(returns, signal)

        expected = returns.copy()
        expected.iloc[3] = returns.iloc[3] - cost_one_side
        pd.testing.assert_series_equal(result, expected)

    def test_round_trip_full(self):
        """Full round trip: entry + exit = 2 * COST_BPS bps total."""
        n = 8
        returns = pd.Series(
            [0.01] * n,
            index=pd.date_range("2024-01-02", periods=n, freq="B"),
        )
        signal = pd.Series([0, 0, 1, 1, 1, 1, 0, 0], index=returns.index)

        cost_one_side = COST_BPS / 10000.0
        result = apply_trading_cost(returns, signal)

        total_cost = (returns - result).sum()
        assert total_cost == pytest.approx(2 * cost_one_side)

    def test_empty_inputs_pass_through(self):
        """Empty returns/signal are returned unchanged."""
        empty = pd.Series([], dtype=float)
        result = apply_trading_cost(empty, empty)
        assert len(result) == 0

    def test_multiple_round_trips(self):
        """Multiple entries and exits each incur proper costs."""
        n = 12
        returns = pd.Series(
            [0.01] * n,
            index=pd.date_range("2024-01-02", periods=n, freq="B"),
        )
        signal = pd.Series(
            [0, 0, 1, 1, 0, 0, -1, -1, -1, 0, 0, 0],
            index=returns.index,
        )
        cost_one_side = COST_BPS / 10000.0
        result = apply_trading_cost(returns, signal)
        total_cost = (returns - result).sum()
        assert total_cost == pytest.approx(4 * cost_one_side)


# -- split_walkforward --


class TestSplitWalkforward:
    def test_60_20_20_split(self):
        """Default 60/20/20 split: exact boundaries, no data loss."""
        df = pd.DataFrame({"Close": range(100)}, index=range(100))
        train, val, holdout = split_walkforward(df)
        assert len(train) == 60
        assert len(val) == 20
        assert len(holdout) == 20
        assert len(train) + len(val) + len(holdout) == 100

    def test_boundary_no_overlap_train_val(self):
        """Train and validation sets are disjoint and contiguous."""
        df = pd.DataFrame({"Close": range(50)}, index=range(50))
        train, val, holdout = split_walkforward(df)
        assert train.index.max() < val.index.min()

    def test_boundary_no_overlap_val_holdout(self):
        """Validation and holdout sets are disjoint and contiguous."""
        df = pd.DataFrame({"Close": range(50)}, index=range(50))
        train, val, holdout = split_walkforward(df)
        assert val.index.max() < holdout.index.min()

    def test_custom_ratio(self):
        """Custom train_ratio honoured."""
        df = pd.DataFrame({"Close": range(100)}, index=range(100))
        train, val, holdout = split_walkforward(df, train_ratio=0.7, val_ratio=0.1, holdout_ratio=0.2)
        assert len(train) == 70
        assert len(val) == 10
        assert len(holdout) == 20

    def test_tiny_dataframe(self):
        """Single-row DataFrame splits cleanly — holdout gets the leftover."""
        df = pd.DataFrame({"Close": [100]}, index=pd.date_range("2024-01-02", periods=1))
        train, val, holdout = split_walkforward(df)
        assert len(train) + len(val) + len(holdout) == 1

    def test_empty_frame(self):
        df = pd.DataFrame({"Close": []})
        train, val, holdout = split_walkforward(df)
        assert len(train) == 0
        assert len(val) == 0
        assert len(holdout) == 0


# -- strategy signal validity --


class TestStrategySignalValidity:
    @pytest.mark.parametrize(
        "strategy_fn,kwargs",
        [
            (run_sma_crossover_strategy, {"fast_window": 5, "slow_window": 20}),
            (run_mean_reversion_strategy, {"window": 10, "threshold": 2.0}),
            (run_momentum_strategy, {"lookback": 10, "hold_period": 3}),
        ],
    )
    def test_signal_in_valid_range(self, strategy_fn, kwargs):
        """Every strategy must only produce Signal in {-1, 0, 1}."""
        df = make_ohlc(252)
        result = strategy_fn(df.copy(), **kwargs)
        signal = result["Signal"].dropna()
        unique_vals = set(signal.unique())
        assert unique_vals.issubset({-1, 0, 1}), f"got {unique_vals}"

    def test_sma_crossover_no_lookahead(self):
        """SMA: Strategy_Returns == Signal.shift(1) * daily Returns."""
        df = make_ohlc(200)
        result = run_sma_crossover_strategy(df.copy(), fast_window=5, slow_window=20)
        daily_rets = result["Close"].pct_change()
        expected = result["Signal"].shift(1) * daily_rets
        valid = result["Strategy_Returns"].notna() & expected.notna()
        pd.testing.assert_series_equal(
            result["Strategy_Returns"][valid],
            expected[valid],
            check_names=False,
        )

    def test_mean_reversion_no_lookahead(self):
        """Mean reversion: Strategy_Returns == Signal.shift(1) * daily Returns."""
        df = make_ohlc(200)
        result = run_mean_reversion_strategy(df.copy(), window=10, threshold=2.0)
        daily_rets = result["Close"].pct_change()
        expected = result["Signal"].shift(1) * daily_rets
        valid = result["Strategy_Returns"].notna() & expected.notna()
        pd.testing.assert_series_equal(
            result["Strategy_Returns"][valid],
            expected[valid],
            check_names=False,
        )

    def test_momentum_no_lookahead(self):
        """Momentum: Strategy_Returns == Position.shift(1) * daily Returns."""
        df = make_ohlc(200)
        result = run_momentum_strategy(df.copy(), lookback=10, hold_period=3)
        daily_rets = result["Close"].pct_change()
        expected = result["Position"].shift(1) * daily_rets
        valid = result["Strategy_Returns"].notna() & expected.notna()
        pd.testing.assert_series_equal(
            result["Strategy_Returns"][valid],
            expected[valid],
            check_names=False,
        )

    def test_all_strategies_produce_strategy_returns(self):
        """Every strategy outputs a Strategy_Returns column (not all NaN)."""
        df = make_ohlc(100)
        for fn, kw in [
            (run_sma_crossover_strategy, {"fast_window": 5, "slow_window": 20}),
            (run_mean_reversion_strategy, {"window": 10, "threshold": 2.0}),
            (run_momentum_strategy, {"lookback": 10, "hold_period": 3}),
        ]:
            result = fn(df.copy(), **kw)
            assert "Strategy_Returns" in result.columns
            assert result["Strategy_Returns"].notna().any()


# -- since_metrics (OOS monitor) --


class TestSinceMetrics:
    """Tests for the metrics_since parameter (post-adoption OOS window)."""

    def test_since_metrics_normal_window(self, tmp_path):
        """since_metrics computes over rows >= metrics_since date."""
        # 500 days of data, compute metrics for last 100 days
        df = make_ohlc(500)
        data_file = tmp_path / "AAPL.csv"
        df.index.name = "Date"  # Ensure index is named for CSV export
        df.to_csv(data_file)
        
        from backtests.backtest_engine import run_backtest
        result = run_backtest(
            strategy_name="sma_crossover",
            symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            walkforward=True,
            data_dir=str(tmp_path),
            metrics_since=df.index[-100].strftime("%Y-%m-%d"),
        )
        
        assert "since_metrics" in result
        sm = result["since_metrics"]
        assert sm["n_days"] == 100
        assert "sharpe_ratio" in sm
        assert "total_return_pct" in sm
        assert "max_drawdown_pct" in sm
        # Sharpe should be a finite number
        assert isinstance(sm["sharpe_ratio"], (int, float))
        assert not np.isnan(sm["sharpe_ratio"])

    def test_since_metrics_empty_window(self, tmp_path):
        """since_metrics with future date returns n_days=0."""
        df = make_ohlc(100)
        data_file = tmp_path / "AAPL.csv"
        df.index.name = "Date"
        df.to_csv(data_file)
        
        from backtests.backtest_engine import run_backtest
        future_date = (df.index[-1] + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        result = run_backtest(
            strategy_name="sma_crossover",
            symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            walkforward=True,
            data_dir=str(tmp_path),
            metrics_since=future_date,
        )
        
        assert "since_metrics" in result
        sm = result["since_metrics"]
        assert sm["n_days"] == 0
        # Should not have sharpe_ratio when n_days=0
        assert "sharpe_ratio" not in sm

    def test_since_metrics_full_range(self, tmp_path):
        """since_metrics with earliest date covers full dataset."""
        df = make_ohlc(252)
        data_file = tmp_path / "AAPL.csv"
        df.index.name = "Date"
        df.to_csv(data_file)
        
        from backtests.backtest_engine import run_backtest
        result = run_backtest(
            strategy_name="sma_crossover",
            symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            walkforward=True,
            data_dir=str(tmp_path),
            metrics_since=df.index[0].strftime("%Y-%m-%d"),
        )
        
        assert "since_metrics" in result
        sm = result["since_metrics"]
        # Should cover all 252 days
        assert sm["n_days"] == 252
        assert "sharpe_ratio" in sm

    def test_since_metrics_warmup_preserved(self, tmp_path):
        """Indicators get full history warmup even when metrics_since truncates."""
        # 300 days, compute since_metrics for last 50 days
        # SMA(30) needs 30 days warmup, so last 50 days should work fine
        df = make_ohlc(300)
        data_file = tmp_path / "AAPL.csv"
        df.index.name = "Date"
        df.to_csv(data_file)
        
        from backtests.backtest_engine import run_backtest
        result = run_backtest(
            strategy_name="sma_crossover",
            symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            walkforward=True,
            data_dir=str(tmp_path),
            metrics_since=df.index[-50].strftime("%Y-%m-%d"),
        )
        
        assert "since_metrics" in result
        sm = result["since_metrics"]
        assert sm["n_days"] == 50
        # Should have valid metrics (indicators had warmup from full history)
        assert "sharpe_ratio" in sm
        assert "total_return_pct" in sm
        assert not np.isnan(sm["sharpe_ratio"])
