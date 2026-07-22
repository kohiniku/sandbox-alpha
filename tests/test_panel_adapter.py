"""
Unit test for _panel_adapter.wrap_single_as_cross.
"""
import numpy as np
import pandas as pd
import pytest

from backtests.strategies._panel_adapter import wrap_single_as_cross
from backtests.strategies._single_name import sma_crossover


def _make_synthetic_panel(n_symbols=3, n_days=30, seed=42):
    """Build a dict-of-DataFrames panel with synthetic Close prices."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    symbols = [f"SYM_{chr(65 + i)}" for i in range(n_symbols)]  # SYM_A, SYM_B, SYM_C
    panel = {}
    for sym in symbols:
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=n_days))
        df = pd.DataFrame({"Close": close, "Open": close * 0.999,
                            "High": close * 1.005, "Low": close * 0.995,
                            "Volume": np.full(n_days, 1_000_000)}, index=dates)
        panel[sym] = df
    return panel, symbols


class TestPanelAdapter:
    def test_output_shape_and_values(self):
        panel, universe = _make_synthetic_panel()
        adapted = wrap_single_as_cross(sma_crossover.compute_signal, name="sma_crossover")
        result = adapted(panel, universe, extras={"fast_window": 5, "slow_window": 10})

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == universe
        assert len(result) == 30

        # Values must be in {-1, 0, 1} for non-NaN rows
        valid = result.dropna(how="all")
        for col in result.columns:
            col_vals = result[col].dropna()
            assert col_vals.isin([-1, 0, 1]).all(), f"Column {col} has values outside {{-1,0,1}}"

    def test_matches_direct_call_per_symbol(self):
        panel, universe = _make_synthetic_panel()
        adapted = wrap_single_as_cross(sma_crossover.compute_signal, name="sma_crossover")
        extras = {"fast_window": 5, "slow_window": 10}
        result = adapted(panel, universe, extras=extras)

        for sym in universe:
            df_out, pos_col = sma_crossover.compute_signal(panel[sym].copy(), **extras)
            direct = df_out[pos_col]
            adapter = result[sym]
            # Only compare non-NaN rows
            mask = adapter.notna() & direct.notna()
            pd.testing.assert_series_equal(adapter[mask], direct[mask], check_names=False)

    def test_missing_symbol_absent_column(self):
        panel, universe = _make_synthetic_panel()
        adapted = wrap_single_as_cross(sma_crossover.compute_signal, name="sma_crossover")
        # Include a symbol not in panel
        extended_universe = universe + ["SYM_MISSING"]
        result = adapted(panel, extended_universe)

        assert list(result.columns) == extended_universe
        assert result["SYM_MISSING"].isna().all()
