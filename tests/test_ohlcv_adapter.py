#!/usr/bin/env python3
"""
Tests for data_adapters.ohlcv — Multi-symbol OHLCV data adapter.

Uses synthetic CSVs in tmp_path:
  - 3 symbols x 250 trading days
  - Known gaps for ffill/drop testing
"""

import os
import pytest
import pandas as pd
import numpy as np

from data_adapters.ohlcv import (
    MissingDataError,
    load_ohlcv,
    align_universe,
    to_wide,
    REQUIRED_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_csv(path, dates, symbol, gaps=None):
    """Write a synthetic OHLCV CSV.

    Parameters
    ----------
    dates : list[pd.Timestamp]
    symbol : str
    gaps : set[int] or None
        Row indices to skip (simulate missing trading days).
    """
    rows = []
    for i, d in enumerate(dates):
        if gaps and i in gaps:
            continue
        # Deterministic prices based on index
        base = 100.0 + i * 0.1
        rows.append({
            "Date": d.strftime("%Y-%m-%d"),
            "Open": round(base, 2),
            "High": round(base + 0.5, 2),
            "Low": round(base - 0.5, 2),
            "Close": round(base + 0.1, 2),
            "Volume": int(1000 + i * 10),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


@pytest.fixture
def universe_data(tmp_path):
    """Create 3 symbols x 250 trading days of CSVs in tmp_path."""
    symbols = ["AAPL", "MSFT", "GOOG"]
    # 250 trading days from 2023-01-03
    all_dates = pd.bdate_range("2023-01-03", periods=250, freq="B")

    for sym in symbols:
        path = os.path.join(str(tmp_path), f"{sym}.csv")
        _make_csv(path, all_dates, sym)

    return {
        "data_dir": str(tmp_path),
        "symbols": symbols,
        "all_dates": all_dates,
    }


@pytest.fixture
def gap_data(tmp_path):
    """Create data with known gaps for ffill testing.

    AAPL: complete 10 days
    MSFT: missing day index 5 (1 day gap)
    GOOG: missing day indices 3 and 4 (2 consecutive days gap)
    """
    symbols = ["AAPL", "MSFT", "GOOG"]
    dates = pd.bdate_range("2023-06-01", periods=10, freq="B")

    _make_csv(os.path.join(str(tmp_path), "AAPL.csv"), dates, "AAPL")
    _make_csv(os.path.join(str(tmp_path), "MSFT.csv"), dates, "MSFT", gaps={5})
    _make_csv(os.path.join(str(tmp_path), "GOOG.csv"), dates, "GOOG", gaps={3, 4})

    return {
        "data_dir": str(tmp_path),
        "symbols": symbols,
        "dates": dates,
    }


# ---------------------------------------------------------------------------
# Tests: load_ohlcv
# ---------------------------------------------------------------------------

class TestLoadOhlcv:

    def test_basic_load(self, universe_data):
        """Loads all symbols, correct shape and columns."""
        data = load_ohlcv(
            universe_data["symbols"],
            start="2023-01-03",
            data_dir=universe_data["data_dir"],
        )
        assert set(data.keys()) == {"AAPL", "MSFT", "GOOG"}
        for sym, df in data.items():
            assert list(df.columns) == REQUIRED_COLUMNS
            assert isinstance(df.index, pd.DatetimeIndex)
            assert len(df) == 250

    def test_window_start(self, universe_data):
        """start bound is respected (inclusive)."""
        data = load_ohlcv(
            ["AAPL"],
            start="2023-06-01",
            data_dir=universe_data["data_dir"],
        )
        assert data["AAPL"].index.min() >= pd.Timestamp("2023-06-01")

    def test_window_end(self, universe_data):
        """end bound is respected (inclusive)."""
        data = load_ohlcv(
            ["AAPL"],
            start="2023-01-03",
            end="2023-06-30",
            data_dir=universe_data["data_dir"],
        )
        assert data["AAPL"].index.max() <= pd.Timestamp("2023-06-30")

    def test_window_both_bounds(self, universe_data):
        """Both start and end bounds applied."""
        data = load_ohlcv(
            ["AAPL", "MSFT"],
            start="2023-03-01",
            end="2023-03-31",
            data_dir=universe_data["data_dir"],
        )
        for sym, df in data.items():
            assert df.index.min() >= pd.Timestamp("2023-03-01")
            assert df.index.max() <= pd.Timestamp("2023-03-31")

    def test_end_none_means_all(self, universe_data):
        """end=None returns through last row."""
        data = load_ohlcv(
            ["AAPL"],
            start="2023-01-03",
            end=None,
            data_dir=universe_data["data_dir"],
        )
        assert len(data["AAPL"]) == 250

    def test_missing_symbol_raises(self, universe_data):
        """MissingDataError with clear message when CSV absent."""
        with pytest.raises(MissingDataError) as exc_info:
            load_ohlcv(
                ["TSLA"],  # not in fixture
                start="2023-01-03",
                data_dir=universe_data["data_dir"],
            )
        assert "TSLA" in str(exc_info.value)
        assert "TSLA.csv" in str(exc_info.value)

    def test_empty_universe(self, universe_data):
        """Empty universe returns empty dict."""
        data = load_ohlcv([], start="2023-01-03", data_dir=universe_data["data_dir"])
        assert data == {}

    def test_idempotent(self, universe_data):
        """Same inputs produce same outputs."""
        d1 = load_ohlcv(["AAPL"], start="2023-01-03", data_dir=universe_data["data_dir"])
        d2 = load_ohlcv(["AAPL"], start="2023-01-03", data_dir=universe_data["data_dir"])
        pd.testing.assert_frame_equal(d1["AAPL"], d2["AAPL"])


# ---------------------------------------------------------------------------
# Tests: align_universe
# ---------------------------------------------------------------------------

class TestAlignUniverse:

    def test_basic_alignment(self, universe_data):
        """Aligned panel has MultiIndex columns and correct dates."""
        data = load_ohlcv(
            universe_data["symbols"],
            start="2023-01-03",
            data_dir=universe_data["data_dir"],
        )
        panel = align_universe(data)
        assert isinstance(panel.columns, pd.MultiIndex)
        assert panel.columns.names == ["symbol", "field"]
        # All 3 symbols present
        assert set(panel.columns.get_level_values("symbol")) == {"AAPL", "MSFT", "GOOG"}
        # All fields present
        assert set(panel.columns.get_level_values("field")) == set(REQUIRED_COLUMNS)
        # 250 dates, no NaN
        assert len(panel) == 250
        assert not panel.isna().any().any()

    def test_intersection_of_dates(self, gap_data):
        """Mismatched date ranges yield the intersection after ffill+drop."""
        data = load_ohlcv(
            gap_data["symbols"],
            start="2023-06-01",
            data_dir=gap_data["data_dir"],
        )
        panel = align_universe(data)

        # GOOG has 2 consecutive gaps (indices 3,4). ffill(limit=1) fills
        # index 3 but NOT index 4. So index 4 is dropped from the panel.
        # MSFT has 1 gap (index 5). ffill(limit=1) fills it. No drop.
        # Result: 10 - 1 (GOOG index 4) = 9 dates
        assert len(panel) == 9
        assert not panel.isna().any().any()

    def test_ffill_one_day_filled(self, gap_data):
        """1 missing day is forward-filled (MSFT gap at index 5)."""
        data = load_ohlcv(
            ["AAPL", "MSFT"],  # Only these two; MSFT has 1-day gap
            start="2023-06-01",
            data_dir=gap_data["data_dir"],
        )
        panel = align_universe(data)
        # MSFT's 1-day gap is ffill'd -> all 10 dates survive
        assert len(panel) == 10

    def test_ffill_two_days_drops(self, gap_data):
        """2 consecutive missing days: 2nd day cannot be filled, date dropped."""
        data = load_ohlcv(
            ["AAPL", "GOOG"],  # GOOG has 2-day gap (indices 3,4)
            start="2023-06-01",
            data_dir=gap_data["data_dir"],
        )
        panel = align_universe(data)
        # ffill(limit=1) fills index 3 but not 4 -> date 4 dropped
        assert len(panel) == 9

    def test_empty_data(self):
        """Empty input returns empty DataFrame."""
        panel = align_universe({})
        assert panel.empty

    def test_single_symbol(self, universe_data):
        """Single symbol alignment works."""
        data = load_ohlcv(
            ["AAPL"],
            start="2023-01-03",
            data_dir=universe_data["data_dir"],
        )
        panel = align_universe(data)
        assert len(panel) == 250
        assert set(panel.columns.get_level_values("symbol")) == {"AAPL"}


# ---------------------------------------------------------------------------
# Tests: to_wide
# ---------------------------------------------------------------------------

class TestToWide:

    def test_same_as_align(self, universe_data):
        """to_wide produces identical output to align_universe."""
        data = load_ohlcv(
            universe_data["symbols"],
            start="2023-01-03",
            data_dir=universe_data["data_dir"],
        )
        pd.testing.assert_frame_equal(to_wide(data), align_universe(data))

    def test_cross_sectional_usage(self, universe_data):
        """Demonstrate typical cross-sectional signal usage."""
        data = load_ohlcv(
            universe_data["symbols"],
            start="2023-01-03",
            data_dir=universe_data["data_dir"],
        )
        panel = to_wide(data)
        # Extract Close prices across all symbols
        closes = panel.xs("Close", level="field", axis=1)
        assert closes.shape == (250, 3)
        assert set(closes.columns) == {"AAPL", "MSFT", "GOOG"}
        # Compute returns
        returns = closes.pct_change()
        assert returns.shape == (250, 3)
