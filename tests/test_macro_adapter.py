#!/usr/bin/env python3
"""
Tests for data_adapters.macro — FRED macro data adapter.

Phase 2 PR-K: covers fixture parsing, resampling correctness, date window,
and empty corpus handling.
"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from data_adapters.macro import load_macro, _FREQ_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_macro_csv(corpus_dir: str, sid: str, dates, values):
    """Write a FRED-style DATE,VALUE CSV for a series."""
    df = pd.DataFrame({"DATE": dates, "VALUE": values})
    df.to_csv(os.path.join(corpus_dir, f"{sid}.csv"), index=False)


@pytest.fixture
def macro_corpus(tmp_path):
    """Create a macro corpus with daily DGS10, monthly UNRATE, quarterly GDP."""
    corpus_dir = os.path.join(str(tmp_path), "macro_corpus")
    os.makedirs(corpus_dir, exist_ok=True)

    # DGS10: daily yields (250 trading days in 2024)
    dgs10_dates = pd.bdate_range("2024-01-02", periods=250, freq="B")
    np.random.seed(42)
    dgs10_vals = 4.0 + np.cumsum(np.random.randn(250) * 0.02)
    _make_macro_csv(corpus_dir, "DGS10", dgs10_dates, dgs10_vals)

    # DGS2: daily yields
    dgs2_vals = 4.5 + np.cumsum(np.random.randn(250) * 0.02)
    _make_macro_csv(corpus_dir, "DGS2", dgs10_dates, dgs2_vals)

    # UNRATE: monthly (Jan-Dec 2024)
    unrate_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    unrate_vals = [3.7, 3.9, 3.8, 3.9, 4.0, 4.1, 4.3, 4.2, 4.1, 4.1, 4.2, 4.0]
    _make_macro_csv(corpus_dir, "UNRATE", unrate_dates, unrate_vals)

    return {
        "data_dir": str(tmp_path),
        "corpus_dir": corpus_dir,
    }


# ---------------------------------------------------------------------------
# Tests: fixture parsing
# ---------------------------------------------------------------------------

class TestFixtureParsing:
    def test_load_single_series_daily(self, macro_corpus):
        """Load a single daily series, verify shape and index type."""
        df = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            frequency="daily",
            data_dir=macro_corpus["data_dir"],
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["DGS10"]
        assert len(df) > 0

    def test_load_multiple_series(self, macro_corpus):
        """Load two series, verify both columns present."""
        df = load_macro(
            series=["DGS10", "DGS2"],
            start="2024-01-01",
            frequency="daily",
            data_dir=macro_corpus["data_dir"],
        )
        assert list(df.columns) == ["DGS10", "DGS2"]
        assert len(df) > 0

    def test_non_existent_series_warns(self, macro_corpus):
        """Non-existent series ID triggers a warning and is skipped."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = load_macro(
                series=["DGS10", "NONEXISTENT"],
                start="2024-01-01",
                frequency="daily",
                data_dir=macro_corpus["data_dir"],
            )
            # Should have warning about the missing CSV
            has_warning = any("NONEXISTENT" in str(warning.message) for warning in w)
            assert has_warning
            # Should still have DGS10
            assert "DGS10" in df.columns
            assert "NONEXISTENT" not in df.columns


# ---------------------------------------------------------------------------
# Tests: resampling correctness
# ---------------------------------------------------------------------------

class TestResampling:
    def test_daily_to_weekly_resample(self, macro_corpus):
        """Daily DGS10 resampled to weekly yields fewer observations."""
        df_daily = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            frequency="daily",
            data_dir=macro_corpus["data_dir"],
        )
        df_weekly = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            frequency="weekly",
            data_dir=macro_corpus["data_dir"],
        )
        assert len(df_weekly) < len(df_daily)
        # All weekly dates should be in the daily index range
        assert df_weekly.index.min() >= df_daily.index.min()

    def test_daily_to_monthly_resample(self, macro_corpus):
        """Daily DGS10 resampled to monthly yields ~12 rows."""
        df_monthly = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            frequency="monthly",
            data_dir=macro_corpus["data_dir"],
        )
        # Should have roughly 12 months for a year of data
        assert 10 <= len(df_monthly) <= 14

    def test_monthly_resample_preserves_native(self, macro_corpus):
        """Monthly UNRATE resampled to monthly stays at ~12 rows."""
        df = load_macro(
            series=["UNRATE"],
            start="2024-01-01",
            frequency="monthly",
            data_dir=macro_corpus["data_dir"],
        )
        assert len(df) == 12

    def test_resample_last_value(self, macro_corpus):
        """Resampling uses last() — last observation in each period."""
        # Create a small dataset with known values (Mon-Fri same week)
        corpus_dir = macro_corpus["corpus_dir"]
        # Use dates all in the same Monday-Sunday week
        dates = pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05",
                                "2024-06-06", "2024-06-07"])
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        _make_macro_csv(corpus_dir, "TEST", dates, values)

        df = load_macro(
            series=["TEST"],
            start="2024-06-03",
            frequency="weekly",
            data_dir=macro_corpus["data_dir"],
        )
        # Weekly resample of Mon-Fri (all same week) should give 1 row with last value = 5.0
        assert len(df) == 1
        assert df["TEST"].iloc[0] == 5.0


# ---------------------------------------------------------------------------
# Tests: date window
# ---------------------------------------------------------------------------

class TestDateWindow:
    def test_start_bound(self, macro_corpus):
        """start bound is respected."""
        df = load_macro(
            series=["DGS10"],
            start="2024-06-01",
            frequency="daily",
            data_dir=macro_corpus["data_dir"],
        )
        assert df.index.min() >= pd.Timestamp("2024-06-01")

    def test_end_bound(self, macro_corpus):
        """end bound is respected."""
        df = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            end="2024-03-31",
            frequency="daily",
            data_dir=macro_corpus["data_dir"],
        )
        assert df.index.max() <= pd.Timestamp("2024-03-31")

    def test_both_bounds(self, macro_corpus):
        """Both start and end bounds applied."""
        df = load_macro(
            series=["DGS10"],
            start="2024-03-01",
            end="2024-03-31",
            frequency="daily",
            data_dir=macro_corpus["data_dir"],
        )
        assert df.index.min() >= pd.Timestamp("2024-03-01")
        assert df.index.max() <= pd.Timestamp("2024-03-31")

    def test_end_none(self, macro_corpus):
        """end=None returns through last row."""
        df = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            end=None,
            frequency="daily",
            data_dir=macro_corpus["data_dir"],
        )
        assert len(df) == 250  # all trading days


# ---------------------------------------------------------------------------
# Tests: empty corpus
# ---------------------------------------------------------------------------

class TestEmptyCorpus:
    def test_missing_corpus_directory(self, tmp_path):
        """No macro_corpus dir -> empty DF + warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = load_macro(
                series=["DGS10"],
                start="2024-01-01",
                frequency="monthly",
                data_dir=str(tmp_path),
            )
            assert df.empty
            has_warning = any("not found" in str(warning.message) for warning in w)
            assert has_warning

    def test_no_series_loaded(self, tmp_path):
        """All series missing -> empty DF + warning."""
        corpus_dir = os.path.join(str(tmp_path), "macro_corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = load_macro(
                series=["NONEXISTENT"],
                start="2024-01-01",
                frequency="monthly",
                data_dir=str(tmp_path),
            )
            assert df.empty
            has_warning = any("No FRED series" in str(warning.message) for warning in w)
            assert has_warning

    def test_empty_series_list(self, macro_corpus):
        """Empty series list returns empty DataFrame."""
        df = load_macro(
            series=[],
            start="2024-01-01",
            frequency="monthly",
            data_dir=macro_corpus["data_dir"],
        )
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_idempotent(self, macro_corpus):
        """Same inputs produce same outputs."""
        df1 = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            frequency="monthly",
            data_dir=macro_corpus["data_dir"],
        )
        df2 = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            frequency="monthly",
            data_dir=macro_corpus["data_dir"],
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_nan_values(self, macro_corpus):
        """Resampled DataFrame should have no NaN values."""
        df = load_macro(
            series=["DGS10", "DGS2"],
            start="2024-01-01",
            frequency="monthly",
            data_dir=macro_corpus["data_dir"],
        )
        assert not df.isna().any().any()

    def test_quarterly_resample(self, macro_corpus):
        """Quarterly resampling works."""
        df = load_macro(
            series=["DGS10"],
            start="2024-01-01",
            frequency="quarterly",
            data_dir=macro_corpus["data_dir"],
        )
        assert len(df) >= 1
        # 2024 = 4 quarters or fewer depending on date range
        assert len(df) <= 4


# ---------------------------------------------------------------------------
# Tests: frequency map
# ---------------------------------------------------------------------------

class TestFrequencyMap:
    def test_all_frequencies_mapped(self):
        """All four frequencies map to valid pandas offsets."""
        for freq in ["daily", "weekly", "monthly", "quarterly"]:
            offset = _FREQ_MAP[freq]
            # Should be a non-empty string
            assert offset and isinstance(offset, str)
            # Should be parseable by pandas
            pd.tseries.frequencies.to_offset(offset)
