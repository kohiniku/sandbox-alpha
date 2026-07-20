"""Tests for data_adapters/sec_13f.py (Phase 2 PR-I)."""

import json
import os

import pandas as pd
import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_adapters.sec_13f import load_sec_13f


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_corpus(tmp_path, records=None):
    """Write a minimal SEC 13F JSONL corpus for testing."""
    if records is None:
        records = [
            {
                "quarter_end": "2024-12-31",
                "cik": "0001067983",
                "filer_name": "Berkshire Hathaway Inc",
                "ticker": "AAPL",
                "shares": 300000000,
                "value_usd": 45000000000.0,
                "pct_of_aum": 12.5,
            },
            {
                "quarter_end": "2024-12-31",
                "cik": "0001341439",
                "filer_name": "Vanguard Group Inc",
                "ticker": "AAPL",
                "shares": 1200000000,
                "value_usd": 180000000000.0,
                "pct_of_aum": 3.2,
            },
            {
                "quarter_end": "2024-12-31",
                "cik": "0001341439",
                "filer_name": "Vanguard Group Inc",
                "ticker": "MSFT",
                "shares": 900000000,
                "value_usd": 150000000000.0,
                "pct_of_aum": 2.7,
            },
            {
                "quarter_end": "2024-12-31",
                "cik": "0000036405",
                "filer_name": "BlackRock Inc",
                "ticker": "GOOGL",
                "shares": 350000000,
                "value_usd": 45000000000.0,
                "pct_of_aum": 0.6,
            },
            {
                "quarter_end": "2024-12-31",
                "cik": "0000036405",
                "filer_name": "BlackRock Inc",
                "ticker": "TSLA",
                "shares": 100000000,
                "value_usd": 25000000000.0,
                "pct_of_aum": 0.33,
            },
            {
                "quarter_end": "2025-03-31",
                "cik": "0001067983",
                "filer_name": "Berkshire Hathaway Inc",
                "ticker": "AAPL",
                "shares": 280000000,
                "value_usd": 42000000000.0,
                "pct_of_aum": 10.8,
            },
            {
                "quarter_end": "2025-03-31",
                "cik": "0001341439",
                "filer_name": "Vanguard Group Inc",
                "ticker": "MSFT",
                "shares": 910000000,
                "value_usd": 155000000000.0,
                "pct_of_aum": 2.75,
            },
        ]

    corpus_dir = os.path.join(tmp_path, "sec_13f_corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    out_path = os.path.join(corpus_dir, "2024-Q4.jsonl")
    with open(out_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadSec13F:
    def test_basic_loading(self, tmp_path):
        """Basic load with no filters."""
        _write_corpus(str(tmp_path))
        df = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.0,
            data_dir=str(tmp_path),
        )
        assert len(df) > 0
        expected_cols = [
            "quarter_end", "cik", "filer_name", "ticker",
            "shares", "value_usd", "pct_of_aum",
        ]
        assert list(df.columns) == expected_cols

    def test_min_position_pct_filter(self, tmp_path):
        """min_position_pct filters out small positions."""
        _write_corpus(str(tmp_path))
        df_all = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.0,
            data_dir=str(tmp_path),
        )
        df_filtered = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.5,
            data_dir=str(tmp_path),
        )
        # With min_position_pct=0.5, TSLA (0.33%) should be dropped
        assert len(df_filtered) < len(df_all)
        assert (df_filtered["pct_of_aum"] >= 0.5).all()

    def test_universe_filter(self, tmp_path):
        """Universe filter restricts to matching tickers."""
        _write_corpus(str(tmp_path))
        df = load_sec_13f(
            universe=["AAPL"],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.0,
            data_dir=str(tmp_path),
        )
        assert len(df) > 0
        assert set(df["ticker"].unique()) == {"AAPL"}

    def test_filer_filter(self, tmp_path):
        """Filer filter restricts to matching CIKs."""
        _write_corpus(str(tmp_path))
        df = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=["0001067983"],
            min_position_pct=0.0,
            data_dir=str(tmp_path),
        )
        assert len(df) > 0
        assert set(df["cik"].unique()) == {"0001067983"}

    def test_top_50_shortcut(self, tmp_path):
        """top_50 shortcut expands to known CIKs."""
        _write_corpus(str(tmp_path))
        df = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=["top_50"],
            min_position_pct=0.0,
            data_dir=str(tmp_path),
        )
        # All three filers in the fixture are in top_50
        assert len(df) > 0
        assert set(df["cik"].unique()).issubset({"0001067983", "0001341439", "0000036405"})

    def test_date_range_filter(self, tmp_path):
        """Date range filter works correctly."""
        _write_corpus(str(tmp_path))
        df = load_sec_13f(
            universe=[],
            start="2025-01-01",
            end="2025-03-31",
            filers=[],
            min_position_pct=0.0,
            data_dir=str(tmp_path),
        )
        assert len(df) > 0
        for d in df["quarter_end"]:
            assert "2025" in str(d)

    def test_empty_corpus_returns_empty_df(self, tmp_path):
        """Missing corpus dir returns empty DataFrame gracefully."""
        df = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.3,
            data_dir=str(tmp_path),
        )
        assert df.empty
        expected_cols = [
            "quarter_end", "cik", "filer_name", "ticker",
            "shares", "value_usd", "pct_of_aum",
        ]
        assert list(df.columns) == expected_cols

    def test_empty_corpus_no_jsonl_files(self, tmp_path):
        """Empty corpus dir returns empty DataFrame."""
        corpus_dir = os.path.join(str(tmp_path), "sec_13f_corpus")
        os.makedirs(corpus_dir, exist_ok=True)
        df = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.3,
            data_dir=str(tmp_path),
        )
        assert df.empty

    def test_quarter_end_datetime_type(self, tmp_path):
        """quarter_end column is DatetimeIndex-compatible UTC naive."""
        _write_corpus(str(tmp_path))
        df = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.0,
            data_dir=str(tmp_path),
        )
        assert pd.api.types.is_datetime64_any_dtype(df["quarter_end"])

    def test_empty_corpus_logs_warning(self, tmp_path, caplog):
        """Missing corpus dir logs a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.3,
            data_dir=str(tmp_path),
        )
        assert "SEC 13F corpus directory not found" in caplog.text

    def test_no_universe_all_rows_included(self, tmp_path):
        """Empty universe means all tickers pass through."""
        _write_corpus(str(tmp_path))
        df = load_sec_13f(
            universe=[],
            start="2024-01-01",
            end=None,
            filers=[],
            min_position_pct=0.0,
            data_dir=str(tmp_path),
        )
        tickers = set(df["ticker"].unique())
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers
