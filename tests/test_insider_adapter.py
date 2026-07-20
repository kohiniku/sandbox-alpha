"""Tests for data_adapters/insider.py (Phase 2 PR-J)."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_adapters.insider import load_insider_trades


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_corpus(tmp_path, trades=None):
    """Write a minimal insider trades JSONL corpus for testing."""
    if trades is None:
        trades = [
            {
                "transaction_date": "2025-01-15",
                "ticker": "AAPL",
                "insider_name": "Tim Cook",
                "role": "CEO",
                "transaction_type": "Sale",
                "shares": 10000,
                "price": 150.25,
                "value_usd": 1502500.00,
            },
            {
                "transaction_date": "2025-02-10",
                "ticker": "MSFT",
                "insider_name": "Satya Nadella",
                "role": "CEO",
                "transaction_type": "Sale",
                "shares": 5000,
                "price": 420.00,
                "value_usd": 2100000.00,
            },
            {
                "transaction_date": "2025-03-01",
                "ticker": "AAPL",
                "insider_name": "Luca Maestri",
                "role": "CFO",
                "transaction_type": "Purchase",
                "shares": 2000,
                "price": 145.00,
                "value_usd": 290000.00,
            },
            {
                "transaction_date": "2025-03-20",
                "ticker": "GOOG",
                "insider_name": "John Doe",
                "role": "Director",
                "transaction_type": "Purchase",
                "shares": 100,
                "price": 180.00,
                "value_usd": 18000.00,
            },
            {
                "transaction_date": "2025-04-01",
                "ticker": "TSLA",
                "insider_name": "Jane Smith",
                "role": "10%_owner",
                "transaction_type": "Sale",
                "shares": 50000,
                "price": 250.00,
                "value_usd": 12500000.00,
            },
            {
                "transaction_date": "2025-04-05",
                "ticker": "AMZN",
                "insider_name": "Bob Wilson",
                "role": "Other",
                "transaction_type": "Purchase",
                "shares": 10,
                "price": 200.00,
                "value_usd": 2000.00,
            },
        ]

    corpus_dir = os.path.join(str(tmp_path), "insider_corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    out_path = os.path.join(corpus_dir, "2025-Q1.jsonl")
    with open(out_path, "w") as fh:
        for trade in trades:
            fh.write(json.dumps(trade) + "\n")


# ---------------------------------------------------------------------------
# load_insider_trades tests
# ---------------------------------------------------------------------------

class TestLoadInsiderTrades:
    def test_basic_loading(self, tmp_path):
        """Basic load with no filters."""
        _write_corpus(str(tmp_path))
        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        assert len(df) == 6
        expected_cols = [
            "transaction_date", "ticker", "insider_name", "role",
            "transaction_type", "shares", "price", "value_usd",
        ]
        assert list(df.columns) == expected_cols

    def test_min_transaction_usd_filter(self, tmp_path):
        """min_transaction_usd filters out small transactions."""
        _write_corpus(str(tmp_path))
        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=100000,
            roles=None,
            data_dir=str(tmp_path),
        )
        # Only trades >= $100k: AAPL(1.5M), MSFT(2.1M), AAPL(290k), TSLA(12.5M)
        assert len(df) == 4
        assert (df["value_usd"] >= 100000).all()

    def test_universe_filter(self, tmp_path):
        """Universe filter restricts to matching tickers."""
        _write_corpus(str(tmp_path))
        df = load_insider_trades(
            universe=["AAPL"],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        assert len(df) == 2
        assert set(df["ticker"].unique()) == {"AAPL"}

    def test_roles_filter(self, tmp_path):
        """Roles filter selects specific insider roles."""
        _write_corpus(str(tmp_path))
        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=["CEO", "CFO"],
            data_dir=str(tmp_path),
        )
        assert len(df) == 3
        assert set(df["role"].unique()) == {"CEO", "CFO"}

    def test_date_range_filter(self, tmp_path):
        """Date range filter works correctly."""
        _write_corpus(str(tmp_path))
        df = load_insider_trades(
            universe=[],
            start="2025-03-01",
            end="2025-03-31",
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        assert len(df) == 2
        for d in df["transaction_date"]:
            assert "2025-03" in str(d)

    def test_combined_filters(self, tmp_path):
        """All filters combined work together."""
        _write_corpus(str(tmp_path))
        df = load_insider_trades(
            universe=["AAPL", "GOOG"],
            start="2025-01-01",
            end="2025-06-30",
            min_transaction_usd=50000,
            roles=["CFO", "Director"],
            data_dir=str(tmp_path),
        )
        # Should only match: Luca Maestri (AAPL, CFO, $290k) and John Doe (GOOG, Director, $18k=excluded)
        assert len(df) == 1
        assert df.iloc[0]["insider_name"] == "Luca Maestri"

    def test_empty_corpus_returns_empty_df(self, tmp_path):
        """Missing corpus dir returns empty DataFrame gracefully."""
        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        assert df.empty
        expected_cols = [
            "transaction_date", "ticker", "insider_name", "role",
            "transaction_type", "shares", "price", "value_usd",
        ]
        assert list(df.columns) == expected_cols

    def test_empty_jsonl_files(self, tmp_path):
        """Empty corpus dir with no jsonl files returns empty DataFrame."""
        corpus_dir = os.path.join(str(tmp_path), "insider_corpus")
        os.makedirs(corpus_dir, exist_ok=True)
        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        assert df.empty

    def test_date_index_compatible(self, tmp_path):
        """transaction_date column is DatetimeIndex-compatible."""
        _write_corpus(str(tmp_path))
        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        assert pd.api.types.is_datetime64_any_dtype(df["transaction_date"])

    def test_empty_corpus_logs_warning(self, tmp_path, caplog):
        """Missing corpus dir logs a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        assert "Insider corpus directory not found" in caplog.text

    def test_malformed_json_skipped(self, tmp_path, caplog):
        """Malformed JSON lines are skipped with a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        corpus_dir = os.path.join(str(tmp_path), "insider_corpus")
        os.makedirs(corpus_dir, exist_ok=True)
        out_path = os.path.join(corpus_dir, "bad.jsonl")
        with open(out_path, "w") as fh:
            fh.write(json.dumps({
                "transaction_date": "2025-01-15",
                "ticker": "AAPL",
                "insider_name": "Good Trade",
                "role": "CEO",
                "transaction_type": "Purchase",
                "shares": 100,
                "price": 150.0,
                "value_usd": 15000.0,
            }) + "\n")
            fh.write("not valid json {{{{{\n")
            fh.write(json.dumps({
                "transaction_date": "2025-02-01",
                "ticker": "MSFT",
                "insider_name": "Good Trade 2",
                "role": "CFO",
                "transaction_type": "Sale",
                "shares": 200,
                "price": 400.0,
                "value_usd": 80000.0,
            }) + "\n")

        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        assert len(df) == 2
        assert any("Skipping malformed JSON" in rec.message for rec in caplog.records)

    def test_default_min_transaction_usd(self, tmp_path):
        """Default min_transaction_usd is 10000."""
        _write_corpus(str(tmp_path))
        # With default 10k: only the $2000 trade should be filtered
        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            roles=None,
            data_dir=str(tmp_path),
        )
        # AMZN $2000 trade excluded
        assert len(df) == 5
        assert (df["value_usd"] >= 10000).all()

    def test_sort_order(self, tmp_path):
        """Results are sorted by transaction_date then ticker."""
        _write_corpus(str(tmp_path))
        df = load_insider_trades(
            universe=[],
            start="2025-01-01",
            end=None,
            min_transaction_usd=0,
            roles=None,
            data_dir=str(tmp_path),
        )
        # Verify sorted by date
        dates = df["transaction_date"].tolist()
        assert dates == sorted(dates)
