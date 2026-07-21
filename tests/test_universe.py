#!/usr/bin/env python3
"""
Tests for data_adapters.universe — UniverseProvider.

All tests mock the Wikipedia scrape (no live network calls).
"""

import csv
import os
import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data_adapters.universe import UniverseProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(manifest_dir: str, name: str, date_str: str, rows: list):
    """Write a manifest CSV for a given universe name and date."""
    path = os.path.join(manifest_dir, f"{name}_{date_str}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "name", "valid_from", "valid_until"])
        for row in rows:
            writer.writerow(row)
    return path


def _setup_mock_refresh(monkeypatch, fake_table):
    """Mock both requests.get and pd.read_html for refresh tests."""
    # Mock requests.get to return a fake response
    mock_response = MagicMock()
    mock_response.text = "<html></html>"
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr("requests.get", lambda url, headers=None, timeout=None: mock_response)
    monkeypatch.setattr("data_adapters.universe.pd.read_html", lambda text: [fake_table])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUniverseProviderBootstrap:
    """Bootstrap and directory creation tests."""

    def test_creates_manifest_dir(self, tmp_path):
        """UniverseProvider creates the manifest directory if absent."""
        md = os.path.join(str(tmp_path), "manifests")
        assert not os.path.isdir(md)
        provider = UniverseProvider(name="russell1000", manifest_dir=md)
        assert os.path.isdir(md)
        manifests = provider._list_manifests()
        assert manifests == []


class TestUniverseProviderLoad:
    """Tests that read existing manifests."""

    def test_reads_existing_manifest(self, tmp_path):
        """get_symbols returns expected symbols from a pre-written manifest."""
        md = str(tmp_path)
        _write_manifest(md, "russell1000", "2026-01-15", [
            ("AAPL", "Apple Inc.", "2026-01-15", ""),
            ("MSFT", "Microsoft Corp.", "2026-01-15", ""),
            ("GOOG", "Alphabet Inc.", "2026-01-15", ""),
        ])
        provider = UniverseProvider(manifest_dir=md)
        symbols = provider.get_symbols(as_of="2026-01-15")
        assert sorted(symbols) == ["AAPL", "GOOG", "MSFT"]

    def test_as_of_selects_correct_manifest(self, tmp_path):
        """With 3 dated manifests, as_of selects the newest one <= as_of date."""
        md = str(tmp_path)
        _write_manifest(md, "russell1000", "2026-01-01", [("A", "A Co", "2026-01-01", "")])
        _write_manifest(md, "russell1000", "2026-02-01", [("B", "B Co", "2026-02-01", "")])
        _write_manifest(md, "russell1000", "2026-03-01", [("C", "C Co", "2026-03-01", "")])

        provider = UniverseProvider(manifest_dir=md)

        # as_of=2026-01-15 → should pick 2026-01-01
        symbols = provider.get_symbols(as_of="2026-01-15")
        assert symbols == ["A"]

        # as_of=2026-02-15 → should pick 2026-02-01
        symbols = provider.get_symbols(as_of="2026-02-15")
        assert symbols == ["B"]

        # as_of=2026-03-15 → should pick 2026-03-01
        symbols = provider.get_symbols(as_of="2026-03-15")
        assert symbols == ["C"]

        # as_of=2025-12-31 → no manifest covers this
        path = provider._manifest_for_date("2025-12-31")
        assert path is None


class TestUniverseProviderRefresh:
    """Tests for refresh_from_source with mocked Wikipedia scrape."""

    def test_refresh_writes_dated_file(self, tmp_path, monkeypatch):
        """Mock pandas.read_html and verify a manifest is written with today's date."""
        md = str(tmp_path)
        today = datetime.date.today().isoformat()

        # Build a fake Wikipedia table
        fake_table = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "GOOG"],
            "Company": ["Apple Inc.", "Microsoft Corp.", "Alphabet Inc."],
        })

        _setup_mock_refresh(monkeypatch, fake_table)

        provider = UniverseProvider(manifest_dir=md)
        path = provider.refresh_from_source()

        expected_path = os.path.join(md, f"russell1000_{today}.csv")
        assert path == expected_path
        assert os.path.isfile(expected_path)

        # Verify content
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            symbols = {r["symbol"] for r in rows}
            assert symbols == {"AAPL", "MSFT", "GOOG"}
            for r in rows:
                assert r["valid_from"] == today or r["valid_from"] != ""
                assert "symbol" in r
                assert "name" in r

        # Verify load_constituents returns correct data
        provider2 = UniverseProvider(manifest_dir=md)
        constituents = provider2.load_constituents(as_of=today)
        assert len(constituents) == 3
        symbols = {c["symbol"] for c in constituents}
        assert symbols == {"AAPL", "MSFT", "GOOG"}

    def test_refresh_deduplicates_by_symbol(self, tmp_path, monkeypatch):
        """Duplicate ticker rows in Wikipedia table → first occurrence wins."""
        md = str(tmp_path)

        fake_table = pd.DataFrame({
            "Ticker": ["AAPL", "AAPL", "MSFT"],
            "Company": ["Apple Inc.", "Apple Computer", "Microsoft Corp."],
        })

        _setup_mock_refresh(monkeypatch, fake_table)

        provider = UniverseProvider(manifest_dir=md)
        path = provider.refresh_from_source()

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            symbols = [r["symbol"] for r in rows]
            assert symbols.count("AAPL") == 1
            assert symbols.count("MSFT") == 1
            assert len(rows) == 2

            # First occurrence should be "Apple Inc."
            for r in rows:
                if r["symbol"] == "AAPL":
                    assert r["name"] == "Apple Inc."

    def test_refresh_no_ticker_column_raises(self, tmp_path, monkeypatch):
        """If no ticker/symbol column found, ValueError is raised."""
        md = str(tmp_path)
        fake_table = pd.DataFrame({
            "Security": ["AAPL", "MSFT"],
            "Industry": ["Tech", "Tech"],
        })

        _setup_mock_refresh(monkeypatch, fake_table)

        provider = UniverseProvider(manifest_dir=md)
        with pytest.raises(ValueError, match="Could not find a 'Ticker' or 'Symbol' column"):
            provider.refresh_from_source()
