"""
Tests for ideation-v3 cross-sectional support — _expand_universe_refs (PR 4e).
"""
from unittest.mock import patch

import pytest

from strategy_ideation import _expand_universe_refs

_FAKE_SYMBOLS = ["A", "B", "C", "D", "E"]


def test_expand_universe_ref_replaces_with_universe_list():
    """universe_ref is consumed and replaced by universe list."""
    manifest_dict = {
        "name": "test_manifest",
        "data_sources": [
            {"type": "ohlcv", "universe_ref": "russell1000_top5", "start": "2020-01-01"}
        ],
    }
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        _expand_universe_refs(manifest_dict)

    ds = manifest_dict["data_sources"][0]
    assert "universe_ref" not in ds
    assert ds["universe"] == _FAKE_SYMBOLS


def test_expand_universe_ref_unknown_alias():
    """Unknown alias raises ValueError."""
    manifest_dict = {
        "name": "bad_manifest",
        "data_sources": [
            {"type": "ohlcv", "universe_ref": "sp500", "start": "2020-01-01"}
        ],
    }
    with pytest.raises(ValueError, match="Unknown universe alias"):
        _expand_universe_refs(manifest_dict)


def test_universe_ref_wins_over_existing_universe():
    """Both universe_ref and universe present — universe_ref wins."""
    manifest_dict = {
        "name": "test_manifest",
        "data_sources": [
            {
                "type": "ohlcv",
                "universe_ref": "russell1000_top5",
                "universe": ["OLD", "SYMBOLS"],
                "start": "2020-01-01",
            }
        ],
    }
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        _expand_universe_refs(manifest_dict)

    ds = manifest_dict["data_sources"][0]
    assert "universe_ref" not in ds
    assert ds["universe"] == _FAKE_SYMBOLS


def test_no_universe_ref_untouched():
    """Data sources without universe_ref are left unchanged."""
    manifest_dict = {
        "name": "test_manifest",
        "data_sources": [
            {"type": "ohlcv", "universe": ["AAPL"], "start": "2020-01-01"}
        ],
    }
    with patch("data_adapters.universe.UniverseProvider.get_symbols") as mock_get:
        _expand_universe_refs(manifest_dict)
        mock_get.assert_not_called()

    assert manifest_dict["data_sources"][0]["universe"] == ["AAPL"]


def test_multiple_data_sources():
    """Only the source with universe_ref gets expanded."""
    manifest_dict = {
        "name": "test_manifest",
        "data_sources": [
            {"type": "ohlcv", "universe_ref": "russell1000_top5", "start": "2020-01-01"},
            {"type": "ohlcv", "universe": ["SPY"], "start": "2019-01-01"},
        ],
    }
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        _expand_universe_refs(manifest_dict)

    assert "universe_ref" not in manifest_dict["data_sources"][0]
    assert manifest_dict["data_sources"][0]["universe"] == _FAKE_SYMBOLS
    assert manifest_dict["data_sources"][1]["universe"] == ["SPY"]


def test_non_dict_data_source_skipped():
    """Non-dict entries in data_sources are silently skipped."""
    manifest_dict = {
        "name": "test_manifest",
        "data_sources": [
            "not a dict",
            {"type": "ohlcv", "universe_ref": "russell1000_top5", "start": "2020-01-01"},
        ],
    }
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        _expand_universe_refs(manifest_dict)

    assert manifest_dict["data_sources"][1]["universe"] == _FAKE_SYMBOLS


def test_empty_data_sources():
    """No data_sources key — no-op."""
    manifest_dict: dict = {"name": "test_manifest"}
    # Should not raise
    _expand_universe_refs(manifest_dict)
