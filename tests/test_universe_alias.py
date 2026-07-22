"""
Tests for data_adapters.universe — resolve_universe_alias (PR 4e).
"""
from unittest.mock import patch

import pytest

from data_adapters.universe import resolve_universe_alias

_FAKE_SYMBOLS = ["AAPL", "BRK.B", "GOOGL", "JPM", "JNJ", "META", "MSFT", "NVDA", "TSLA", "V"]


def test_resolve_full():
    """resolve_universe_alias('russell1000') returns all symbols."""
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        result = resolve_universe_alias("russell1000")
    assert result == _FAKE_SYMBOLS


def test_resolve_top5():
    """resolve_universe_alias('russell1000_top5') returns first 5 symbols."""
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        result = resolve_universe_alias("russell1000_top5")
    assert result == _FAKE_SYMBOLS[:5]


def test_resolve_top100():
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        result = resolve_universe_alias("russell1000_top100")
    assert len(result) == 10  # all 10 returned (less than 100 available)


def test_resolve_top500():
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        result = resolve_universe_alias("russell1000_top500")
    assert len(result) == 10


def test_resolve_top200():
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        result = resolve_universe_alias("russell1000_top200")
    assert len(result) == 10


def test_resolve_top50():
    with patch("data_adapters.universe.UniverseProvider.get_symbols", return_value=_FAKE_SYMBOLS):
        result = resolve_universe_alias("russell1000_top50")
    assert len(result) == 10


def test_unknown_alias_raises_valueerror():
    """Unknown alias raises ValueError."""
    with pytest.raises(ValueError, match="Unknown universe alias"):
        resolve_universe_alias("sp500")


def test_invalid_top_alias_raises_valueerror():
    """Malformed top alias raises ValueError."""
    with pytest.raises(ValueError, match="Invalid universe alias"):
        resolve_universe_alias("russell1000_topXYZ")
