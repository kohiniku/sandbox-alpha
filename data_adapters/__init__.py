"""
Data adapters for sandbox-alpha v2.

Each adapter reads from the runner's cached data volume and returns
pandas objects aligned with the existing engine conventions
(DatetimeIndex, columns Open/High/Low/Close/Volume).
"""

from data_adapters.ohlcv import (
    MissingDataError,
    align_universe,
    load_ohlcv,
    to_wide,
)
from data_adapters.news_sentiment import load_news_sentiment
from data_adapters.insider import load_insider_trades

__all__ = [
    "MissingDataError",
    "align_universe",
    "load_insider_trades",
    "load_news_sentiment",
    "load_ohlcv",
    "to_wide",
]
