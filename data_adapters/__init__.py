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
from data_adapters.macro import load_macro

__all__ = [
    "MissingDataError",
    "align_universe",
    "load_macro",
    "load_news_sentiment",
    "load_ohlcv",
    "to_wide",
]
