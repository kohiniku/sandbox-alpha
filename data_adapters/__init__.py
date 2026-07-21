"""
Data adapters for sandbox-alpha v2.

Each adapter reads from the runner's cached data volume and returns
pandas objects aligned with the existing engine conventions
(DatetimeIndex, columns Open/High/Low/Close/Volume).
"""

from data_adapters.ohlcv import (
    MissingDataError,
    align_universe,
    align_universe_chunked,
    load_ohlcv,
    to_wide,
)
from data_adapters.news_sentiment import load_news_sentiment
from data_adapters.sec_13f import load_sec_13f
from data_adapters.insider import load_insider_trades
from data_adapters.macro import load_macro
from data_adapters.universe import UniverseProvider
from data_adapters.panel_loader import load_panel, panel_coverage_report

__all__ = [
    "MissingDataError",
    "UniverseProvider",
    "align_universe",
    "align_universe_chunked",
    "load_insider_trades",
    "load_macro",
    "load_news_sentiment",
    "load_ohlcv",
    "load_panel",
    "panel_coverage_report",
    "load_sec_13f",
    "to_wide",
]
