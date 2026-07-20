#!/usr/bin/env python3
"""
Multi-symbol OHLCV data adapter for sandbox-alpha v2.

Phase 0 PR-C: reads the runner's per-symbol CSV cache and returns
aligned pandas DataFrames ready for cross-sectional signal code.

Dependencies: pandas, numpy (already in the runner image).

Design choices
--------------
- Pure functions: no I/O side effects beyond reading CSVs.
- Idempotent: same inputs always yield same outputs.
- Chunked reading for large universes (up to 500 symbols).
- Forward-fill policy: within each symbol, fill up to 1 consecutive
  missing day; dates where ANY symbol still has NaN are dropped.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MissingDataError(FileNotFoundError):
    """Raised when a required symbol's CSV is absent from data_dir."""

    def __init__(self, symbol: str, data_dir: str = ""):
        self.symbol = symbol
        self.data_dir = data_dir
        msg = f"Missing OHLCV data for symbol '{symbol}'"
        if data_dir:
            msg += f" (expected: {os.path.join(data_dir, f'{symbol}.csv')})"
        msg += ". The runner's ensure_data path must populate the cache before calling the adapter."
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Expected column names (matches existing engine convention)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns are exactly Open/High/Low/Close/Volume.

    yfinance sometimes writes 'Adj Close' or lowercase variants;
    we normalise to the engine standard.
    """
    # Map common variations
    rename_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low == "open":
            rename_map[col] = "Open"
        elif low == "high":
            rename_map[col] = "High"
        elif low == "low":
            rename_map[col] = "Low"
        elif low == "close":
            rename_map[col] = "Close"
        elif low in ("volume", "vol"):
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)

    # Keep only required columns (drop Adj Close etc.)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df[REQUIRED_COLUMNS]


# ---------------------------------------------------------------------------
# load_ohlcv
# ---------------------------------------------------------------------------

def load_ohlcv(
    universe: List[str],
    start: str,
    end: Optional[str] = None,
    data_dir: str = "/data",
) -> Dict[str, pd.DataFrame]:
    """Load per-symbol OHLCV CSVs and return a dict of DataFrames.

    Parameters
    ----------
    universe : list[str]
        Ticker symbols to load (e.g. ["AAPL", "MSFT", "GOOG"]).
    start : str
        Inclusive start date, ISO format "YYYY-MM-DD".
    end : str or None
        Inclusive end date "YYYY-MM-DD". None = through last row.
    data_dir : str
        Directory containing per-symbol CSVs (``{symbol}.csv``).

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{symbol: df}`` where each df has a DatetimeIndex (name 'Date')
        and columns ``Open, High, Low, Close, Volume``.

    Raises
    ------
    MissingDataError
        If a symbol's CSV file is absent from ``data_dir``.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None

    result: Dict[str, pd.DataFrame] = {}

    for symbol in universe:
        path = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.isfile(path):
            raise MissingDataError(symbol, data_dir)

        # Read CSV — first column is assumed to be the date index.
        # utc=True normalises mixed-tz CSVs (e.g. crypto has UTC offset,
        # equities are naive) so downstream concat/reindex across symbols
        # doesn't raise "Mixed timezones detected". tz_convert(None) keeps
        # the wall-clock date because the rest of the pipeline is date-only.
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        df.index.name = "Date"

        # Canonicalize columns
        df = _canonicalize_columns(df)

        # Ensure numeric dtypes
        for col in REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by date (idempotent)
        df = df.sort_index()

        # Slice to [start, end] inclusive
        if end_ts is not None:
            df = df.loc[start_ts:end_ts]
        else:
            df = df.loc[start_ts:]

        result[symbol] = df

    return result


# ---------------------------------------------------------------------------
# align_universe
# ---------------------------------------------------------------------------

def align_universe(
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align multiple symbol DataFrames on a common DatetimeIndex.

    Forward-fill policy
    -------------------
    1. Within each symbol, forward-fill up to 1 consecutive missing day
       (i.e. limit=1).
    2. After ffill, any date where ANY symbol still has NaN is dropped.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Output of :func:`load_ohlcv`.

    Returns
    -------
    pd.DataFrame
        Wide-format panel with MultiIndex columns ``(symbol, field)``.
        Index is the intersection DatetimeIndex after ffill + drop.
    """
    if not data:
        return pd.DataFrame()

    symbols = sorted(data.keys())

    # Step 1: Build UNION of all date indices across symbols
    union_idx = data[symbols[0]].index
    for sym in symbols[1:]:
        union_idx = union_idx.union(data[sym].index)
    union_idx = union_idx.sort_values()

    # Step 2: Reindex each symbol to the union (NaN for missing dates)
    # Step 3: Forward-fill with limit=1 (fills at most 1 consecutive NaN)
    reindexed = {}
    for sym in symbols:
        df = data[sym].reindex(union_idx)
        df = df.ffill(limit=1)
        reindexed[sym] = df

    # Step 4: Build wide panel with MultiIndex columns (symbol, field)
    frames = []
    for sym in symbols:
        df = reindexed[sym].copy()
        df.columns = pd.MultiIndex.from_product([[sym], df.columns], names=["symbol", "field"])
        frames.append(df)

    panel = pd.concat(frames, axis=1)

    # Step 5: Drop dates where ANY symbol still has NaN
    panel = panel.dropna(how="any", axis=0)

    return panel


# ---------------------------------------------------------------------------
# to_wide
# ---------------------------------------------------------------------------

def to_wide(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convenience wrapper: load + align in one call.

    Accepts the raw dict from :func:`load_ohlcv` and returns the same
    wide panel as :func:`align_universe`.

    This exists because cross-sectional signal code often just needs::

        panel = to_wide(load_ohlcv(universe, start, end, data_dir))
        returns = panel.xs("Close", level="field", axis=1).pct_change()
    """
    return align_universe(data)
