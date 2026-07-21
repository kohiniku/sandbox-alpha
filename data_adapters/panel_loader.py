#!/usr/bin/env python3
"""
Panel data loader for cross-sectional strategies.

PR 4a: Loads per-symbol cached CSVs, returns dict-of-DataFrames
(matching the existing manifest_runner convention). Missing symbols
are skipped with a warning — universe drift is expected.

Dependencies: pandas (already in requirements.txt).
"""

import logging
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

from data_adapters.ohlcv import REQUIRED_COLUMNS, _canonicalize_columns

logger = logging.getLogger(__name__)

# Maximum number of consecutive NaN days to forward-fill.
# Beyond this, we assume a delisting or extended halt.
MAX_FFILL_DAYS = 5


def load_panel(
    symbols: List[str],
    start: str,
    end: str,
    data_dir: str,
    forward_fill: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load per-symbol OHLCV CSVs and return a dict of DataFrames.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to load.
    start : str
        Inclusive start date, ISO format ``YYYY-MM-DD``.
    end : str
        Inclusive end date, ISO format ``YYYY-MM-DD``.
    data_dir : str
        Directory containing ``{symbol}.csv`` files.
    forward_fill : bool
        If True, forward-fill NaN rows within each symbol's frame for gaps
        up to ``MAX_FFILL_DAYS`` trading days. Gaps longer than this are
        left as NaN (likely delisting or extended halt).

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{symbol: df}`` where each df has a DatetimeIndex (name ``Date``)
        and columns ``Open, High, Low, Close, Volume``. Only symbols with
        loadable data are included — missing symbols are skipped.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    result: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        path = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.isfile(path):
            msg = f"[panel_loader] Skipping {symbol}: no CSV at {path}"
            print(msg, file=sys.stderr)
            logger.warning(msg)
            continue

        try:
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            df.index.name = "Date"
        except Exception as exc:
            msg = f"[panel_loader] Skipping {symbol}: read error — {exc}"
            print(msg, file=sys.stderr)
            logger.warning(msg)
            continue

        # Canonicalize and enforce numeric columns
        try:
            df = _canonicalize_columns(df)
            for col in REQUIRED_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        except ValueError as exc:
            msg = f"[panel_loader] Skipping {symbol}: column error — {exc}"
            print(msg, file=sys.stderr)
            logger.warning(msg)
            continue

        # Sort and slice to date range
        df = df.sort_index()
        df = df.loc[start_ts:end_ts]

        if len(df) == 0:
            msg = f"[panel_loader] Skipping {symbol}: no data in [{start}, {end}]"
            print(msg, file=sys.stderr)
            logger.warning(msg)
            continue

        # Forward-fill small gaps (up to MAX_FFILL_DAYS days)
        if forward_fill:
            df = df.ffill(limit=MAX_FFILL_DAYS)

        result[symbol] = df

    return result


def panel_coverage_report(
    loaded: Dict[str, pd.DataFrame],
    requested: List[str],
) -> dict:
    """Produce a coverage summary of what was loaded vs. requested.

    Returns
    -------
    dict
        ``{"requested": N, "loaded": M, "missing": [...], "date_range": (min, max)}``
    """
    loaded_symbols = sorted(loaded.keys())
    missing = [s for s in requested if s not in loaded]

    date_min = None
    date_max = None
    for df in loaded.values():
        if len(df) == 0:
            continue
        dmin = df.index.min()
        dmax = df.index.max()
        if date_min is None or dmin < date_min:
            date_min = dmin
        if date_max is None or dmax > date_max:
            date_max = dmax

    return {
        "requested": len(requested),
        "loaded": len(loaded),
        "missing": missing,
        "date_range": (date_min, date_max),
    }
