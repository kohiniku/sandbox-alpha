#!/usr/bin/env python3
"""
FRED macro data adapter for sandbox-alpha v2.

Phase 2 PR-K: reads pre-cached FRED series CSVs from a macro corpus
directory and returns a resampled pandas DataFrame ready for strategy
signal code.

Dependencies: pandas, numpy (already in the runner image).

Design choices
--------------
- Pure functions: no I/O side effects beyond reading CSVs.
- CSV format: ``DATE,VALUE`` (FRED standard download).
- Empty corpus is graceful: empty DataFrame + warning, no exception.
- Resampling: ``.resample(freq).last()`` — takes the last observed
  value in each period, standard for macro indicators.
"""

import os
import warnings
from typing import List, Optional

import pandas as pd

_FREQ_MAP = {
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",
    "quarterly": "QE",
}

# Fallback for older pandas where 'ME'/'QE' don't exist
try:
    pd.tseries.frequencies.to_offset("ME")
except ValueError:
    _FREQ_MAP["monthly"] = "M"
try:
    pd.tseries.frequencies.to_offset("QE")
except ValueError:
    _FREQ_MAP["quarterly"] = "Q"


def load_macro(
    series: List[str],
    start: str,
    end: Optional[str] = None,
    frequency: str = "monthly",
    data_dir: str = "/data",
) -> pd.DataFrame:
    """Load pre-cached FRED series CSVs and return a resampled DataFrame.

    Parameters
    ----------
    series : list[str]
        FRED series IDs (e.g. ``["DGS10", "DGS2", "UNRATE"]``).
    start : str
        Inclusive start date, ISO format ``"YYYY-MM-DD"``.
    end : str or None
        Inclusive end date. None = through last row.
    frequency : str
        One of ``{"daily", "weekly", "monthly", "quarterly"}``.
    data_dir : str
        Base directory. CSVs expected at ``{data_dir}/macro_corpus/{id}.csv``.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (resampled to *frequency*). Columns: *series* IDs.
        Values are the last observed value within each resample period.
        Empty DataFrame if the corpus directory or all CSVs are absent.
    """
    if not series:
        return pd.DataFrame()

    corpus_dir = os.path.join(data_dir, "macro_corpus")
    if not os.path.isdir(corpus_dir):
        warnings.warn(
            f"Macro corpus directory not found: {corpus_dir}. "
            f"Returning empty DataFrame."
        )
        return pd.DataFrame()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    freq = _FREQ_MAP.get(frequency, "ME")

    frames: List[pd.DataFrame] = []
    loaded_ids: List[str] = []

    for sid in series:
        path = os.path.join(corpus_dir, f"{sid}.csv")
        if not os.path.isfile(path):
            warnings.warn(
                f"FRED series CSV not found: {path}. Skipping '{sid}'."
            )
            continue

        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        # FRED CSVs have a single VALUE column (or the series name).
        # Normalise: take the first data column.
        if "VALUE" in df.columns:
            value_col = "VALUE"
        elif sid in df.columns:
            value_col = sid
        else:
            value_col = df.columns[0]

        df = df[[value_col]].rename(columns={value_col: sid})

        # Slice to [start, end] inclusive
        if end_ts is not None:
            df = df.loc[start_ts:end_ts]
        else:
            df = df.loc[start_ts:]

        # Resample to declared frequency
        if freq != "D":
            df = df.resample(freq).last()

        frames.append(df)
        loaded_ids.append(sid)

    if not frames:
        warnings.warn(
            f"No FRED series CSVs loaded from {corpus_dir}. "
            f"Returning empty DataFrame."
        )
        return pd.DataFrame()

    result = pd.concat(frames, axis=1)
    result = result.sort_index()
    # Drop rows where ALL values are NaN (e.g., after resample before first obs)
    result = result.dropna(how="all")
    return result
