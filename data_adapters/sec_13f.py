#!/usr/bin/env python3
"""
SEC 13F institutional holdings data adapter for sandbox-alpha v2 (Phase 2 PR-I).

Reads a pre-fetched JSONL corpus from disk and returns a pandas DataFrame
with institutional holdings. No network calls at runtime.

Corpus format (one JSON object per line in *.jsonl):
{
    "quarter_end": "2025-03-31",
    "cik": "0001067983",
    "filer_name": "Berkshire Hathaway Inc",
    "ticker": "AAPL",
    "shares": 300000000,
    "value_usd": 45000000000.0,
    "pct_of_aum": 12.5
}
"""

import json
import logging
import os
from typing import List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_sec_13f(
    universe: List[str],
    start: str,
    end: Optional[str],
    filers: List[str],
    min_position_pct: float,
    data_dir: str,
) -> pd.DataFrame:
    """Load SEC 13F holdings data from a pre-fetched JSONL corpus.

    Parameters
    ----------
    universe : list[str]
        Ticker symbols to filter on. Empty list = no ticker filter (all rows).
    start : str
        Inclusive start date, ISO format "YYYY-MM-DD".
    end : str or None
        Inclusive end date "YYYY-MM-DD". None = through last row.
    filers : list[str]
        CIK list to filter filers. ``['top_50']`` expands to the 50 largest
        asset managers (hardcoded in manifest.TOP_50_CIKS).
    min_position_pct : float
        Minimum position size as % of filer AUM. Rows with pct_of_aum below
        this threshold are silently dropped. Default 0.5.
    data_dir : str
        Root data directory (usually ``/data``). The corpus lives at
        ``{data_dir}/sec_13f_corpus/*.jsonl``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[quarter_end, cik, filer_name, ticker, shares,
        value_usd, pct_of_aum]``.
        - ``quarter_end``: DatetimeIndex-compatible (UTC naive).
        Empty DataFrame if no corpus files are found (graceful degradation).
    """
    corpus_dir = os.path.join(data_dir, "sec_13f_corpus")
    if not os.path.isdir(corpus_dir):
        logger.warning(
            "SEC 13F corpus directory not found: %s. "
            "Returning empty DataFrame (run scripts/ingest_sec_13f.py first).",
            corpus_dir,
        )
        return _empty_df()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    universe_set: Optional[Set[str]] = set(universe) if universe else None

    # Expand 'top_50' shortcut
    filer_set: Optional[Set[str]] = None
    if filers:
        if "top_50" in filers:
            from manifest import TOP_50_CIKS
            filer_set = set(TOP_50_CIKS)
        else:
            filer_set = set(filers)

    rows: list = []
    jsonl_files = sorted(
        f for f in os.listdir(corpus_dir) if f.endswith(".jsonl")
    )

    if not jsonl_files:
        logger.warning(
            "No .jsonl files in %s. Returning empty DataFrame.", corpus_dir
        )
        return _empty_df()

    for filename in jsonl_files:
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON line in %s", filepath
                    )
                    continue

                # Filter by ticker (universe)
                ticker = rec.get("ticker", "")
                if universe_set is not None and ticker not in universe_set:
                    continue

                # Filter by filer CIK
                cik = str(rec.get("cik", ""))
                if filer_set is not None and cik not in filer_set:
                    continue

                # Filter by min_position_pct
                pct = float(rec.get("pct_of_aum", 0.0))
                if pct < min_position_pct:
                    continue

                # Filter by date
                qe = rec.get("quarter_end", "")
                if not qe:
                    continue
                try:
                    qe_ts = pd.Timestamp(qe)
                except (ValueError, TypeError):
                    continue
                if qe_ts < start_ts:
                    continue
                if end_ts is not None and qe_ts > end_ts:
                    continue

                rows.append({
                    "quarter_end": qe_ts,
                    "cik": cik,
                    "filer_name": rec.get("filer_name", ""),
                    "ticker": ticker,
                    "shares": int(rec.get("shares", 0)),
                    "value_usd": float(rec.get("value_usd", 0.0)),
                    "pct_of_aum": pct,
                })

    if not rows:
        return _empty_df()

    df = pd.DataFrame(rows)

    # Ensure quarter_end is DatetimeIndex-compatible UTC naive
    df["quarter_end"] = pd.to_datetime(df["quarter_end"], utc=True).dt.tz_localize(None)

    # Sort
    df = df.sort_values(["quarter_end", "ticker", "cik"]).reset_index(drop=True)

    # Reorder columns
    return df[
        ["quarter_end", "cik", "filer_name", "ticker", "shares", "value_usd", "pct_of_aum"]
    ]


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard schema."""
    return pd.DataFrame(
        columns=[
            "quarter_end",
            "cik",
            "filer_name",
            "ticker",
            "shares",
            "value_usd",
            "pct_of_aum",
        ]
    )
