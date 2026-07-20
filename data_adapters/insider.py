#!/usr/bin/env python3
"""
Insider trades data adapter for sandbox-alpha v2 (Phase 2 PR-J).

Reads a pre-fetched SEC Form 4 insider trading corpus from JSONL files
on disk and returns a pandas DataFrame with transaction-level records.

No network calls at runtime. No external SDK.
"""

import json
import logging
import os
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_insider_trades(
    universe: List[str],
    start: str,
    end: Optional[str],
    min_transaction_usd: float = 10000.0,
    roles: Optional[List[str]] = None,
    data_dir: str = "",
) -> pd.DataFrame:
    """Load insider trading data from a pre-fetched JSONL corpus.

    Parameters
    ----------
    universe : list[str]
        Ticker symbols to filter on. Empty list = no ticker filter (all rows).
    start : str
        Inclusive start date, ISO format "YYYY-MM-DD".
    end : str or None
        Inclusive end date "YYYY-MM-DD". None = through last row.
    min_transaction_usd : float
        Minimum transaction value in USD. Rows below this threshold are
        silently dropped. Default 10000.
    roles : list[str] or None
        Insider roles to include (e.g. ["CEO", "CFO"]). None = all roles.
    data_dir : str
        Root data directory. The corpus lives at
        ``{data_dir}/insider_corpus/*.jsonl``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[transaction_date, ticker, insider_name, role,
        transaction_type, shares, price, value_usd]``.
        Empty DataFrame if no corpus files are found (graceful degradation).
    """
    corpus_dir = os.path.join(data_dir, "insider_corpus")
    if not os.path.isdir(corpus_dir):
        logger.warning(
            "Insider corpus directory not found: %s. "
            "Returning empty DataFrame (run scripts/ingest_insider.py first).",
            corpus_dir,
        )
        return _empty_df()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    universe_set: Optional[set] = set(universe) if universe else None
    roles_set: Optional[set] = set(roles) if roles else None

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
                    logger.warning("Skipping malformed JSON line in %s", filepath)
                    continue

                # Filter by transaction value
                value = float(rec.get("value_usd", 0))
                if value < min_transaction_usd:
                    continue

                # Filter by date
                txn_date = rec.get("transaction_date", "")
                if not txn_date:
                    continue
                try:
                    txn_ts = pd.Timestamp(txn_date)
                except (ValueError, TypeError):
                    continue
                if txn_ts < start_ts:
                    continue
                if end_ts is not None and txn_ts > end_ts:
                    continue

                # Filter by ticker
                ticker = rec.get("ticker", "")
                if universe_set is not None and ticker not in universe_set:
                    continue

                # Filter by role
                role = rec.get("role", "")
                if roles_set is not None and role not in roles_set:
                    continue

                rows.append({
                    "transaction_date": txn_ts,
                    "ticker": ticker,
                    "insider_name": rec.get("insider_name", ""),
                    "role": role,
                    "transaction_type": rec.get("transaction_type", ""),
                    "shares": float(rec.get("shares", 0)),
                    "price": float(rec.get("price", 0)),
                    "value_usd": value,
                })

    if not rows:
        return _empty_df()

    df = pd.DataFrame(rows)

    # Ensure date is DatetimeIndex-compatible UTC naive
    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"], utc=True
    ).dt.tz_localize(None)

    # Sort
    df = df.sort_values(["transaction_date", "ticker"]).reset_index(drop=True)

    # Reorder columns
    return df[
        [
            "transaction_date",
            "ticker",
            "insider_name",
            "role",
            "transaction_type",
            "shares",
            "price",
            "value_usd",
        ]
    ]


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard schema."""
    return pd.DataFrame(
        columns=[
            "transaction_date",
            "ticker",
            "insider_name",
            "role",
            "transaction_type",
            "shares",
            "price",
            "value_usd",
        ]
    )
