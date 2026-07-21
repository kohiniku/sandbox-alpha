#!/usr/bin/env python3
"""
Bulk OHLCV fetcher for universe constituents.

PR 4a: One-shot / monthly-incremental script. Hermes-side only.
Primary source: yfinance. Fallback: Stooq (no auth required).
Rate limited: 0.2s delay between symbols, 3-retry exponential backoff.

Usage:
  python scripts/fetch_universe.py \\
    --data-dir /path/to/cache \\
    --universe russell1000 \\
    --as-of 2026-07-21 \\
    --limit 10 \\
    --dry-run
"""

import argparse
import datetime
import logging
import os
import sys
import time
from typing import Optional

import pandas as pd

# -- try/except dual-import pattern for flat-container safety
try:
    from data_adapters.universe import UniverseProvider
except ImportError:
    from universe import UniverseProvider  # type: ignore[no-redef]

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import requests as _requests
except ImportError:
    _requests = None

logger = logging.getLogger(__name__)

# yfinance rate limiting
REQUEST_DELAY = 0.2  # seconds between symbols
MAX_RETRIES = 3

# Full history to fetch on first run (5 years)
DEFAULT_HISTORY_YEARS = 5


def _stooq_url(symbol: str) -> str:
    """Build Stooq daily CSV URL for a US-listed symbol."""
    return f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"


def _fetch_yfinance(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from yfinance. Returns DataFrame or None on failure."""
    if yf is None:
        return None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            if df is None or df.empty:
                return None
            # Normalize columns to Open/High/Low/Close/Volume
            df = df.rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in cols if c in df.columns]]
            return df
        except Exception as exc:
            if attempt < MAX_RETRIES:
                delay = REQUEST_DELAY * (2 ** attempt)
                time.sleep(delay)
            else:
                logger.debug("yfinance failed for %s after %d attempts: %s", symbol, MAX_RETRIES, exc)
                return None
    return None


def _fetch_stooq(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Stooq. Returns DataFrame or None on failure."""
    if _requests is None:
        return None
    try:
        url = _stooq_url(symbol)
        resp = _requests.get(url, timeout=30)
        if resp.status_code != 200 or not resp.text.strip():
            return None
        # Stooq returns: Date,Open,High,Low,Close,Volume (no header row for date)
        # Use StringIO + pandas
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), parse_dates=["Date"], index_col="Date")
        df.index.name = "Date"
        # Rename to canonical
        df = df.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume",
        })
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in cols if c in df.columns]]
        return df
    except Exception as exc:
        logger.debug("Stooq failed for %s: %s", symbol, exc)
        return None


def _read_existing_csv(path: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Read existing CSV and return (DataFrame_or_None, last_date_str_or_None)."""
    if not os.path.isfile(path):
        return None, None
    try:
        df = pd.read_csv(path, index_col=0)
        if df.empty:
            return None, None
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        df.index.name = "Date"
        last_date = df.index.max().strftime("%Y-%m-%d")
        return df, last_date
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data for universe constituents (PR 4a)"
    )
    parser.add_argument("--data-dir", required=True, help="Directory for cached CSVs")
    parser.add_argument("--universe", default="russell1000", help="Universe name")
    parser.add_argument("--as-of", default=None, help="Constituent snapshot date (default: today)")
    parser.add_argument("--limit", type=int, default=None, help="Fetch at most N symbols (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, don't fetch")
    parser.add_argument("--start", default=None, help="Override start date for full fetch")
    argv = parser.parse_args()

    as_of = argv.as_of or datetime.date.today().isoformat()
    end_date = datetime.date.today().isoformat()
    default_start = (datetime.date.today() - datetime.timedelta(days=DEFAULT_HISTORY_YEARS * 365)).isoformat()
    start_date = argv.start or default_start

    os.makedirs(argv.data_dir, exist_ok=True)

    # Load constituents
    provider = UniverseProvider(name=argv.universe)
    symbols = provider.get_symbols(as_of=as_of)
    if argv.limit:
        symbols = symbols[: argv.limit]

    # Dry-run: just print the plan
    if argv.dry_run:
        print(f"Universe: {argv.universe}")
        print(f"As-of:   {as_of}")
        print(f"Symbols: {len(symbols)}")
        print(f"Data dir: {argv.data_dir}")
        print(f"Date range: {start_date} → {end_date}")
        print("\nPlan:")
        for i, sym in enumerate(symbols, 1):
            csv_path = os.path.join(argv.data_dir, f"{sym}.csv")
            existing, last_date = _read_existing_csv(csv_path)
            if existing is not None and last_date:
                # Incremental: from day after last_date to end
                inc_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if inc_start > end_date:
                    print(f"  [{i}/{len(symbols)}] {sym} → up-to-date (last: {last_date})")
                else:
                    print(f"  [{i}/{len(symbols)}] {sym} → incremental {inc_start} → {end_date}")
            else:
                print(f"  [{i}/{len(symbols)}] {sym} → full fetch {start_date} → {end_date}")
        return

    # Fetch
    fetched = 0
    cached = 0
    failed = 0
    new_rows_total = 0

    for i, sym in enumerate(symbols, 1):
        csv_path = os.path.join(argv.data_dir, f"{sym}.csv")
        existing_df, last_date = _read_existing_csv(csv_path)
        source = "cached"

        if existing_df is not None and last_date:
            # Incremental: fetch only new days
            inc_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if inc_start > end_date:
                print(f"[{i}/{len(symbols)}] {sym} status=up_to_date rows={len(existing_df)} source=cached", file=sys.stderr)
                cached += 1
                continue
            # Fetch new data
            df = _fetch_yfinance(sym, inc_start, end_date)
            source = "yfinance"
            if df is None:
                df = _fetch_stooq(sym)
                source = "stooq"
            if df is not None and not df.empty:
                # Filter to only dates after last_date
                df = df.loc[inc_start:]
                if not df.empty:
                    df.to_csv(csv_path, mode="a", header=False)
                    new_rows = len(df)
                    existing_df = pd.concat([existing_df, df])
                    existing_df = existing_df[~existing_df.index.duplicated(keep="last")]
                    existing_df = existing_df.sort_index()
                    new_rows_total += new_rows
                    print(f"[{i}/{len(symbols)}] {sym} status=appended rows=+{new_rows} total={len(existing_df)} source={source}", file=sys.stderr)
                    fetched += 1
                else:
                    print(f"[{i}/{len(symbols)}] {sym} status=no_new_data rows={len(existing_df)} source={source}", file=sys.stderr)
                    cached += 1
            else:
                print(f"[{i}/{len(symbols)}] {sym} status=failed rows=0 source=failed", file=sys.stderr)
                failed += 1
        else:
            # Full fetch
            df = _fetch_yfinance(sym, start_date, end_date)
            source = "yfinance"
            if df is None:
                df = _fetch_stooq(sym)
                source = "stooq"
            if df is not None and not df.empty:
                # Ensure canonical column order
                cols = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in cols if c in df.columns]]
                df.to_csv(csv_path)
                new_rows = len(df)
                new_rows_total += new_rows
                print(f"[{i}/{len(symbols)}] {sym} status=fetched rows={new_rows} source={source}", file=sys.stderr)
                fetched += 1
            else:
                print(f"[{i}/{len(symbols)}] {sym} status=failed rows=0 source=failed", file=sys.stderr)
                failed += 1

        # Rate limit
        time.sleep(REQUEST_DELAY)

    # Final summary (stdout only)
    print(f"SUMMARY total={len(symbols)} fetched={fetched} cached={cached} failed={failed} new_rows_total={new_rows_total}")


if __name__ == "__main__":
    main()
