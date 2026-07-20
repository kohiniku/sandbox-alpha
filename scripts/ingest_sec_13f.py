#!/usr/bin/env python3
"""
Offline SEC 13F corpus ingester for sandbox-alpha v2 (Phase 2 PR-I).

Reads holdings data from a static JSON fixture and writes it into
/data/sec_13f_corpus/*.jsonl for the data adapter to consume.

NO NETWORK — this is a local-only ingester for offline backtesting.
The fixture format is documented inline.

Plugging in live SEC EDGAR (future work)
----------------------------------------
Replace ``_load_fixture()`` with a function that:
  1. Queries the SEC EDGAR full-text search API for 13F-HR filings:
     https://efts.sec.gov/LATEST/search-index?q=...
  2. Downloads the primary_doc.xml for each filing.
  3. Parses the XML ``<informationTable>`` to extract issuer/ticker,
     shares, and value.
  4. Normalises CIKs to 10-digit zero-padded strings.
  5. Writes each quarter's records as one JSONL line.

The fixture schema already matches the expected adapter input, so the
live ingester just needs to produce the same JSON shape.
"""

import json
import os
import sys
from datetime import date

# ---------------------------------------------------------------------------
# Static fixture: sample 13F holdings data
# ---------------------------------------------------------------------------
# This fixture provides synthetic but realistic holdings for testing.
# In production, replace with live EDGAR ingestion (see docstring above).
# fmt: off

FIXTURE = [
    {
        "quarter_end": "2024-12-31",
        "cik": "0001067983",
        "filer_name": "Berkshire Hathaway Inc",
        "ticker": "AAPL",
        "shares": 300000000,
        "value_usd": 45000000000.0,
        "pct_of_aum": 12.5,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0001067983",
        "filer_name": "Berkshire Hathaway Inc",
        "ticker": "BAC",
        "shares": 800000000,
        "value_usd": 28000000000.0,
        "pct_of_aum": 7.8,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0001067983",
        "filer_name": "Berkshire Hathaway Inc",
        "ticker": "AXP",
        "shares": 150000000,
        "value_usd": 25000000000.0,
        "pct_of_aum": 6.9,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0001341439",
        "filer_name": "Vanguard Group Inc",
        "ticker": "AAPL",
        "shares": 1200000000,
        "value_usd": 180000000000.0,
        "pct_of_aum": 3.2,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0001341439",
        "filer_name": "Vanguard Group Inc",
        "ticker": "MSFT",
        "shares": 900000000,
        "value_usd": 150000000000.0,
        "pct_of_aum": 2.7,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0001341439",
        "filer_name": "Vanguard Group Inc",
        "ticker": "NVDA",
        "shares": 500000000,
        "value_usd": 70000000000.0,
        "pct_of_aum": 1.25,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0001341439",
        "filer_name": "Vanguard Group Inc",
        "ticker": "AMZN",
        "shares": 400000000,
        "value_usd": 60000000000.0,
        "pct_of_aum": 1.07,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0000036405",
        "filer_name": "BlackRock Inc",
        "ticker": "AAPL",
        "shares": 900000000,
        "value_usd": 135000000000.0,
        "pct_of_aum": 1.8,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0000036405",
        "filer_name": "BlackRock Inc",
        "ticker": "MSFT",
        "shares": 700000000,
        "value_usd": 117000000000.0,
        "pct_of_aum": 1.56,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0000036405",
        "filer_name": "BlackRock Inc",
        "ticker": "GOOGL",
        "shares": 350000000,
        "value_usd": 45000000000.0,
        "pct_of_aum": 0.6,
    },
    {
        "quarter_end": "2024-12-31",
        "cik": "0000036405",
        "filer_name": "BlackRock Inc",
        "ticker": "TSLA",
        "shares": 100000000,
        "value_usd": 25000000000.0,
        "pct_of_aum": 0.33,
    },
    {
        "quarter_end": "2025-03-31",
        "cik": "0001067983",
        "filer_name": "Berkshire Hathaway Inc",
        "ticker": "AAPL",
        "shares": 280000000,
        "value_usd": 42000000000.0,
        "pct_of_aum": 10.8,
    },
    {
        "quarter_end": "2025-03-31",
        "cik": "0001067983",
        "filer_name": "Berkshire Hathaway Inc",
        "ticker": "BAC",
        "shares": 780000000,
        "value_usd": 27000000000.0,
        "pct_of_aum": 7.0,
    },
    {
        "quarter_end": "2025-03-31",
        "cik": "0001341439",
        "filer_name": "Vanguard Group Inc",
        "ticker": "AAPL",
        "shares": 1220000000,
        "value_usd": 185000000000.0,
        "pct_of_aum": 3.3,
    },
    {
        "quarter_end": "2025-03-31",
        "cik": "0001341439",
        "filer_name": "Vanguard Group Inc",
        "ticker": "MSFT",
        "shares": 910000000,
        "value_usd": 155000000000.0,
        "pct_of_aum": 2.75,
    },
    {
        "quarter_end": "2025-03-31",
        "cik": "0000036405",
        "filer_name": "BlackRock Inc",
        "ticker": "AAPL",
        "shares": 880000000,
        "value_usd": 132000000000.0,
        "pct_of_aum": 1.75,
    },
]

# fmt: on


def _load_fixture():
    """Return the static fixture records.

    Replace this with live SEC EDGAR ingestion for production use.
    See module docstring for the plug-in approach.
    """
    return list(FIXTURE)


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/sec_13f_corpus"
    os.makedirs(output_dir, exist_ok=True)

    records = _load_fixture()

    # Group by quarter_end to produce one JSONL file per quarter
    by_quarter: dict = {}
    for rec in records:
        qe = rec["quarter_end"]
        by_quarter.setdefault(qe, []).append(rec)

    total_written = 0
    for qe, recs in sorted(by_quarter.items()):
        fname = f"{qe}.jsonl"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
                total_written += 1
        print(f"Wrote {len(recs)} records to {fpath}")

    print(f"\nDone. {total_written} total records written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
