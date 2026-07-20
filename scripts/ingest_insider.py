#!/usr/bin/env python3
"""
Ingest static SEC Form 4 insider trading fixtures into JSONL corpus.

Generates synthetic insider trading records for a fixed set of tickers and
dates, then writes them as per-quarter JSONL files under
/data/insider_corpus/.

Usage
-----
    python scripts/ingest_insider.py --output-dir /data/insider_corpus

No live SEC EDGAR fetch in this PR (Phase 2 PR-J). This is a static
fixture-based ingest for development and testing.

Schema (per JSONL line)
------------------------
    {
        "transaction_date": "YYYY-MM-DD",
        "ticker": "AAPL",
        "insider_name": "Insider Name",
        "role": "CEO|CFO|COO|President|Director|10%_owner|Other",
        "transaction_type": "Purchase|Sale",
        "shares": int,
        "price": float,
        "value_usd": float
    }

Idempotent: re-running overwrites the output directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Static fixture data
# ---------------------------------------------------------------------------

# Each entry: (date, ticker, name, role, type, shares, price)
_FIXTURES = [
    # Q1 2025 — AAPL insiders
    ("2025-01-15", "AAPL", "Tim Cook", "CEO", "Sale", 10000, 175.50),
    ("2025-01-20", "AAPL", "Luca Maestri", "CFO", "Sale", 5000, 173.00),
    ("2025-02-10", "AAPL", "Jeff Williams", "COO", "Sale", 8000, 180.25),
    ("2025-02-28", "AAPL", "Arthur Levinson", "Director", "Sale", 3000, 182.00),
    ("2025-03-05", "AAPL", "Katherine Adams", "Other", "Purchase", 1200, 170.00),
    # Q2 2025 — MSFT insiders
    ("2025-04-01", "MSFT", "Satya Nadella", "CEO", "Sale", 15000, 420.00),
    ("2025-04-15", "MSFT", "Amy Hood", "CFO", "Sale", 6000, 415.50),
    ("2025-05-10", "MSFT", "Brad Smith", "President", "Sale", 10000, 430.00),
    ("2025-05-20", "MSFT", "Reid Hoffman", "Director", "Purchase", 2000, 425.00),
    ("2025-06-01", "MSFT", "John Thompson", "Other", "Purchase", 500, 418.00),
    # Q1 2025 — GOOG insiders
    ("2025-01-10", "GOOG", "Sundar Pichai", "CEO", "Sale", 12000, 145.00),
    ("2025-02-14", "GOOG", "Ruth Porat", "CFO", "Sale", 4000, 142.00),
    ("2025-03-01", "GOOG", "Philipp Schindler", "Other", "Sale", 7000, 148.00),
    # Q2 2025 — AMZN insiders
    ("2025-04-20", "AMZN", "Andy Jassy", "CEO", "Sale", 20000, 185.00),
    ("2025-05-05", "AMZN", "Brian Olsavsky", "CFO", "Sale", 8000, 190.00),
    ("2025-06-10", "AMZN", "Jeff Bezos", "10%_owner", "Sale", 50000, 188.50),
    # Q1 2025 — NVDA insiders
    ("2025-02-01", "NVDA", "Jensen Huang", "CEO", "Sale", 5000, 950.00),
    ("2025-03-15", "NVDA", "Colette Kress", "CFO", "Sale", 2000, 940.00),
    # Q2 2025 — META insiders
    ("2025-04-10", "META", "Mark Zuckerberg", "CEO", "Sale", 25000, 510.00),
    ("2025-05-20", "META", "Susan Li", "CFO", "Sale", 3000, 505.00),
    ("2025-06-15", "META", "Javier Olivan", "COO", "Sale", 5000, 520.00),
    # Q1 2025 — TSLA insiders
    ("2025-01-05", "TSLA", "Elon Musk", "CEO", "Purchase", 30000, 250.00),
    ("2025-02-20", "TSLA", "Zachary Kirkhorn", "CFO", "Purchase", 2000, 245.00),
    ("2025-03-10", "TSLA", "Robyn Denholm", "Director", "Purchase", 1000, 248.00),
    # Edge cases
    ("2025-03-31", "BRK.B", "Warren Buffett", "CEO", "Purchase", 100000, 400.00),
    ("2025-06-30", "JPM", "Jamie Dimon", "CEO", "Purchase", 5000, 200.00),
    ("2025-01-01", "DIS", "Bob Iger", "CEO", "Sale", 15000, 95.00),
]


# ---------------------------------------------------------------------------
# Main ingest pipeline
# ---------------------------------------------------------------------------


def ingest(output_dir: str) -> dict:
    """Generate static fixtures and write per-quarter JSONL files.

    Returns stats dict.
    """
    output_path = Path(output_dir)
    os.makedirs(str(output_path), exist_ok=True)

    # Group by quarter (YYYY-QN)
    by_quarter: dict[str, list] = {}
    for date_str, ticker, name, role, txn_type, shares, price in _FIXTURES:
        year = date_str[:4]
        month = int(date_str[5:7])
        quarter = f"{year}-Q{(month - 1) // 3 + 1}"

        value_usd = round(float(shares) * price, 2)
        record = {
            "transaction_date": date_str,
            "ticker": ticker,
            "insider_name": name,
            "role": role,
            "transaction_type": txn_type,
            "shares": shares,
            "price": price,
            "value_usd": value_usd,
        }
        by_quarter.setdefault(quarter, []).append(record)

    # Write per-quarter JSONL
    written = 0
    for quarter, records in sorted(by_quarter.items()):
        out_file = output_path / f"{quarter}.jsonl"
        with open(out_file, "w", encoding="utf-8") as fh:
            for rec in records:
                print(json.dumps(rec, ensure_ascii=False), file=fh)
                written += 1

    print(f"Ingested {written} trades into {len(by_quarter)} quarter files")
    for quarter in sorted(by_quarter.keys()):
        print(f"  {quarter}: {len(by_quarter[quarter])} records")

    return {
        "trades_ingested": written,
        "quarter_files": len(by_quarter),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest static SEC Form 4 insider trading fixtures -> JSONL corpus"
    )
    parser.add_argument(
        "--output-dir",
        default="/data/insider_corpus",
        help="Output directory for per-quarter .jsonl files",
    )
    args = parser.parse_args()

    stats = ingest(output_dir=args.output_dir)
    print(f"\nDone. Stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
