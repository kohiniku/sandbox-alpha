#!/usr/bin/env python3
"""
FRED macro data ingest script for sandbox-alpha v2.

Phase 2 PR-K: static fixture-based ingest.

This script manages the macro corpus cache at ``/data/macro_corpus/``.
For Phase 2, data is ingested from static CSV fixtures (no network calls).
A future version will integrate ``fredapi`` for live FRED pulls.

Usage
-----
::

    python scripts/ingest_fred.py --series DGS10 DGS2 UNRATE CPIAUCSL DFF \\
                                  --start 2020-01-01 --end 2024-12-31

    # With fredapi (future):
    python scripts/ingest_fred.py --use-fredapi --api-key $FRED_API_KEY \\
                                  --series DGS10 DGS2 UNRATE

Directory layout
----------------
::

    /data/macro_corpus/
      DGS10.csv       # FRED 10-Year Treasury Constant Maturity Rate
      DGS2.csv        # FRED 2-Year Treasury Constant Maturity Rate
      UNRATE.csv      # FRED Unemployment Rate
      CPIAUCSL.csv    # FRED CPI All Urban Consumers
      DFF.csv         # FRED Federal Funds Effective Rate

CSV format (FRED standard)
--------------------------
::

    DATE,VALUE
    2020-01-01,1.88
    2020-02-01,1.50
    ...

Static fixture method (current)
-------------------------------
Place pre-downloaded FRED CSVs directly into ``/data/macro_corpus/``.
Each file must be named ``{series_id}.csv`` and use the FRED ``DATE,VALUE``
column format.

FredAPI method (future)
-----------------------
When ``fredapi`` is available (``pip install fredapi``), this script will
support ``--use-fredapi``::

    from fredapi import Fred

    fred = Fred(api_key=api_key)

    for sid in series:
        df = fred.get_series(sid, observation_start=start,
                             observation_end=end)
        df.to_csv(f"/data/macro_corpus/{sid}.csv")

No network calls are made in the current implementation.
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest FRED macro data into the macro corpus cache."
    )
    parser.add_argument(
        "--series", nargs="+", required=True,
        help="FRED series IDs (e.g. DGS10 DGS2 UNRATE)",
    )
    parser.add_argument(
        "--start", required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--data-dir", default="/data",
        help="Base data directory (default: /data)",
    )
    parser.add_argument(
        "--use-fredapi", action="store_true",
        help="Use fredapi (requires FRED API key)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="FRED API key (only with --use-fredapi)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Check that all series CSVs exist, do not fetch",
    )

    args = parser.parse_args()

    corpus_dir = os.path.join(args.data_dir, "macro_corpus")

    if args.use_fredapi:
        print("--use-fredapi is reserved for future implementation.", file=sys.stderr)
        print("Install fredapi: pip install fredapi", file=sys.stderr)
        print("Future usage:", file=sys.stderr)
        print("  from fredapi import Fred", file=sys.stderr)
        print("  fred = Fred(api_key=os.environ['FRED_API_KEY'])", file=sys.stderr)
        print("  for sid in args.series:", file=sys.stderr)
        print("      df = fred.get_series(sid, ...)", file=sys.stderr)
        print("      df.to_csv(f'{corpus_dir}/{sid}.csv')", file=sys.stderr)
        sys.exit(0)

    if args.validate_only:
        print(f"Validating macro corpus at {corpus_dir} ...")
        missing = []
        for sid in args.series:
            path = os.path.join(corpus_dir, f"{sid}.csv")
            if os.path.isfile(path):
                print(f"  OK: {sid}.csv")
            else:
                print(f"  MISSING: {sid}.csv")
                missing.append(sid)
        if missing:
            print(f"\nMissing series: {missing}")
            print(f"Place CSV files in {corpus_dir}/ with FRED DATE,VALUE format.")
            sys.exit(1)
        else:
            print("\nAll series present.")
        sys.exit(0)

    # Default: inform user this is a static fixture-based ingest
    print(f"Corpus directory: {corpus_dir}")
    print(f"Series requested: {args.series}")
    print(f"Date range: {args.start} to {args.end or 'today'}")
    print()
    print("This script uses static fixture-based ingest.")
    print(f"To populate the corpus, place FRED CSVs directly into {corpus_dir}/")
    print("Expected format: DATE,VALUE")
    print()
    print("For automated FRED API fetch, install fredapi and use --use-fredapi.")


if __name__ == "__main__":
    main()
