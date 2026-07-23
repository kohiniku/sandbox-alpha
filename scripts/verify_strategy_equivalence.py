#!/usr/bin/env python3
"""Behavior-equivalence harness for backtest_engine strategy refactors.

Runs the engine CLI over a deterministic strategy/symbol/mode matrix on
synthetic data and writes normalized (sorted-key) JSON outputs. To prove a
refactor changes nothing:

    git checkout main
    python scripts/verify_strategy_equivalence.py --out /tmp/golden_before
    git checkout <refactor-branch>
    python scripts/verify_strategy_equivalence.py --out /tmp/golden_after
    diff -r /tmp/golden_before /tmp/golden_after   # must be empty

Synthetic data is seeded, so outputs are reproducible across runs and hosts.
The engine is invoked as a subprocess in script mode, exercising the same
import path the container uses.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
ENGINE = REPO / "backtests" / "backtest_engine.py"

# Keys introduced after the baseline that should be stripped from normalized
# output so equivalence diffs don't flag additive-only schema changes.
STRIP_KEYS = {"turnover"}

DATA_SPECS = {"TRND": (7, 0.0008, 0.012), "CHOP": (13, 0.0000, 0.020)}

MATRIX = [
    ("sma_crossover", {}, []),
    ("sma_crossover", {"fast_window": 5, "slow_window": 50}, []),
    ("sma_crossover", {}, ["--no-walkforward", "--metrics-since", "2024-01-01"]),
    ("mean_reversion", {}, []),
    ("mean_reversion", {"window": 10, "threshold": 1.5}, []),
    ("mean_reversion", {}, ["--no-walkforward", "--metrics-since", "2024-01-01"]),
    ("momentum", {}, []),
    ("momentum", {"lookback": 60, "hold_period": 10}, []),
    ("momentum", {}, ["--no-walkforward", "--metrics-since", "2024-01-01"]),
    ("rsi", {}, []),
    ("rsi", {"rsi_window": 7, "oversold": 25, "overbought": 75}, []),
    ("rsi", {}, ["--no-walkforward", "--metrics-since", "2024-01-01"]),
]


def generate_data(data_dir: pathlib.Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for symbol, (seed, mu, sigma) in DATA_SPECS.items():
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-02", periods=1300, freq="B", tz="UTC")
        ret = rng.normal(mu, sigma, size=len(dates))
        close = 100.0 * np.cumprod(1.0 + ret)
        df = pd.DataFrame(
            {
                "Open": close * (1 - 0.0005),
                "High": close * (1 + 0.006),
                "Low": close * (1 - 0.006),
                "Close": close,
                "Volume": np.full(len(dates), 1_000_000),
            },
            index=dates,
        )
        df.index.name = "Date"
        df.to_csv(data_dir / f"{symbol}.csv")


def _strip_keys(obj, keys):
    """Recursively strip keys from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_keys(v, keys) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Directory for normalized JSON outputs")
    parser.add_argument(
        "--data-dir", default=None,
        help="Synthetic data dir (default: <out>/../equivalence_data, generated if missing)",
    )
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else out.parent / "equivalence_data"
    if not all((data_dir / f"{s}.csv").is_file() for s in DATA_SPECS):
        generate_data(data_dir)

    failures = 0
    for i, (strategy, params, extra) in enumerate(MATRIX):
        for symbol in DATA_SPECS:
            cmd = [sys.executable, str(ENGINE), "--strategy", strategy, "--symbol", symbol,
                   "--params", json.dumps(params, sort_keys=True),
                   "--data-dir", str(data_dir)] + extra
            proc = subprocess.run(cmd, capture_output=True, text=True)
            name = f"{i:02d}_{strategy}_{symbol}.json"
            if proc.returncode != 0:
                print(f"FAIL {name}: {proc.stderr[-300:]}")
                failures += 1
                continue
            normalized = json.dumps(
                _strip_keys(json.loads(proc.stdout), STRIP_KEYS),
                indent=1, sort_keys=True)
            (out / name).write_text(normalized + "\n")
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
