#!/usr/bin/env python3
"""
Characterize gate-v2 against every near_miss in knowledge.json.

Re-runs each near-miss through the runner with cv_folds=3, embargo_days=21,
computes the v2 CV verdict, and reports v1-v2 reversals.

Read-only — does NOT modify knowledge.json or trigger adoption.
"""
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add repo root to path so we can import autonomous_loop modules
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from autonomous_loop import (
    _eval_val_gate_cv,
    _eval_holdout_gate_cv,
    MISSING_METRIC,
)
from loop_constants import CV_FOLDS, EMBARGO_DAYS


def load_knowledge(path):
    with open(path) as f:
        return json.load(f)


def call_runner(runner_url, strategy, symbol, params):
    """Call the sandbox runner /run with cv_folds and embargo_days."""
    url = f"{runner_url.rstrip('/')}/run"
    payload = {
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "cv_folds": CV_FOLDS,
        "embargo_days": EMBARGO_DAYS,
    }
    body = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except Exception as e:
        print(f"  ❌ Runner error: {e}", file=sys.stderr)
        return {"error": str(e)}


def compute_v2_verdict(result, n_family=1):
    """Extract cv block and compute gate v2 verdict."""
    cv = result.get("cv")
    if not cv or "folds" not in cv:
        return None, "no cv block"

    try:
        per_fold_returns = [
            pd.Series(f["val_daily_returns"], index=pd.to_datetime(f["val_dates"]))
            for f in cv["folds"]
        ]
        holdout_returns = pd.Series(
            cv["holdout"]["daily_returns"],
            index=pd.to_datetime(cv["holdout"]["dates"])
        )

        val_passed, lcb_val, val_reasons = _eval_val_gate_cv(per_fold_returns, n_family)

        if val_passed:
            ho_passed, lcb_ho, ho_reasons = _eval_holdout_gate_cv(holdout_returns, lcb_val)
            passed = ho_passed
        else:
            passed = False
            lcb_ho = None
            ho_passed = None
            ho_reasons = None

        verdict = "adopted" if passed else "rejected"
        return {
            "verdict": verdict,
            "val_passed": val_passed,
            "lcb_sharpe": round(lcb_val, 4),
            "holdout_passed": ho_passed,
            "holdout_lcb_sharpe": round(lcb_ho, 4) if lcb_ho is not None else None,
        }, None

    except Exception as e:
        return None, f"error: {e}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Characterize gate-v2 against near_misses")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max near_misses to evaluate (default: all)")
    parser.add_argument("--runner-url", default=None,
                        help="Sandbox runner URL (default: $SANDBOX_RUNNER_URL)")
    parser.add_argument("--knowledge", default=str(REPO_ROOT / "knowledge.json"),
                        help="Path to knowledge.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic data instead of calling runner")
    args = parser.parse_args()

    runner_url = args.runner_url or os.environ.get("SANDBOX_RUNNER_URL")
    if not runner_url and not args.dry_run:
        print("❌ SANDBOX_RUNNER_URL not set and --runner-url not provided", file=sys.stderr)
        sys.exit(1)

    knowledge = load_knowledge(args.knowledge)
    near_misses = knowledge.get("near_misses", [])

    if not near_misses:
        print("📭 No near_misses found in knowledge.json")
        return

    limit = args.limit if args.limit else len(near_misses)
    near_misses = near_misses[-limit:]  # most recent

    print(f"🔍 Characterizing gate-v2 against {len(near_misses)} near_misses...")
    print()

    # Table header
    print("| strategy/symbol | v1 verdict | v1 val_sharpe | v2 verdict | v2 lcb_sharpe | reversal |")
    print("|---|---|---|---|---|---|")

    results = []
    for i, nm in enumerate(near_misses):
        strategy = nm["strategy"]
        symbol = nm["symbol"]
        params = nm["params"]
        v1_val_sharpe = nm["val_sharpe"]
        tag = f"{strategy}/{symbol}"

        if args.dry_run:
            # Synthetic: construct a plausible v2 result
            import numpy as np
            rng = np.random.default_rng(i + 1)
            # Generate mock cv folds with similar Sharpe to v1
            mu = v1_val_sharpe / (252 ** 0.5) * 0.01
            v2_lcb = v1_val_sharpe * (0.7 + rng.random() * 0.6)  # 0.7-1.3x v1
            v2_passed = v2_lcb >= 0.5
            v2_verdict = "adopted" if v2_passed else "rejected"
        else:
            result = call_runner(runner_url, strategy, symbol, params)
            if "error" in result:
                print(f"| {tag} | rejected | {v1_val_sharpe:.2f} | error | — | — |")
                results.append({"tag": tag, "v1": "rejected", "v2": "error",
                                "v1_val": v1_val_sharpe, "v2_lcb": None, "reversal": False})
                continue

            v2_data, err = compute_v2_verdict(result)
            if err:
                print(f"| {tag} | rejected | {v1_val_sharpe:.2f} | error | — | — |")
                results.append({"tag": tag, "v1": "rejected", "v2": "error",
                                "v1_val": v1_val_sharpe, "v2_lcb": None, "reversal": False})
                continue

            v2_verdict = v2_data["verdict"]
            v2_lcb = v2_data["lcb_sharpe"]

        reversal = v2_verdict == "adopted"  # near_misses are all v1=rejected
        reversal_str = "⬆️ REVERSAL" if reversal else "—"

        v2_lcb_str = f"{v2_lcb:.2f}" if v2_lcb is not None else "—"

        print(f"| {tag} | rejected | {v1_val_sharpe:.2f} | {v2_verdict} | {v2_lcb_str} | {reversal_str} |")
        results.append({
            "tag": tag,
            "v1": "rejected",
            "v2": v2_verdict,
            "v1_val": v1_val_sharpe,
            "v2_lcb": v2_lcb,
            "reversal": reversal,
        })

    # Summary
    total = len(results)
    adopted_reversals = sum(1 for r in results if r["v2"] == "adopted")
    rejected_reversals = sum(1 for r in results if r["v2"] == "rejected" and r["v1"] == "adopted")  # 常に0
    errors = sum(1 for r in results if r["v2"] == "error")
    unchanged = sum(1 for r in results if r["v2"] == "rejected")
    # All near_misses are v1=rejected, so "adopted" entries in v2 are reversals
    # and "rejected" entries are unchanged

    print()
    print(f"SUMMARY total={total} adopted_reversals={adopted_reversals} rejected_reversals={rejected_reversals} unchanged={unchanged} errors={errors}")


if __name__ == "__main__":
    main()
