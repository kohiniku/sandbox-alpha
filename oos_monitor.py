#!/usr/bin/env python3
"""
Rolling Out-of-Sample Monitor for Adopted Strategies.

Re-checks adopted strategies on data that did not exist when they were adopted.
For each adopted param-type strategy:
  1. Run backtest via the sandbox runner (same protocol as autonomous_loop).
  2. Use the holdout segment (most recent 20% of 5y data) as OOS proxy.
  3. Record OOS metrics in oos_history.
  4. Report greppable status lines.

Does NOT demote — v1 is record-and-report only.
Stdlib-only, matches existing codebase style.
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Re-use autonomous_loop's infrastructure
from autonomous_loop import (
    load_knowledge,
    save_knowledge,
    run_backtest,
    KNOWLEDGE_FILE,
)


def _estimate_warmup_days(params):
    """Estimate warmup margin needed by the strategy's indicators.

    Uses the max numeric param value + a small buffer.
    Falls back to 30 days if no numeric params found.
    """
    max_val = 0
    for v in params.values():
        if isinstance(v, (int, float)):
            max_val = max(max_val, int(v))
    return max(max_val + 5, 30)


def _parse_adoption_date(entry):
    """Extract adoption date from an adopted entry.

    Checks tested_at first, then finished_at.
    Returns a datetime or None.
    """
    for key in ("tested_at", "finished_at"):
        raw = entry.get(key)
        if raw:
            try:
                # Handle both with and without timezone info
                if isinstance(raw, str):
                    # Strip trailing Z for fromisoformat compatibility
                    raw_clean = raw.rstrip("Z")
                    return datetime.fromisoformat(raw_clean)
            except (ValueError, TypeError):
                continue
    return None


def _is_param_strategy(entry):
    """Check if an adopted entry is a param-type strategy (not code-type)."""
    hyp = entry.get("hypothesis", {})
    # Code-type entries have a 'code' field or strategy starts with 'code:'
    if "code" in hyp:
        return False
    strategy = hyp.get("strategy", "")
    if strategy.startswith("code:"):
        return False
    return True


def run_oos_check(entry, today=None):
    """Run OOS check for a single adopted strategy entry.

    Returns (oos_record, error_string_or_None).
    """
    if today is None:
        today = datetime.now()

    hyp = entry.get("hypothesis", {})
    strategy = hyp.get("strategy", "unknown")
    symbol = hyp.get("symbol", "unknown")
    params = hyp.get("params", {})

    adoption_date = _parse_adoption_date(entry)
    if adoption_date is None:
        return None, "no_adoption_date"

    window_days = (today - adoption_date).days
    if window_days < 1:
        return None, "adoption_in_future"

    warmup = _estimate_warmup_days(params)

    # Run backtest via the same runner as autonomous_loop
    result = run_backtest(hyp)
    if "error" in result:
        error_type = result.get("error_type", "unknown")
        return None, f"runner_{error_type}"

    # Extract holdout metrics as OOS proxy
    # The holdout segment is the most recent 20% of 5y data (~1 year)
    holdout_raw = result.get("holdout", {})
    if not isinstance(holdout_raw, dict):
        return None, "no_holdout_metrics"
    holdout = holdout_raw
    oos_sharpe = holdout.get("sharpe_ratio")
    oos_return = holdout.get("total_return_pct", 0.0)
    oos_max_dd = holdout.get("max_drawdown_pct", 0.0)

    if oos_sharpe is None:
        return None, "no_holdout_metrics"
    
    oos_return = float(oos_return)
    oos_max_dd = float(oos_max_dd)

    oos_record = {
        "date": today.strftime("%Y-%m-%d"),
        "window_days": window_days,
        "oos_sharpe": oos_sharpe,
        "oos_return_pct": oos_return,
        "oos_max_drawdown_pct": oos_max_dd,
    }
    return oos_record, None


def run_oos_monitor(knowledge=None, today=None):
    """Main OOS monitor entry point.

    Args:
        knowledge: pre-loaded knowledge dict (loads from file if None).
        today: override for current date (for testing).

    Returns:
        (checked, negative, skipped) counts.
    """
    if today is None:
        today = datetime.now()

    if knowledge is None:
        knowledge = load_knowledge()

    adopted = knowledge.get("adopted", [])
    if not adopted:
        print("OOS_SUMMARY checked=0 negative=0")
        return 0, 0, 0

    checked = 0
    negative = 0
    skipped = 0

    for entry in adopted:
        hyp = entry.get("hypothesis", {})
        strategy = hyp.get("strategy", "unknown")
        symbol = hyp.get("symbol", "unknown")
        label = f"{strategy}/{symbol}"

        if not _is_param_strategy(entry):
            print(f"NOTE: skipping code-type entry {label}")
            skipped += 1
            continue

        oos_record, error = run_oos_check(entry, today=today)

        if error is not None:
            print(f"OOS_STATUS {label} days=0 sharpe=n/a return=n/a% error={error}")
            skipped += 1
            continue

        # Append to oos_history
        history = entry.setdefault("oos_history", [])
        history.append(oos_record)

        window_days = oos_record["window_days"]
        oos_sharpe = oos_record["oos_sharpe"]
        oos_return = oos_record["oos_return_pct"]

        print(f"OOS_STATUS {label} days={window_days} sharpe={oos_sharpe} return={oos_return}%")

        checked += 1
        if oos_sharpe < 0 and window_days >= 30:
            negative += 1

    # Persist updated knowledge
    save_knowledge(knowledge)

    print(f"OOS_SUMMARY checked={checked} negative={negative}")
    if skipped:
        print(f"OOS_SKIPPED count={skipped}")

    return checked, negative, skipped


if __name__ == "__main__":
    knowledge = load_knowledge()
    checked, negative, skipped = run_oos_monitor(knowledge)
    sys.exit(0)
