#!/usr/bin/env python3
"""
Strategy Review — Diagnosis Stage (PR-B2).

Picks recently-failed strategy families, re-measures them via the
sandbox runner, computes machine-readable diagnosis flags, and
persists reports.  NO LLM calls, NO verdicts/refine/kill actions.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from loop_constants import FamilyLifecycle

BASE_DIR = Path(__file__).resolve().parent
REVIEW_REPORTS_DIR = BASE_DIR / "review_reports"
KNOWLEDGE_FILE = BASE_DIR / "knowledge.json"

# Import autonomous_loop helpers (import-time mkdir side effect is acceptable,
# same pattern as the loop itself).
from autonomous_loop import _family_key, _derive_family_type, load_knowledge, save_knowledge

REVIEW_REPORTS_DIR.mkdir(exist_ok=True)

# --- Flag thresholds (module constants) ---
COST_BOUND_DELTA = 0.3
NO_SIGNAL_THRESHOLD = 0.2
UNSTABLE_SPREAD = 1.5
REGIME_POSITIVE = 0.5
REGIME_NEGATIVE = -0.5
HIGH_TURNOVER_THRESHOLD = 50.0

# Default max families (env-overridable, read at call time)
DEFAULT_REVIEW_MAX_FAMILIES = 3


def _read_max_families():
    """Read REVIEW_MAX_FAMILIES from env, default to DEFAULT_REVIEW_MAX_FAMILIES."""
    val = os.environ.get("REVIEW_MAX_FAMILIES")
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return DEFAULT_REVIEW_MAX_FAMILIES


# ============================================================================
# HTTP helper (stdlib-only, same pattern as autonomous_loop.py)
# ============================================================================

def _post_json(url, payload, timeout=180):
    """POST JSON payload to url. Returns parsed JSON dict. Raises on error."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response_body = resp.read().decode("utf-8")
        status = resp.status
    if status != 200:
        raise RuntimeError(f"Runner HTTP {status}: {response_body[:500]}")
    return json.loads(response_body)


# ============================================================================
# Candidate selection
# ============================================================================

def _get_entry_timestamp(entry):
    """Get the timestamp from a rejected or near_miss entry."""
    # rejected entries have 'tested_at'; near_miss entries have 'date'
    return entry.get("tested_at") or entry.get("date", "")


def _family_has_new_evidence(family_key, knowledge):
    """Check if family has rejected/near-miss entries after last_review_at."""
    review_state = knowledge.get("review_state", {})
    last_review = review_state.get("last_review_at", "1970-01-01T00:00:00")

    # Check last_diagnosed_at for this specific family
    reviewed = review_state.get("reviewed", {})
    family_review = reviewed.get(family_key, {})
    last_diagnosed = family_review.get("last_diagnosed_at", "1970-01-01T00:00:00")

    # Collect all evidence timestamps for this family
    timestamps = []

    family = knowledge.get("families", {}).get(family_key, {})
    family_type = family.get("family_type", "single")

    for entry in knowledge.get("rejected", []):
        hyp = entry.get("hypothesis", {})
        strategy = hyp.get("strategy", "")
        symbol = hyp.get("symbol", "")
        ft = _derive_family_type(hyp)
        key = _family_key(strategy, symbol, ft)
        if key == family_key:
            ts = _get_entry_timestamp(entry)
            if ts:
                timestamps.append(ts)

    near_list = "near_misses_cross" if family_type == "cross" else "near_misses"
    for entry in knowledge.get(near_list, []):
        strategy = entry.get("strategy", "")
        symbol = entry.get("symbol", "")
        ft = _derive_family_type(entry)
        key = _family_key(strategy, symbol, ft)
        if key == family_key:
            ts = _get_entry_timestamp(entry)
            if ts:
                timestamps.append(ts)

    if not timestamps:
        return False

    newest_failure = max(timestamps)
    return newest_failure > last_review and newest_failure > last_diagnosed


def select_candidates(knowledge, now, max_families=None):
    """
    Select up to max_families family keys eligible for diagnosis.

    Eligible: lifecycle not KILLED, has new evidence since last review.
    Priority: families appearing in near-misses first, then by best_val_sharpe desc.
    """
    if max_families is None:
        max_families = _read_max_families()

    families = knowledge.get("families", {})
    near_set = set()
    candidate_set = set()

    # Collect families from near_misses and near_misses_cross
    for list_name in ("near_misses", "near_misses_cross"):
        for entry in knowledge.get(list_name, []):
            strategy = entry.get("strategy", "")
            symbol = entry.get("symbol", "")
            ft = _derive_family_type(entry)
            key = _family_key(strategy, symbol, ft)
            if key in families:
                fam = families[key]
                if fam.get("lifecycle") != FamilyLifecycle.KILLED:
                    near_set.add(key)

    # Collect families from rejected
    for entry in knowledge.get("rejected", []):
        hyp = entry.get("hypothesis", {})
        strategy = hyp.get("strategy", "")
        symbol = hyp.get("symbol", "")
        ft = _derive_family_type(hyp)
        key = _family_key(strategy, symbol, ft)
        if key in families:
            fam = families[key]
            if fam.get("lifecycle") != FamilyLifecycle.KILLED:
                candidate_set.add(key)

    # Near-miss families first (sorted by best_val_sharpe desc)
    near_eligible = []
    for key in near_set:
        if _family_has_new_evidence(key, knowledge):
            fam = families[key]
            near_eligible.append((fam.get("best_val_sharpe", -999), key))
    near_eligible.sort(key=lambda x: (-x[0], x[1]))

    # Then rejected-only families
    rejected_eligible = []
    for key in candidate_set - near_set:
        if _family_has_new_evidence(key, knowledge):
            fam = families[key]
            rejected_eligible.append((fam.get("best_val_sharpe", -999), key))
    rejected_eligible.sort(key=lambda x: (-x[0], x[1]))

    candidates = [key for _, key in near_eligible] + [key for _, key in rejected_eligible]
    return candidates[:max_families]


# ============================================================================
# Evidence recovery
# ============================================================================

def _find_most_recent_evidence(family_key, knowledge):
    """
    Find the most recent rejected or near-miss entry for a family.

    Returns the entry dict (for near_miss/near_misses_cross) or
    the rejected entry. Returns None if no evidence found.
    """
    family = knowledge.get("families", {}).get(family_key, {})
    family_type = family.get("family_type", "single")
    # Also derive from key: manifest: prefix → cross
    if family_key.startswith("manifest:"):
        family_type = "cross"

    best_ts = ""
    best_entry = None

    # Check near_misses first
    near_list = "near_misses_cross" if family_type == "cross" else "near_misses"
    for entry in knowledge.get(near_list, []):
        strategy = entry.get("strategy", "")
        symbol = entry.get("symbol", "")
        ft = _derive_family_type(entry)
        key = _family_key(strategy, symbol, ft)
        if key == family_key:
            ts = _get_entry_timestamp(entry)
            if ts > best_ts:
                best_ts = ts
                best_entry = entry

    # Check rejected
    for entry in knowledge.get("rejected", []):
        hyp = entry.get("hypothesis", {})
        strategy = hyp.get("strategy", "")
        symbol = hyp.get("symbol", "")
        ft = _derive_family_type(hyp)
        key = _family_key(strategy, symbol, ft)
        if key == family_key:
            ts = _get_entry_timestamp(entry)
            if ts > best_ts:
                best_ts = ts
                best_entry = entry

    return best_entry


# ============================================================================
# Runner response parsing
# ============================================================================

def _parse_baseline_response(response):
    """Extract val sharpe, turnover, and fold sharpes from runner response."""
    oos = response.get("out_of_sample", {})
    val_sharpe = oos.get("sharpe_ratio", 0.0)
    val_turnover = oos.get("turnover", 0.0)

    fold_sharpes = None
    cv_block = response.get("cv", {})
    folds = cv_block.get("folds", [])
    if folds:
        fold_sharpes = []
        for fold in folds:
            val = fold.get("val", {})
            fold_sharpes.append(val.get("sharpe_ratio", 0.0))

    return val_sharpe, val_turnover, fold_sharpes


def _parse_cost0_response(response):
    """Extract val sharpe from cost-free runner response."""
    oos = response.get("out_of_sample", {})
    return oos.get("sharpe_ratio", 0.0)


# ============================================================================
# Flag computation (pure Python arithmetic)
# ============================================================================

def _compute_flags(baseline_val_sharpe, cost0_val_sharpe, fold_sharpes, baseline_val_turnover):
    """
    Compute diagnosis flags from val-segment metrics only.

    Returns dict with keys: cost_bound, no_signal, unstable,
    regime_dependent, high_turnover.
    """
    cost_bound = (
        (cost0_val_sharpe - baseline_val_sharpe) >= COST_BOUND_DELTA
        and cost0_val_sharpe > 0
    )

    no_signal = cost0_val_sharpe <= NO_SIGNAL_THRESHOLD

    unstable = False
    regime_dependent = False
    if fold_sharpes and len(fold_sharpes) > 1:
        spread = max(fold_sharpes) - min(fold_sharpes)
        unstable = spread > UNSTABLE_SPREAD
        regime_dependent = (
            any(s > REGIME_POSITIVE for s in fold_sharpes)
            and any(s < REGIME_NEGATIVE for s in fold_sharpes)
        )

    high_turnover = baseline_val_turnover > HIGH_TURNOVER_THRESHOLD

    return {
        "cost_bound": cost_bound,
        "no_signal": no_signal,
        "unstable": unstable,
        "regime_dependent": regime_dependent,
        "high_turnover": high_turnover,
    }


# ============================================================================
# Diagnosis
# ============================================================================

def diagnose_family(family_key, knowledge, runner_url):
    """
    Diagnose a single family by re-running with CV folds (single) or
    cost-free (both single and cross).

    Returns (report: dict, error: str|None).
    On error: report has diagnosis_error, error is the message.
    On success: error is None.
    """
    family = knowledge.get("families", {}).get(family_key, {})
    family_type = family.get("family_type", "single")

    evidence = _find_most_recent_evidence(family_key, knowledge)
    if evidence is None:
        error_msg = f"No evidence (rejected/near_miss) found for {family_key}"
        return {"diagnosis_error": error_msg, "family_key": family_key}, error_msg

    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        if family_type == "cross":
            return _diagnose_cross_family(family_key, knowledge, evidence, runner_url, now_iso)
        else:
            return _diagnose_single_family(family_key, evidence, runner_url, now_iso)
    except Exception as e:
        error_msg = f"{e}"
        return {
            "diagnosis_error": error_msg,
            "family_key": family_key,
            "family_type": family_type,
        }, error_msg


def _diagnose_single_family(family_key, evidence, runner_url, now_iso):
    """Run diagnosis for a single-name (param) family."""
    # Recover params from evidence
    params = evidence.get("params", {})
    strategy = evidence.get("strategy", evidence.get("hypothesis", {}).get("strategy", ""))
    symbol = evidence.get("symbol", evidence.get("hypothesis", {}).get("symbol", ""))

    url = f"{runner_url.rstrip('/')}/run"

    # 1. Baseline: /run with cv_folds=3
    baseline_payload = {
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "cv_folds": 3,
    }
    baseline_resp = _post_json(url, baseline_payload)
    baseline_val_sharpe, baseline_val_turnover, fold_sharpes = _parse_baseline_response(baseline_resp)

    # 2. Cost-free: /run with cost_bps=0.0
    cost0_payload = {
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "cost_bps": 0.0,
    }
    cost0_resp = _post_json(url, cost0_payload)
    cost0_val_sharpe = _parse_cost0_response(cost0_resp)

    flags = _compute_flags(baseline_val_sharpe, cost0_val_sharpe, fold_sharpes, baseline_val_turnover)

    report = {
        "family_key": family_key,
        "family_type": "single",
        "diagnosed_at": now_iso,
        "flags": flags,
        "baseline": {
            "val_sharpe": baseline_val_sharpe,
            "val_turnover": baseline_val_turnover,
            "fold_sharpes": fold_sharpes,
        },
        "cost_free": {
            "val_sharpe": cost0_val_sharpe,
        },
        "folds_available": fold_sharpes is not None,
    }

    return report, None


def _diagnose_cross_family(family_key, knowledge, evidence, runner_url, now_iso):
    """Run diagnosis for a cross-sectional (manifest) family."""
    # For cross families, use the recorded evaluation as baseline.
    # We check if the evidence is a rejected entry (which has evaluation)
    # or a near_miss entry.
    if "evaluation" in evidence:
        baseline_eval = evidence.get("evaluation", {})
    else:
        # near_miss entry — use its val_sharpe as baseline
        baseline_eval = {"sharpe_ratio": evidence.get("val_sharpe", 0.0)}

    baseline_val_sharpe = baseline_eval.get("sharpe_ratio", 0.0)

    # Recover strategy info
    strategy = evidence.get("strategy", evidence.get("hypothesis", {}).get("strategy", ""))
    symbol = evidence.get("symbol", evidence.get("hypothesis", {}).get("symbol", ""))
    params = evidence.get("params", evidence.get("hypothesis", {}).get("params", {}))

    # For cross families, we attempt the cost-free run via /run with cost_bps=0.0
    # using the strategy/symbol/params we have.  Full manifest re-run is not
    # possible without the original manifest spec (see PR description).
    url = f"{runner_url.rstrip('/')}/run"

    cost0_payload = {
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "cost_bps": 0.0,
    }

    try:
        cost0_resp = _post_json(url, cost0_payload)
        cost0_val_sharpe = _parse_cost0_response(cost0_resp)
        cost0_error = None
    except Exception as e:
        # If /run doesn't work for cross (no manifest), try /run_manifest
        # with reconstructed minimal manifest
        cost0_val_sharpe = baseline_val_sharpe
        cost0_error = f"cost-free run failed (cross family, /run may not support manifest): {e}"

    if cost0_error and "backtest_result" in evidence:
        # Last resort: try to use backtest_result config to run via /run_manifest
        bt = evidence["backtest_result"]
        # Extract manifest info if available
        manifest_name = bt.get("manifest_name") or strategy.replace("manifest:", "")
        xs_config = bt.get("cross_sectional", {})
        wf_config = bt.get("walkforward", {})

        manifest_payload = {
            "name": manifest_name,
            "code_b64": "",
            "data_sources": [],
            "evaluator": {
                "type": "portfolio",
                "metrics": ["sharpe"],
                "extras": {"cost_bps": 0.0},
            },
            "execution_mode": params.get("execution_mode", "structured"),
        }
        if xs_config:
            manifest_payload["evaluator"]["extras"]["construction_mode"] = xs_config.get("construction_mode", "top_k")
            manifest_payload["evaluator"]["extras"]["rebalance"] = xs_config.get("rebalance", "monthly")

        try:
            manifest_url = f"{runner_url.rstrip('/')}/run_manifest"
            cost0_resp = _post_json(manifest_url, manifest_payload)
            # Parse manifest response
            oos = cost0_resp.get("out_of_sample", {})
            cost0_val_sharpe = oos.get("sharpe_ratio", cost0_resp.get("metrics", {}).get("val_sharpe", baseline_val_sharpe))
            cost0_error = None
        except Exception:
            pass  # keep cost0_val_sharpe as baseline

    flags = _compute_flags(baseline_val_sharpe, cost0_val_sharpe, None, 0.0)

    report = {
        "family_key": family_key,
        "family_type": "cross",
        "diagnosed_at": now_iso,
        "flags": flags,
        "baseline": {
            "val_sharpe": baseline_val_sharpe,
        },
        "cost_free": {
            "val_sharpe": cost0_val_sharpe,
        },
        "folds_available": False,
    }
    if cost0_error:
        report.setdefault("warnings", []).append(cost0_error)

    return report, None


# ============================================================================
# Persistence
# ============================================================================

def _sanitize_filename(name):
    """Replace |, /, and : with _ in filenames."""
    return name.replace("|", "_").replace("/", "_").replace(":", "_")


def _save_report(report):
    """Write report JSON to review_reports/<timestamp>_<sanitized-key>.json."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    family_key = report.get("family_key", "unknown")
    safe_key = _sanitize_filename(family_key)
    filename = f"{ts}_{safe_key}.json"
    REVIEW_REPORTS_DIR.mkdir(exist_ok=True)
    path = REVIEW_REPORTS_DIR / filename
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def _append_review_summary(knowledge, summary):
    """Append compact summary to knowledge['reviews'], cap at 50."""
    reviews = knowledge.setdefault("reviews", [])
    reviews.append(summary)
    if len(reviews) > 50:
        knowledge["reviews"] = reviews[-50:]


def _update_review_state(knowledge, family_key, iso_timestamp):
    """Update review_state with last_review_at and per-family last_diagnosed_at."""
    state = knowledge.setdefault("review_state", {})
    state["last_review_at"] = iso_timestamp
    reviewed = state.setdefault("reviewed", {})
    reviewed.setdefault(family_key, {})["last_diagnosed_at"] = iso_timestamp


# ============================================================================
# Flags to comma-separated string
# ============================================================================

def _active_flags(flags):
    """Return comma-separated list of True flag names, or 'none'."""
    active = [k for k, v in flags.items() if v]
    return ",".join(active) if active else "none"


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Strategy Review — Diagnosis Stage (PR-B2)")
    parser.add_argument("--max-families", type=int, default=None,
                        help="Max families to diagnose (default: REVIEW_MAX_FAMILIES env or 3)")
    parser.add_argument("--family", type=str, default=None,
                        help="Diagnose exactly this family key")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select and print candidates, no runner calls, no state writes")
    args = parser.parse_args()

    knowledge = load_knowledge()
    now = datetime.now(timezone.utc).isoformat()
    runner_url = os.environ.get("SANDBOX_RUNNER_URL", "")

    if args.family:
        candidates = [args.family]
    else:
        max_fam = args.max_families if args.max_families is not None else _read_max_families()
        candidates = select_candidates(knowledge, now, max_fam)

    if args.dry_run:
        print(f"DRY_RUN: {len(candidates)} candidate(s) selected")
        for c in candidates:
            fam = knowledge.get("families", {}).get(c, {})
            print(f"  CANDIDATE {c} lifecycle={fam.get('lifecycle', '?')} "
                  f"best_val_sharpe={fam.get('best_val_sharpe', '?')} "
                  f"family_type={fam.get('family_type', '?')}")
        print("DRY_RUN: no runner calls, no state writes")
        return

    if not runner_url:
        print("ERROR: SANDBOX_RUNNER_URL not set", file=sys.stderr)
        sys.exit(1)

    diagnosed = 0
    errors = 0
    skipped = 0

    for family_key in candidates:
        fam = knowledge.get("families", {}).get(family_key, {})
        if fam.get("lifecycle") == FamilyLifecycle.KILLED:
            skipped += 1
            continue

        report, error = diagnose_family(family_key, knowledge, runner_url)

        if error:
            errors += 1
            print(f"REVIEW_DIAG_ERROR {family_key} error={error}")
            continue

        diagnosed += 1
        flags_str = _active_flags(report.get("flags", {}))
        base_sharpe = report.get("baseline", {}).get("val_sharpe", 0)
        cost0_sharpe = report.get("cost_free", {}).get("val_sharpe", 0)
        print(f"REVIEW_DIAG {family_key} flags={flags_str} "
              f"base_val_sharpe={base_sharpe:.4f} cost0_val_sharpe={cost0_sharpe:.4f}")

        # Persist
        report_path = _save_report(report)
        summary = {
            "family_key": family_key,
            "flags": flags_str,
            "base_val_sharpe": round(base_sharpe, 4),
            "cost0_val_sharpe": round(cost0_sharpe, 4),
            "report": str(report_path.name),
            "diagnosed_at": report.get("diagnosed_at", ""),
        }
        _append_review_summary(knowledge, summary)
        _update_review_state(knowledge, family_key, now)
        save_knowledge(knowledge)

    print(f"REVIEW_SUMMARY diagnosed={diagnosed} errors={errors} skipped={skipped}")


if __name__ == "__main__":
    main()
