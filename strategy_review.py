#!/usr/bin/env python3
"""
Strategy Review — Diagnosis + Verdict stage (PR-B2 + PR-C).

Picks recently-failed strategy families, re-measures them via the
sandbox runner, computes machine-readable diagnosis flags, feeds
reports to an LLM judge for refine/keep/kill verdicts, and applies
those verdicts mechanically.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from loop_constants import FamilyLifecycle, REFINE_CAP

BASE_DIR = Path(__file__).resolve().parent
REVIEW_REPORTS_DIR = BASE_DIR / "review_reports"
KNOWLEDGE_FILE = BASE_DIR / "knowledge.json"

# Import autonomous_loop helpers (import-time mkdir side effect is acceptable,
# same pattern as the loop itself).
from autonomous_loop import _family_key, _derive_family_type, load_knowledge, save_knowledge, get_killed_families

# LLM HTTP helper (stdlib-only, same pattern as strategy_ideation.py)
from llm_hypothesis import _http_post_json as _llm_post_json

# Backlog for refine entries
from backlog import Backlog, _new_entry as _bl_new_entry

# Minimum trials before a family can be killed
MIN_TRIALS_FOR_KILL = 3

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
    """Run diagnosis for a cross-sectional (manifest) family.

    Baseline-only: the full manifest spec (code_b64, data_sources) is not
    persisted in knowledge.json near_misses/rejected entries, so cost-free
    re-run is unavailable.  Zero HTTP calls.
    """
    # Use the recorded evaluation as baseline.
    if "evaluation" in evidence:
        baseline_eval = evidence.get("evaluation", {})
        source = "recorded_evaluation"
    else:
        # near_miss entry — use its val_sharpe as baseline
        baseline_eval = {"sharpe_ratio": evidence.get("val_sharpe", 0.0)}
        source = "recorded_near_miss"

    baseline_val_sharpe = baseline_eval.get("sharpe_ratio", 0.0)

    report = {
        "family_key": family_key,
        "family_type": "cross",
        "diagnosed_at": now_iso,
        "diagnosis_scope": "baseline_only",
        "flags": [],
        "baseline": {
            "val_sharpe": baseline_val_sharpe,
            "source": source,
        },
        "cost_free": None,
        "folds_available": False,
        "warnings": [
            "manifest spec not persisted in knowledge; cost-free re-run unavailable"
        ],
    }

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
    """Return comma-separated list of True flag names, or 'none'.

    Accepts both a dict (single-family) and a list (cross-family baseline-only).
    """
    if isinstance(flags, list):
        return ",".join(flags) if flags else "none"
    active = [k for k, v in flags.items() if v]
    return ",".join(active) if active else "none"


# ============================================================================
# LLM judge (PR-C)
# ============================================================================

def _get_review_llm_config():
    """Read LLM config at call time (not import time)."""
    model = os.environ.get("REVIEW_LLM_MODEL", "deepseek-v4-pro")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("REVIEW_LLM_BASE_URL", "https://api.deepseek.com/v1")
    return {"model": model, "api_key": api_key, "base_url": base_url}


def _call_review_llm(messages):
    """Call the LLM for a review verdict. Returns parsed JSON dict.

    Raises on HTTP/parse errors — callers must fail-open.
    """
    cfg = _get_review_llm_config()
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }
    body = _llm_post_json(
        f"{cfg['base_url'].rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=120,
    )
    content = body["choices"][0]["message"]["content"].strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3].strip()
    return json.loads(content)


def _build_judge_prompt(report, family, knowledge):
    """Build the LLM messages for judging a family."""
    family_key = report.get("family_key", "")
    family_type = family.get("family_type", "single")
    flags = report.get("flags", {})
    if isinstance(flags, list):
        flags_str = ", ".join(flags) if flags else "none"
    else:
        active = [k for k, v in flags.items() if v]
        flags_str = ", ".join(active) if active else "none"

    # Aggregate data
    n_trials = family.get("n_trials", 0)
    best_val_sharpe = family.get("best_val_sharpe", 0)
    gate_failures = family.get("gate_failures", {})
    refine_count = family.get("refine_count", 0)

    # Val-segment numbers from report (holdout already excluded by PR-B2)
    baseline = report.get("baseline", {})
    cost_free = report.get("cost_free")
    val_sharpe = baseline.get("val_sharpe", 0)
    val_turnover = baseline.get("val_turnover", 0)
    fold_sharpes = baseline.get("fold_sharpes")
    cost0_sharpe = cost_free.get("val_sharpe") if cost_free else None

    # Last 5 near-miss entries for this family
    near_miss_lines = []
    family_type_for_lookup = family_type
    if family_key.startswith("manifest:"):
        family_type_for_lookup = "cross"
    near_list = "near_misses_cross" if family_type_for_lookup == "cross" else "near_misses"
    family_near_misses = []
    for entry in knowledge.get(near_list, []):
        strategy = entry.get("strategy", "")
        symbol = entry.get("symbol", "")
        ft = _derive_family_type(entry)
        key = _family_key(strategy, symbol, ft)
        if key == family_key:
            family_near_misses.append(entry)
    if family_near_misses:
        family_near_misses.sort(key=lambda e: e.get("date", ""), reverse=True)
        for nm in family_near_misses[:5]:
            p = nm.get("params", {})
            vs = nm.get("val_sharpe", "?")
            hs = nm.get("holdout_sharpe", "?")
            gate = nm.get("failed_gate", "?")
            near_miss_lines.append(
                f"  - params={json.dumps(p)}, val_sharpe={vs}, holdout_sharpe={hs}, failed_gate={gate}"
            )

    system_prompt = (
        "You are a quantitative strategy reviewer. Your job is to read a machine-generated "
        "diagnosis report and decide whether to refine, keep, or kill the strategy family. "
        "All arithmetic is precomputed — do not recompute. Flags are ground truth. "
        "Choose exactly one verdict."
    )

    baseline_block = f"val_sharpe={val_sharpe}"
    if val_turnover:
        baseline_block += f", val_turnover={val_turnover}"
    if fold_sharpes is not None:
        baseline_block += f", fold_sharpes={fold_sharpes}"

    cost_free_block = f"cost_free.val_sharpe={cost0_sharpe}" if cost0_sharpe is not None else "cost_free: n/a"

    near_miss_block = ""
    if near_miss_lines:
        near_miss_block = "Last 5 near-misses for this family:\n" + "\n".join(near_miss_lines)
    else:
        near_miss_block = "No near-miss entries for this family."

    user_prompt = f"""Diagnosis report for family: {family_key}

Family aggregates:
  n_trials={n_trials}, best_val_sharpe={best_val_sharpe}, refine_count={refine_count}, family_type={family_type}
  gate_failures: validation={gate_failures.get('validation', 0)}, deflation={gate_failures.get('deflation', 0)}, holdout={gate_failures.get('holdout', 0)}

Diagnosis:
  active_flags: {flags_str}
  baseline: {baseline_block}
  {cost_free_block}

{near_miss_block}

Rules:
- All arithmetic is precomputed — do not recompute anything.
- Flags are ground truth. If a flag is active, it is real.
- Choose exactly one verdict: refine, keep, or kill.
- "refine": the strategy has an addressable flaw (e.g. cost_bound → try longer windows; high_turnover → reduce trade frequency). Provide a refine_proposal with changed params (same strategy/symbol) and a change_summary motivating the change from the flag.
- "keep": the strategy shows promise but needs more data; or the flags are mild and not actionable.
- "kill": the strategy is consistently bad, has no signal, or has already been refined multiple times with no improvement.
- If verdict is refine, refine_proposal is REQUIRED. Otherwise it must be null.
- refine_proposal.params must be a dict with the same strategy/symbol and changed param values (int/float/str/bool).

Return ONLY this strict JSON:
{{"verdict": "refine|keep|kill", "rationale": "one sentence explaining why", "refine_proposal": {{"params": {{...}}, "change_summary": "..."}} | null}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def judge_family(report, family, knowledge):
    """Call the LLM to get a verdict for one family.

    Returns a verdict dict: {verdict, rationale, refine_proposal}.
    On ANY failure (HTTP, parse, validation) → fail-open keep.
    Prints REVIEW_JUDGE_FAILOPEN to stdout on failure.
    """
    family_key = report.get("family_key", "unknown")
    try:
        messages = _build_judge_prompt(report, family, knowledge)
        response = _call_review_llm(messages)
    except Exception as e:
        print(f"REVIEW_JUDGE_FAILOPEN {family_key} reason=exception: {e}")
        return {"verdict": "keep", "rationale": f"llm_failure: exception: {e}", "refine_proposal": None}

    # Validate response shape
    if not isinstance(response, dict):
        print(f"REVIEW_JUDGE_FAILOPEN {family_key} reason=not_a_dict")
        return {"verdict": "keep", "rationale": "llm_failure: not_a_dict", "refine_proposal": None}

    verdict = response.get("verdict", "")
    if verdict not in ("refine", "keep", "kill"):
        print(f"REVIEW_JUDGE_FAILOPEN {family_key} reason=invalid_verdict: {verdict!r}")
        return {"verdict": "keep", "rationale": f"llm_failure: invalid_verdict: {verdict!r}", "refine_proposal": None}

    rationale = response.get("rationale", "")
    refine_proposal = response.get("refine_proposal")

    if verdict == "refine":
        if refine_proposal is None:
            print(f"REVIEW_JUDGE_FAILOPEN {family_key} reason=missing_refine_proposal")
            return {"verdict": "keep", "rationale": "llm_failure: missing_refine_proposal", "refine_proposal": None}
        if not isinstance(refine_proposal, dict):
            print(f"REVIEW_JUDGE_FAILOPEN {family_key} reason=refine_proposal_not_dict")
            return {"verdict": "keep", "rationale": "llm_failure: refine_proposal_not_dict", "refine_proposal": None}
        params = refine_proposal.get("params")
        if not isinstance(params, dict) or len(params) == 0:
            print(f"REVIEW_JUDGE_FAILOPEN {family_key} reason=bad_refine_params")
            return {"verdict": "keep", "rationale": "llm_failure: bad_refine_params", "refine_proposal": None}
        for v in params.values():
            if not isinstance(v, (int, float, str, bool)):
                print(f"REVIEW_JUDGE_FAILOPEN {family_key} reason=bad_param_type: {type(v).__name__}")
                return {"verdict": "keep", "rationale": f"llm_failure: bad_param_type: {type(v).__name__}", "refine_proposal": None}

    return {"verdict": verdict, "rationale": rationale, "refine_proposal": refine_proposal}


# ============================================================================
# Verdict application (PR-C) — pure mechanics, no LLM
# ============================================================================

def apply_verdict(family_key, verdict_dict, knowledge, backlog):
    """Apply a verdict to the knowledge base and backlog.

    Returns a string summary of what happened.
    """
    families = knowledge.setdefault("families", {})
    family = families.get(family_key, {})
    verdict = verdict_dict["verdict"]
    rationale = verdict_dict["rationale"]
    now_iso = datetime.now(timezone.utc).isoformat()
    family_type = family.get("family_type", "single")

    # Append to reviews summary
    summary = {
        "family_key": family_key,
        "verdict": verdict,
        "rationale": rationale,
        "at": now_iso,
    }
    _append_review_summary(knowledge, summary)

    if verdict == "kill":
        n_trials = family.get("n_trials", 0)
        if n_trials < MIN_TRIALS_FOR_KILL:
            # Downgrade: insufficient evidence
            downgraded_rationale = rationale + " (downgraded: insufficient evidence)"
            print(f"REVIEW_VERDICT {family_key} verdict=keep rationale=\"{rationale}\"")
            return "downgrade_to_keep"

        family["lifecycle"] = FamilyLifecycle.KILLED
        family["kill_reason"] = "auto: " + rationale
        print(f"REVIEW_VERDICT {family_key} verdict=kill rationale=\"{rationale}\"")

    elif verdict == "refine":
        if family_type == "cross":
            # Cross families cannot be refined (no manifest spec persisted)
            print(f"REVIEW_VERDICT {family_key} verdict=keep rationale=\"{rationale} (downgraded: cross refine unavailable)\"")
            return "downgrade_cross_refine"

        refine_count = family.get("refine_count", 0)
        if refine_count >= REFINE_CAP:
            # Auto-kill: refine cap exhausted (ignores MIN_TRIALS_FOR_KILL)
            family["lifecycle"] = FamilyLifecycle.KILLED
            family["kill_reason"] = "auto: refine cap exhausted"
            print(f"REVIEW_VERDICT {family_key} verdict=kill rationale=\"refine cap exhausted\"")
            return "kill_refine_cap"

        # Refine: increment count, set lifecycle, add backlog entry
        family["refine_count"] = refine_count + 1
        family["lifecycle"] = FamilyLifecycle.REFINING

        refine_proposal = verdict_dict.get("refine_proposal") or {}
        params = refine_proposal.get("params", {})
        change_summary = refine_proposal.get("change_summary", "")

        # Determine strategy/symbol from family_key
        parts = family_key.split("|", 1)
        strategy = parts[0] if len(parts) >= 1 else ""
        symbol = parts[1] if len(parts) >= 2 else ""

        backlog_entry = _bl_new_entry(
            "param",
            0.95,
            {"kind": "review_refine", "ref": family_key},
            {"strategy": strategy, "symbol": symbol, "params": params},
            {"extra_criteria": []},
        )
        backlog_entry["created_at"] = now_iso
        accepted, eid = backlog.add_entry(backlog_entry)

        print(f"REVIEW_VERDICT {family_key} verdict=refine rationale=\"{rationale}\"")

    else:  # keep
        print(f"REVIEW_VERDICT {family_key} verdict=keep rationale=\"{rationale}\"")

    return verdict


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Strategy Review — Diagnosis + Verdict stage (PR-C)")
    parser.add_argument("--max-families", type=int, default=None,
                        help="Max families to diagnose (default: REVIEW_MAX_FAMILIES env or 3)")
    parser.add_argument("--family", type=str, default=None,
                        help="Diagnose exactly this family key")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select candidates, print what would happen, no LLM calls, no writes")
    parser.add_argument("--no-judge", action="store_true",
                        help="Diagnosis only — skip LLM judging (PR-B2 behavior)")
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
        print("DRY_RUN: no runner calls, no LLM calls, no state writes")
        return

    if not runner_url:
        print("ERROR: SANDBOX_RUNNER_URL not set", file=sys.stderr)
        sys.exit(1)

    # Initialize backlog for refine entries
    backlog_path = os.environ.get("BACKLOG_PATH", str(BASE_DIR / "backlog.json"))
    backlog = Backlog(backlog_path)

    diagnosed = 0
    errors = 0
    skipped = 0
    kill_count = 0
    refine_count = 0
    keep_count = 0
    failopen_count = 0

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
        cost_free = report.get("cost_free")
        cost0_sharpe = None
        if cost_free is not None:
            cost0_sharpe = cost_free.get("val_sharpe", 0)
            print(f"REVIEW_DIAG {family_key} flags={flags_str} "
                  f"base_val_sharpe={base_sharpe:.4f} cost0_val_sharpe={cost0_sharpe:.4f}")
        else:
            print(f"REVIEW_DIAG {family_key} flags={flags_str} "
                  f"base_val_sharpe={base_sharpe:.4f} cost0_val_sharpe=n/a")

        # Persist report
        report_path = _save_report(report)

        if args.no_judge:
            # PR-B2 behavior: diagnosis only
            summary = {
                "family_key": family_key,
                "flags": flags_str,
                "base_val_sharpe": round(base_sharpe, 4),
                "cost0_val_sharpe": round(cost0_sharpe, 4) if cost0_sharpe is not None else None,
                "report": str(report_path.name),
                "diagnosed_at": report.get("diagnosed_at", ""),
            }
            _append_review_summary(knowledge, summary)
            _update_review_state(knowledge, family_key, now)
            save_knowledge(knowledge)
            continue

        # PR-C: judge + apply
        verdict_dict = judge_family(report, fam, knowledge)
        result = apply_verdict(family_key, verdict_dict, knowledge, backlog)

        # Tally
        v = verdict_dict["verdict"]
        if v == "kill":
            kill_count += 1
        elif v == "refine":
            refine_count += 1
        elif v == "keep":
            if "llm_failure" in verdict_dict["rationale"]:
                failopen_count += 1
            else:
                keep_count += 1

        _update_review_state(knowledge, family_key, now)
        save_knowledge(knowledge)

    if args.no_judge:
        print(f"REVIEW_SUMMARY diagnosed={diagnosed} errors={errors} skipped={skipped}")
    else:
        print(f"REVIEW_SUMMARY diagnosed={diagnosed} errors={errors} skipped={skipped} "
              f"kill={kill_count} refine={refine_count} keep={keep_count} failopen={failopen_count}")


if __name__ == "__main__":
    main()
