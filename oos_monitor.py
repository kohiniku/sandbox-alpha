#!/usr/bin/env python3
"""
Rolling Out-of-Sample Monitor for Adopted Strategies.

Re-checks adopted strategies on data that did not exist when they were adopted.
Dispatch is per adoption type (issue #88 — before the fix every manifest:/codegen
entry was funnelled through /run and deterministically skipped with
"unknown strategy", and checked=0 was silently reported as [SILENT]):

  - param / single-name entries (strategy in STRATEGY_TEMPLATES): POST /run with
    metrics_since=adoption_date (the only path the runner restricts by date),
    read since_metrics, record oos_kind=since_adoption.
  - manifest:* entries: resolve the manifest spec (stored on the record, else
    from backlog history keyed by name + universe hash), re-run via /run_manifest
    with data_sources[].start overridden to the adoption date — the runner
    ignores metrics_since on /run_manifest, so the window override is the only
    way to restrict to post-adoption data. Record the holdout-split metrics
    (oos_kind=manifest_since_adoption).
  - codegen entries: resolve the code (code_spec on the record, else from
    backlog history by code_hash + symbol), re-run via /run_code, record the
    holdout split. /run_code cannot restrict the window, so the record is
    explicitly marked oos_kind=codegen_holdout (freshest 20% of full history).

Loudness (issue #88): when the monitor cannot check anything for structural
reasons (checked=0 with error-class skips) it prints an OOS_BROKEN marker and
__main__ exits 3, so a broken monitor can never again be mistaken for "no
adopted strategies" (which the delivery policy maps to [SILENT]).

Does NOT demote — v1 is record-and-report only.
Stdlib-only, matches existing codebase style.
"""
import base64
import copy
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Re-use autonomous_loop's infrastructure
from autonomous_loop import (
    load_knowledge,
    save_knowledge,
    run_backtest,
    _classify_http_status,
    _extract_universe_meta,
    _get_loop_config,
    KNOWLEDGE_FILE,
)

# Skips that are NOT structural failures: the entry simply has nothing to
# check yet, or too little post-adoption data. A run whose every skip is in
# this set is legitimately quiet; any other skip is a broken-monitor signal.
_BENIGN_SKIP_ERRORS = frozenset({
    "adoption_in_future",
    "no_adoption_date",
    "insufficient_data",
})

# Per-run counters, read by __main__ to decide the exit code (mirrors the
# _RUN_STATS pattern in autonomous_loop, issue #83/#88).
_OOS_STATS = {}


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


# ---------------------------------------------------------------------------
# Runner HTTP helper
# ---------------------------------------------------------------------------

def _post_json(url, payload, timeout=300):
    """POST a JSON payload and return the parsed dict.

    Never raises — network/HTTP/parse failures are returned as
    {"error": ..., "error_type": ...} dicts (same contract as
    autonomous_loop._run_backtest_sandbox).
    """
    body = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read().decode("utf-8")
            status = resp.status

        if status != 200:
            return {"error": f"Sandbox runner returned HTTP {status}: {response_body[:500]}",
                    "error_type": _classify_http_status(status)}

        result = json.loads(response_body)
        if isinstance(result, dict) and "error" in result:
            result.setdefault("error_type", "code")
        return result

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        return {"error": f"Sandbox runner HTTP {e.code}: {error_body}",
                "error_type": _classify_http_status(e.code)}
    except urllib.error.URLError as e:
        return {"error": f"Sandbox runner connection error: {e.reason}",
                "error_type": "infra"}
    except json.JSONDecodeError as e:
        return {"error": f"Sandbox runner JSON parse error: {e}",
                "error_type": "infra"}
    except Exception as e:
        return {"error": f"Sandbox runner error: {e}", "error_type": "infra"}


# ---------------------------------------------------------------------------
# Spec resolution (issue #88)
# ---------------------------------------------------------------------------
# Adopted manifest/codegen records historically carry only the strategy name
# and universe hash — not the manifest/code body. The consumed specs survive
# forever in backlog history (archive_stale only touches PENDING entries), so
# we resolve by name + universe-hash / code-hash + symbol. Go-forward records
# carry the spec directly (see autonomous_loop._consume_backlog_entry) and are
# preferred over backlog resolution.

def _backlog_entries():
    from backlog import Backlog
    bl = Backlog(_get_loop_config()["backlog_path"])
    return bl.load().get("entries", [])


def _resolve_manifest_spec(name, expected_universe_hash=None):
    """Resolve a manifest spec by name from backlog history.

    Only reuses an entry whose universe hashes to the adopted entry's
    universe:<hash> symbol — a same-name manifest on a different universe is a
    different strategy and must not be reused. Returns the spec dict or None.
    """
    for e in _backlog_entries():
        if e.get("type") != "manifest":
            continue
        spec = e.get("spec", {})
        if spec.get("name") != name:
            continue
        _, universe_hash, _ = _extract_universe_meta(spec)
        if expected_universe_hash and universe_hash != expected_universe_hash:
            continue
        return spec
    return None


def _resolve_code_spec(code_hash, symbol):
    """Resolve a codegen spec by code-hash + symbol from backlog history."""
    for e in _backlog_entries():
        if e.get("type") != "code":
            continue
        spec = e.get("spec", {})
        if spec.get("symbol") != symbol:
            continue
        code = spec.get("code", "")
        if code_hash and hashlib.sha256(code.encode("utf-8")).hexdigest() == code_hash:
            return spec
    return None


# ---------------------------------------------------------------------------
# Per-type OOS checks
# ---------------------------------------------------------------------------

def _run_param_oos(entry, adoption_date, today):
    """Param / single-name strategy: /run with metrics_since=adoption_date."""
    hyp = entry.get("hypothesis", {})
    params = hyp.get("params", {})

    warmup = _estimate_warmup_days(params)

    # Run backtest with metrics_since=adoption_date
    metrics_since_str = adoption_date.strftime("%Y-%m-%d")
    result = run_backtest(hyp, metrics_since=metrics_since_str)
    if "error" in result:
        error_type = result.get("error_type", "unknown")
        return None, f"runner_{error_type}"

    # Extract since_metrics (post-adoption data only)
    since_metrics = result.get("since_metrics")
    if since_metrics is None:
        return None, "no_since_metrics"

    n_days = since_metrics.get("n_days", 0)
    if n_days < 7:
        return None, "insufficient_data"

    oos_sharpe = since_metrics.get("sharpe_ratio")
    oos_return = since_metrics.get("total_return_pct", 0.0)
    oos_max_dd = since_metrics.get("max_drawdown_pct", 0.0)

    if oos_sharpe is None:
        return None, "no_sharpe_in_since_metrics"

    oos_record = {
        "date": today.strftime("%Y-%m-%d"),
        "window_days": (today - adoption_date).days,
        "oos_kind": "since_adoption",
        "oos_sharpe": oos_sharpe,
        "oos_return_pct": float(oos_return),
        "oos_max_drawdown_pct": float(oos_max_dd),
    }
    return oos_record, None


def _run_manifest_oos(entry, adoption_date, today):
    """Manifest strategy: /run_manifest with data_sources[].start overridden to
    the adoption date (the runner ignores metrics_since on /run_manifest)."""
    hyp = entry.get("hypothesis", {})
    strategy = hyp.get("strategy", "manifest:?")
    symbol = hyp.get("symbol", "")
    name = strategy[len("manifest:"):]
    expected_hash = symbol[len("universe:"):] if symbol.startswith("universe:") else None

    # Prefer the spec persisted on the record (go-forward), else backlog history.
    spec = entry.get("manifest_spec")
    if not spec:
        spec = _resolve_manifest_spec(name, expected_hash)
    if spec is None:
        return None, "manifest_spec_unresolved"

    runner_url = _get_loop_config()["runner_url"]
    if not runner_url:
        return None, "runner_unconfigured"

    spec2 = copy.deepcopy(spec)
    start_override = adoption_date.strftime("%Y-%m-%d")
    found_ohlcv = False
    for ds in spec2.get("data_sources", []):
        if ds.get("type") == "ohlcv":
            ds["start"] = start_override
            ds.pop("end", None)
            found_ohlcv = True
    if not found_ohlcv:
        return None, "manifest_no_ohlcv_source"

    # Runner validates code_b64 only (no logic_spec support) — compile DSL to
    # code before sending (issue #99). Adopted DSL manifests persist
    # logic_spec in manifest_spec; the prepared copy is what goes on the wire.
    from strategy_dsl import prepare_spec_for_runner
    prepared, prep_err = prepare_spec_for_runner(spec2)
    if prepared is None:
        return None, f"runner_payload_prep_failed: {prep_err}"
    spec2 = prepared

    result = _post_json(f"{runner_url.rstrip('/')}/run_manifest", spec2, timeout=300)
    if result.get("status") != "ok":
        error_type = result.get("error_type", "unknown")
        return None, f"runner_{error_type}"

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None, "no_manifest_metrics"

    n_days = result.get("n_days", 0)
    if n_days < 7:
        return None, "insufficient_data"

    # The runner splits the post-adoption window internally; the holdout slice
    # is the freshest portion and the one that never existed at adoption.
    oos_sharpe = metrics.get("holdout_sharpe")
    if oos_sharpe is None:
        return None, "no_holdout_metrics"

    oos_record = {
        "date": today.strftime("%Y-%m-%d"),
        "window_days": (today - adoption_date).days,
        "n_days": n_days,
        "oos_kind": "manifest_since_adoption",
        "oos_sharpe": oos_sharpe,
        "oos_return_pct": float(metrics.get("holdout_total_return_pct", 0.0)),
        "oos_max_drawdown_pct": float(metrics.get("holdout_max_drawdown_pct", 0.0)),
        "oos_val_sharpe": metrics.get("val_sharpe"),
    }
    return oos_record, None


def _run_codegen_oos(entry, adoption_date, today):
    """Codegen strategy: /run_code with the resolved code.

    The runner cannot restrict the window on /run_code (probed live: metrics_since
    ignored), so the record uses the holdout split of the full-history re-run and
    is explicitly marked oos_kind=codegen_holdout.
    """
    hyp = entry.get("hypothesis", {})
    symbol = hyp.get("symbol", "unknown")
    code_hash = (entry.get("backtest_result") or {}).get("code_hash")

    code_spec = entry.get("code_spec")
    if not code_spec:
        code_spec = _resolve_code_spec(code_hash, symbol)
    if code_spec is None:
        return None, "code_spec_unresolved"

    code = code_spec.get("code", "")
    if not code:
        return None, "code_spec_unresolved"

    runner_url = _get_loop_config()["runner_url"]
    if not runner_url:
        return None, "runner_unconfigured"

    code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    result = _post_json(
        f"{runner_url.rstrip('/')}/run_code",
        {"code_b64": code_b64, "symbol": symbol},
        timeout=300,
    )
    if "error" in result:
        error_type = result.get("error_type", "unknown")
        return None, f"runner_{error_type}"

    holdout = result.get("holdout") or {}
    oos_sharpe = holdout.get("sharpe_ratio")
    if oos_sharpe is None:
        return None, "no_holdout_metrics"

    n_days = holdout.get("num_days", 0)
    if n_days < 7:
        return None, "insufficient_data"

    oos_record = {
        "date": today.strftime("%Y-%m-%d"),
        "window_days": (today - adoption_date).days,
        "n_days": n_days,
        "oos_kind": "codegen_holdout",
        "oos_sharpe": oos_sharpe,
        "oos_return_pct": float(holdout.get("total_return_pct", 0.0)),
        "oos_max_drawdown_pct": float(holdout.get("max_drawdown_pct", 0.0)),
    }
    return oos_record, None


def run_oos_check(entry, today=None):
    """Run OOS check for a single adopted strategy entry.

    Dispatches per adoption type (issue #88):
      - manifest:<name>  -> /run_manifest with start=adoption_date
      - codegen / code:  -> /run_code (holdout split)
      - anything else    -> /run with metrics_since (param strategies)

    Returns (oos_record, error_string_or_None).
    """
    if today is None:
        today = datetime.now()

    hyp = entry.get("hypothesis", {})
    strategy = hyp.get("strategy", "unknown")
    symbol = hyp.get("symbol", "unknown")

    adoption_date = _parse_adoption_date(entry)
    if adoption_date is None:
        return None, "no_adoption_date"

    window_days = (today - adoption_date).days
    if window_days < 1:
        return None, "adoption_in_future"

    if strategy.startswith("manifest:"):
        return _run_manifest_oos(entry, adoption_date, today)
    if strategy == "codegen" or strategy.startswith("code:"):
        return _run_codegen_oos(entry, adoption_date, today)
    return _run_param_oos(entry, adoption_date, today)


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
        print("OOS_SUMMARY checked=0 negative=0 skipped=0")
        _OOS_STATS.update({"checked": 0, "negative": 0, "skipped": 0, "error_skips": 0})
        return 0, 0, 0

    checked = 0
    negative = 0
    skipped = 0
    error_skips = 0

    for entry in adopted:
        hyp = entry.get("hypothesis", {})
        strategy = hyp.get("strategy", "unknown")
        symbol = hyp.get("symbol", "unknown")
        label = f"{strategy}/{symbol}"

        oos_record, error = run_oos_check(entry, today=today)

        if error is not None:
            print(f"OOS_STATUS {label} days=0 sharpe=n/a return=n/a% error={error}")
            skipped += 1
            if error not in _BENIGN_SKIP_ERRORS:
                error_skips += 1
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

    print(f"OOS_SUMMARY checked={checked} negative={negative} skipped={skipped}")
    if skipped:
        print(f"OOS_SKIPPED count={skipped}")
    # Issue #88: a monitor that checked nothing while adoptions exist is broken,
    # not quiet. Print a machine-greppable marker so the delivery policy can
    # distinguish "nothing to check" from "cannot check anything".
    if checked == 0 and error_skips > 0:
        print(f"OOS_BROKEN checked=0 skipped={skipped} error_skips={error_skips} adopted={len(adopted)}")

    _OOS_STATS.update({
        "checked": checked, "negative": negative, "skipped": skipped,
        "error_skips": error_skips, "adopted": len(adopted),
    })
    return checked, negative, skipped


if __name__ == "__main__":
    import cp_emit
    _cp_run_id = cp_emit.emit_run_started("d690a1508eaf", "oos_monitor.py")
    try:
        knowledge = load_knowledge()
        checked, negative, skipped = run_oos_monitor(knowledge)
        # Issue #88: exit non-zero when the monitor exists, adoptions exist, and
        # nothing was checked for structural reasons — never report such a run
        # as a clean 'ok' (jobs.json last_status must not stay 'ok' on OOS_BROKEN).
        broken = (
            _OOS_STATS.get("adopted", 0) > 0
            and _OOS_STATS.get("checked", 0) == 0
            and _OOS_STATS.get("error_skips", 0) > 0
        )
        if broken:
            cp_emit.emit_run_finished(
                _cp_run_id, "error",
                error="OOS_BROKEN: every adopted strategy was skipped for structural reasons",
            )
            sys.exit(3)
        cp_emit.emit_run_finished(_cp_run_id, "ok")
        sys.exit(0)
    except BaseException as _cp_exc:
        cp_emit.emit_run_finished(
            _cp_run_id, "error", error=f"{type(_cp_exc).__name__}: {_cp_exc}"
        )
        raise
