#!/usr/bin/env python3
"""
Strategy Ideation — nightly, research-driven proposal pipeline.

Reads accumulated research knowledge + failure history, calls an LLM
for up to N proposals, validates each, and writes them to the backlog.

Usage:
    python3 strategy_ideation.py [--max-proposals 5] [--dry-run]

Config (env):
    HYPO_LLM_BASE_URL, HYPO_LLM_MODEL, HYPO_LLM_API_KEY_ENV — same as llm_hypothesis
    RESEARCH_DIRS — colon-separated paths to scan for .md/.json docs (default: ./research)
    KNOWLEDGE_PATH — path to knowledge.json (default: ./knowledge.json)
    BACKLOG_PATH — path to backlog.json (default: ./backlog.json)
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Reuse existing helpers (llm_hypothesis is not owned by either session)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Import STDLIB-only HTTP helper and symbol validator from llm_hypothesis
# (This file only depends on stdlib urllib, matching the runtime container.)
sys.path.insert(0, str(BASE_DIR))
from llm_hypothesis import _http_post_json as _post_json
from llm_hypothesis import _SYMBOL_RE

# Import STRATEGY_TEMPLATES from autonomous_loop (read-only)
from autonomous_loop import STRATEGY_TEMPLATES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MAX_PROPOSALS_DEFAULT = 5
_TOKEN_CAP_CHARS = 16_000  # ~4000 tokens
_RESEARCH_PREVIEW_CHARS = 500
_MAX_RESEARCH_FILES = 10

_SYMBOL_RE_COMPILED = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$")


def _get_llm_config():
    return {
        "base_url": os.environ.get("HYPO_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.environ.get("HYPO_LLM_MODEL", "deepseek-v4-pro"),
        "api_key": os.environ.get(
            os.environ.get("HYPO_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY"), ""
        ),
    }


def _get_research_dirs():
    default = str(BASE_DIR / "research")
    raw = os.environ.get("RESEARCH_DIRS", default)
    return [Path(p).expanduser().resolve() for p in raw.split(":") if p.strip()]


def _get_knowledge_path():
    return Path(os.environ.get("KNOWLEDGE_PATH", str(BASE_DIR / "knowledge.json")))


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def _load_knowledge(path):
    """Load knowledge.json, falling back to empty dict."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {
        "tested": [],
        "tested_combinations": [],
        "adopted": [],
        "rejected": [],
        "superseded": [],
        "families": {},
        "iterations": 0,
    }


def _gather_research_docs(dirs):
    """Collect ~10 most recent .md/.json files across research dirs.

    Returns a list of (filename, first-N-chars) tuples.
    Caps total output at _TOKEN_CAP_CHARS bytes.
    """
    candidates = []
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        for fpath in sorted(d.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if fpath.suffix in (".md", ".json") and fpath.is_file():
                candidates.append(fpath)
            if len(candidates) >= _MAX_RESEARCH_FILES * 3:
                break  # per-dir limit so we don't blow up

    # Global sort by mtime, keep _MAX_RESEARCH_FILES
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = candidates[:_MAX_RESEARCH_FILES]

    previews = []
    total_chars = 0
    for fpath in candidates:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # First line as "title", then _RESEARCH_PREVIEW_CHARS of body
        lines = content.split("\n")
        title = lines[0].strip().lstrip("#").strip() if lines else fpath.name
        body = content[:_RESEARCH_PREVIEW_CHARS]
        chunk = f"## {title}\n{body}"
        if total_chars + len(chunk) > _TOKEN_CAP_CHARS:
            # Truncate last chunk to fit
            remaining = _TOKEN_CAP_CHARS - total_chars
            if remaining > 50:
                chunk = chunk[:remaining]
                previews.append((fpath.name, chunk))
            break
        previews.append((fpath.name, chunk))
        total_chars += len(chunk)

    return previews


def _summarise_rejects(knowledge):
    """Build a compact summary of last 15 rejected entries with gate reasons."""
    rejected = knowledge.get("rejected", [])
    if not rejected:
        return "No rejected entries yet."

    lines = []
    for r in rejected[-15:]:
        hyp = r.get("hypothesis", {})
        ev = r.get("evaluation", {})
        gate = ev.get("gate_results", {})
        strategy = hyp.get("strategy", "?")
        symbol = hyp.get("symbol", "?")
        params = json.dumps(hyp.get("params", {}))
        sharpe = ev.get("sharpe_ratio", "?")

        # Determine gate reason
        if not gate:
            reason = "unknown"
        elif not gate.get("validation", True):
            reason = f"validation failed (sharpe={sharpe})"
        elif not gate.get("holdout", True):
            reason = f"holdout failed (val_sharpe={sharpe})"
        elif gate.get("cluster") == "duplicate_cluster":
            reason = "duplicate_cluster"
        elif gate.get("cluster") == "exhausted_cluster":
            reason = "exhausted_cluster"
        else:
            reason = f"unknown (sharpe={sharpe})"

        lines.append(
            f"  - {strategy}/{symbol} params={params} → {reason}"
        )

    families = knowledge.get("families", {})
    if families:
        lines.append("\nFamily aggregates:")
        for key, fam in sorted(families.items()):
            n = fam.get("n_trials", 0)
            best = fam.get("best_val_sharpe", -999)
            exhausted = " [EXHAUSTED]" if (n >= 3 and best < 0) else ""
            lines.append(f"  {key}: {n} trials, best sharpe={best:.2f}{exhausted}")

    return "\n".join(lines)


def _summarise_code_errors(knowledge):
    """Build a compact RECENT CODE ERRORS section for the LLM context.

    These are strategy ideas whose generated code crashed at runtime —
    they were never actually evaluated. The LLM may re-propose the idea
    with FIXED code, avoiding these exact bugs.
    """
    errors = knowledge.get("errors", [])
    code_errors = [e for e in errors
                   if e.get("evaluation", {}).get("error_type") == "code"]
    if not code_errors:
        return None

    lines = []
    for ce in code_errors[-10:]:  # last 10
        hyp = ce.get("hypothesis", {})
        name = hyp.get("description", hyp.get("strategy", "?"))
        symbol = hyp.get("symbol", "?")
        err_text = ce.get("evaluation", {}).get("error", "?")
        # One-line error: first 120 chars
        one_line = err_text.split("\n")[0][:120]
        lines.append(f"  {name}/{symbol} — {one_line}")

    header = (
        "RECENT CODE ERRORS (these were never evaluated — their generated code crashed):\n"
        "You may re-propose the idea with FIXED code. IMPORTANT: df columns are "
        "capitalized (e.g. 'Close' not 'close', 'Open' not 'open')."
    )
    return header + "\n" + "\n".join(lines)


def _summarise_near_misses(knowledge):
    """Build a compact summary of near-miss entries for the LLM context.

    These are directions that showed signal on validation but FAILED --
    do not re-propose near-identical specs; instead vary symbol, regime
    filter, or horizon. Holdout failures are a warning.
    """
    near_misses = knowledge.get("near_misses", [])
    if not near_misses:
        return None

    lines = []
    for nm in near_misses[-20:]:  # last 20 for token budget
        strategy = nm.get("strategy", "?")
        symbol = nm.get("symbol", "?")
        params = json.dumps(nm.get("params", {}))
        val_s = nm.get("val_sharpe", -999)
        thresh = nm.get("deflated_threshold", 0)
        holdout_s = nm.get("holdout_sharpe")
        gate = nm.get("failed_gate", "?")
        holdout_str = f" holdout_sharpe={holdout_s:.2f}" if holdout_s is not None else ""
        lines.append(
            f"  {strategy}/{symbol} params={params} val_sharpe={val_s:.2f}"
            f" (thresh={thresh:.2f}){holdout_str} -- {gate}"
        )

    header = (
        "NEAR-MISS ARCHIVE (signal on validation but FAILED):\n"
        "Do NOT re-propose near-identical specs (validation-set hill-climbing). "
        "Instead vary direction: different symbol, regime filter, or horizon. "
        "Holdout failures = overfit warning."
    )
    return header + "\n" + "\n".join(lines)


def _summarise_backlog(backlog):
    """Build a compact summary of current backlog to avoid duplicates."""
    data = backlog.load()
    pending = [e for e in data["entries"] if e["status"] == "pending"]
    if not pending:
        return "No pending entries."
    lines = []
    for e in pending[:30]:
        s = e["spec"]
        if e["type"] == "param":
            lines.append(
                f"  param: {s['strategy']}/{s['symbol']} "
                f"params={json.dumps(s['params'])} priority={e['priority']:.2f}"
            )
        else:
            lines.append(
                f"  code: {s.get('name','?')}/{s.get('symbol','?')} "
                f"priority={e['priority']:.2f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_PROPOSAL_JSON_SCHEMA = """{
  "proposals": [
    {
      "type": "param",
      "priority": 0.85,
      "source": {"kind": "paper", "ref": "some-research-report.md"},
      "spec": {
        "strategy": "mean_reversion",
        "symbol": "AAPL",
        "params": {"window": 20, "threshold": 2.0}
      },
      "eval_plan": {"extra_criteria": ["max_hold_days <= 5"]}
    },
    {
      "type": "code",
      "priority": 0.70,
      "source": {"kind": "failure_response", "ref": "sma_crossover|AAPL"},
      "spec": {
        "name": "adaptive_sma",
        "description": "Adaptive SMA with volatility-adjusted windows",
        "symbol": "SPY",
        "code": "import numpy as np\\nimport pandas as pd\\n\\ndef generate_signals(df):\\n    ..."
      },
      "eval_plan": {"extra_criteria": ["max_drawdown >= -20"]}
    }
  ]
}"""


def _build_prompt(knowledge, templates, research_docs, backlog_summary, max_proposals):
    """Build the messages payload for the LLM."""
    strategy_lines = []
    for name, tmpl in templates.items():
        desc = tmpl.get("description", name)
        ps = tmpl["param_space"]
        parts = []
        for pn, pv in ps.items():
            if isinstance(pv, range):
                parts.append(f"{pn}=int({pv.start}..{pv.stop - 1})")
            elif isinstance(pv, list):
                parts.append(f"{pn}=one_of{pv}")
        strategy_lines.append(f"- {name}: {desc}  param_space: {{{', '.join(parts)}}}")

    rejects_summary = _summarise_rejects(knowledge)
    near_misses_text = _summarise_near_misses(knowledge)
    code_errors_text = _summarise_code_errors(knowledge)

    research_text = ""
    if research_docs:
        chunks = [body for _, body in research_docs]
        research_text = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a quantitative researcher proposing trading-strategy candidates.

=== AVAILABLE STRATEGIES (exact param spaces) ===
{chr(10).join(strategy_lines)}

=== FAILURE HISTORY (rejected entries + family aggregates) ===
{rejects_summary}

=== NEAR-MISS ARCHIVE (signal on validation, failed later) ===
{near_misses_text or "(no near-misses recorded yet)"}

=== CURRENT BACKLOG (avoid duplicates) ===
{backlog_summary}

=== RECENT RESEARCH ===
{research_text or "(no research documents available)"}

=== RECENT CODE ERRORS (never evaluated — code crashed) ===
{code_errors_text or "(no recent code errors)"}

=== YOUR TASK ===
Propose up to {max_proposals} new strategy candidates. Ground each in either:
- A cited research document (source.kind="paper", ref=<filename>)
- A specific failure pattern (source.kind="failure_response", ref=<family key like "strategy|symbol">)
- A novel idea (source.kind="idea", ref=<short description>)

For type=param: use EXACT param spaces above. Symbol must match ^[A-Z0-9][A-Z0-9.\\-]{{0,11}}$.
For type=code: include a COMPLETE `def generate_signals(df):` function using only numpy/pandas/math stdlib.

=== INTERFACE CONTRACT (code-type ONLY — VIOLATIONS WILL BE REJECTED) ===
generate_signals(df) receives a pandas DataFrame with:
  - DatetimeIndex (there is NO 'Date' column — do NOT reference df['Date'])
  - Columns: Open, High, Low, Close, Volume  (ALL capitalized — 'close' will crash)
  - Typical length: 200–1000 rows of daily OHLCV data
It MUST return a pandas Series:
  - Index aligned to df.index (same length, same dates)
  - Values in {{-1, 0, 1}} only  (NaN filled to 0)
  - -1 = short, 0 = flat, 1 = long
Allowed imports: numpy and pandas ONLY (no sklearn, no talib, no requests).

GOLDEN EXAMPLE (use as reference — simple SMA crossover):
```
import numpy as np
import pandas as pd

def generate_signals(df):
    fast = df["Close"].rolling(10).mean()
    slow = df["Close"].rolling(30).mean()
    signals = pd.Series(0, index=df.index)
    signals[fast > slow] = 1
    signals[fast < slow] = -1
    return signals
```

CRITICAL CONSTRAINTS:
- extra_criteria may ONLY ADD constraints, never relax the global gates (min_sharpe, max_drawdown, holdout).
- Assign priority (0.0-1.0) with a brief rationale comment.
- AVOID exhausted families (>=3 trials, best sharpe < 0) and duplicate specs already in the backlog.

Return ONLY this exact JSON schema (no markdown, no commentary):
{_PROPOSAL_JSON_SCHEMA}"""

    return [
        {"role": "system", "content": "You output ONLY valid JSON. No markdown, no commentary."},
        {"role": "user", "content": prompt},
    ]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(messages, max_tokens=4096):
    """Call the OpenAI-compatible chat completions endpoint.

    Returns the parsed JSON content. Raises on HTTP/parse errors.
    Hook point for test mocking — tests can patch this function.
    """
    cfg = _get_llm_config()
    body = _post_json(
        f"{cfg['base_url'].rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        payload={
            "model": cfg["model"],
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    content = body["choices"][0]["message"]["content"].strip()
    # Strip markdown fences
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3].strip()
    return json.loads(content)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_param_spec(spec, templates):
    """Validate a type=param spec against STRATEGY_TEMPLATES.

    Returns (True, None) or (False, error_string).
    """
    strategy = spec.get("strategy")
    symbol = spec.get("symbol")
    params = spec.get("params")

    if strategy not in templates:
        return False, f"Unknown strategy '{strategy}'"

    if not _SYMBOL_RE_COMPILED.match(symbol or ""):
        return False, f"Invalid symbol '{symbol}'"

    if not isinstance(params, dict):
        return False, f"params must be dict, got {type(params).__name__}"

    param_space = templates[strategy]["param_space"]
    expected = set(param_space.keys())
    actual = set(params.keys())
    if actual != expected:
        return False, f"Param key mismatch: expected {expected}, got {actual}"

    for pname, pval in params.items():
        pspace = param_space[pname]
        if isinstance(pspace, range):
            if not isinstance(pval, int) or pval not in pspace:
                return False, f"'{pname}'={pval} out of range {pspace.start}..{pspace.stop - 1}"
        elif isinstance(pspace, list):
            if pval not in pspace:
                return False, f"'{pname}'={pval} not in {pspace}"

    return True, None


def _validate_code_spec(spec):
    """Validate a type=code spec.

    Returns (True, None) or (False, error_string).
    """
    name = spec.get("name", "")
    code = spec.get("code", "")
    symbol = spec.get("symbol", "")

    if not name or not isinstance(name, str):
        return False, "Missing or invalid 'name'"

    if not _SYMBOL_RE_COMPILED.match(symbol or ""):
        return False, f"Invalid symbol '{symbol}'"

    if not code or not isinstance(code, str):
        return False, "Missing or empty 'code'"

    if "def generate_signals" not in code:
        return False, "Code must contain 'def generate_signals'"

    if len(code.encode("utf-8")) > 64 * 1024:
        return False, f"Code size {len(code.encode('utf-8'))} exceeds 64KB"

    return True, None


def _validate_proposal(proposal, templates):
    """Validate a single proposal. Returns (True, None) or (False, reason)."""
    ptype = proposal.get("type")
    spec = proposal.get("spec", {})

    if ptype not in ("param", "code"):
        return False, f"Unknown type '{ptype}'"

    # Validate eval_plan: extra_criteria must be a list of strings
    eval_plan = proposal.get("eval_plan", {})
    extra = eval_plan.get("extra_criteria", [])
    if not isinstance(extra, list):
        return False, "extra_criteria must be a list"
    for item in extra:
        if not isinstance(item, str):
            return False, f"extra_criteria item must be str, got {type(item).__name__}"

    # Validate source
    source = proposal.get("source", {})
    kind = source.get("kind")
    if kind not in ("paper", "failure_response", "idea"):
        return False, f"Invalid source.kind '{kind}'"
    if not source.get("ref"):
        return False, "source.ref is required"

    # Validate priority
    priority = proposal.get("priority")
    if not isinstance(priority, (int, float)) or not (0 <= priority <= 1):
        return False, f"priority must be float 0-1, got {priority}"

    # Type-specific validation
    if ptype == "param":
        return _validate_param_spec(spec, templates)
    else:
        return _validate_code_spec(spec)


# ---------------------------------------------------------------------------
# Preflight validation via sandbox runner
# ---------------------------------------------------------------------------

_MAX_PREFLIGHT_FIX_ATTEMPTS = 2


def _preflight_validate(code_str):
    """POST code to sandbox runner /validate endpoint. Returns (valid, error_msg, traceback).

    If SANDBOX_RUNNER_URL is unset or HTTP error → returns (None, "skipped", "") to signal skip.
    """
    runner_url = os.environ.get("SANDBOX_RUNNER_URL", "").rstrip("/")
    if not runner_url:
        return None, "skipped", ""

    import base64 as _b64
    code_b64 = _b64.b64encode(code_str.encode("utf-8")).decode("ascii")
    payload = json.dumps({"code_b64": code_b64}).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{runner_url}/validate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        # HTTP error or network failure → skip preflight, don't block
        print(f"  ⚠️  Preflight skipped: runner unreachable ({e})")
        return None, "skipped", ""

    valid = body.get("valid", False)
    error = body.get("error", "")
    traceback = body.get("traceback", "")
    return valid, error, traceback


def _preflight_fix_attempt(code_str, error_msg, traceback_str, spec_context):
    """Send error back to LLM asking for corrected code. Returns fixed code or None."""
    fix_prompt = f"""Your previous code for strategy "{spec_context}" failed preflight validation.

ERROR: {error_msg}

TRACEBACK (last lines):
{traceback_str}

ORIGINAL CODE:
```
{code_str}
```

Please provide ONLY the corrected `def generate_signals(df):` function (with imports).
Remember the INTERFACE CONTRACT:
- df has DatetimeIndex, columns Open/High/Low/Close/Volume (capitalized), NO 'Date' column
- Return pd.Series aligned to df.index with values in {{-1, 0, 1}}
- Only numpy and pandas imports allowed
"""
    messages = [
        {"role": "system", "content": "You output ONLY corrected Python code. No markdown fences, no commentary."},
        {"role": "user", "content": fix_prompt},
    ]

    try:
        response = _call_llm(messages, max_tokens=2048)
        # LLM may return JSON or raw text; try to extract code
        if isinstance(response, dict):
            code = response.get("code", response.get("corrected_code", ""))
        else:
            code = str(response)
        # Strip markdown fences if present
        code = code.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[-1]
            if code.endswith("```"):
                code = code[:-3].strip()
        if "def generate_signals" not in code:
            return None
        return code
    except Exception as e:
        print(f"  ⚠️  Preflight fix LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(max_proposals=5, dry_run=False):
    """Execute the full ideation pipeline. Returns list of added entry IDs."""
    from backlog import Backlog

    # ── (a) Gather context ──
    knowledge = _load_knowledge(_get_knowledge_path())
    research_docs = _gather_research_docs(_get_research_dirs())
    backlog = Backlog()
    backlog_summary = _summarise_backlog(backlog)

    print(f"📚 Research docs loaded: {len(research_docs)}")
    for fname, _ in research_docs:
        print(f"   - {fname}")

    # ── (b) LLM call ──
    messages = _build_prompt(knowledge, STRATEGY_TEMPLATES, research_docs, backlog_summary, max_proposals)
    print(f"🧠 Calling LLM ({_get_llm_config()['model']}) for up to {max_proposals} proposals...")

    try:
        response = _call_llm(messages, max_tokens=4096)
    except Exception as e:
        print(f"❌ LLM call failed: {e}", file=sys.stderr)
        return []

    proposals = response.get("proposals", [])
    if not isinstance(proposals, list):
        print("❌ LLM returned no 'proposals' array", file=sys.stderr)
        return []

    print(f"📥 LLM returned {len(proposals)} raw proposals")

    # ── (c) Validate + Preflight ──
    accepted = []
    # Preflight counters (code-type proposals only)
    pf_passed = 0
    pf_fixed = 0
    pf_dropped = 0
    pf_skipped = 0

    for i, p in enumerate(proposals):
        ok, reason = _validate_proposal(p, STRATEGY_TEMPLATES)
        if not ok:
            print(f"  ⚠️  Proposal {i+1} dropped: {reason}")
            continue

        ptype = p["type"]

        # ── Preflight for code-type proposals ──
        if ptype == "code":
            code = p["spec"].get("code", "")
            spec_context = f"{p['spec'].get('name', '?')}/{p['spec'].get('symbol', '?')}"
            valid, error_msg, tb_str = _preflight_validate(code)

            if valid is None:
                # Runner unreachable — skip preflight, don't block
                pf_skipped += 1
                print(f"  ⏭️  Proposal {i+1} ({spec_context}): preflight skipped (runner unavailable)")
            elif valid:
                pf_passed += 1
                print(f"  ✅ Preflight passed for Proposal {i+1} ({spec_context})")
            else:
                # Failed — attempt fix retries
                fixed = False
                for attempt in range(_MAX_PREFLIGHT_FIX_ATTEMPTS):
                    print(f"  🔄 Proposal {i+1} ({spec_context}): preflight failed, fix attempt {attempt+1}/{_MAX_PREFLIGHT_FIX_ATTEMPTS}")
                    fixed_code = _preflight_fix_attempt(code, error_msg, tb_str, spec_context)
                    if fixed_code is None:
                        print(f"  ⚠️  Fix attempt {attempt+1}: LLM did not return valid code")
                        continue
                    # Re-validate fixed code
                    valid2, error_msg2, tb_str2 = _preflight_validate(fixed_code)
                    if valid2 is None:
                        pf_skipped += 1
                        # Runner went down mid-fix; accept with warning
                        p["spec"]["code"] = fixed_code
                        fixed = True
                        pf_fixed += 1
                        print(f"  ⏭️  Proposal {i+1}: runner unavailable on re-validate, accepting with warning")
                        break
                    elif valid2:
                        p["spec"]["code"] = fixed_code
                        fixed = True
                        pf_fixed += 1
                        print(f"  ✅ Proposal {i+1} ({spec_context}): fixed on attempt {attempt+1}")
                        break
                    else:
                        error_msg = error_msg2
                        tb_str = tb_str2
                        print(f"  ❌ Fix attempt {attempt+1} still failing: {error_msg}")

                if not fixed:
                    pf_dropped += 1
                    print(f"  🚫 Proposal {i+1} ({spec_context}): DROPPED — preflight failed after {_MAX_PREFLIGHT_FIX_ATTEMPTS} fix attempts")
                    print(f"     Last error: {error_msg}")
                    continue

        entry = {
            "id": p.get("id", ""),  # will be replaced on add
            "type": ptype,
            "status": "pending",
            "priority": float(p["priority"]),
            "created_at": None,  # set by backlog.add_entry
            "source": p["source"],
            "spec": p["spec"],
            "eval_plan": p.get("eval_plan", {"extra_criteria": []}),
            "result": None,
        }

        if dry_run:
            entry["id"] = f"dry_{i}"
            accepted.append(entry)
            spec = entry["spec"]
            if ptype == "param":
                desc = f"{spec['strategy']}/{spec['symbol']} params={json.dumps(spec['params'])}"
            else:
                desc = f"{spec.get('name','?')}/{spec.get('symbol','?')}"
            print(f"  [DRY-RUN] {ptype} | {desc} | priority={entry['priority']:.2f} | src={entry['source']['ref']}")
        else:
            ok_add, result_id = backlog.add_entry(entry)
            if ok_add:
                accepted.append(entry)
                entry["id"] = result_id
                spec = entry["spec"]
                if ptype == "param":
                    desc = f"{spec['strategy']}/{spec['symbol']} params={json.dumps(spec['params'])}"
                else:
                    desc = f"{spec.get('name','?')}/{spec.get('symbol','?')}"
                print(f"  ✅ {ptype} | {desc} | priority={entry['priority']:.2f} | src={entry['source']['ref']}")
            else:
                print(f"  ⚠️  Duplicate (spec matches entry {result_id}), skipped")

    print(f"\n📊 Accepted: {len(accepted)}/{len(proposals)} proposals")

    # Preflight summary line (machine-greppable)
    n_code_proposals = sum(1 for p in proposals if p.get("type") == "code")
    if n_code_proposals > 0:
        print(f"PREFLIGHT passed={pf_passed} fixed={pf_fixed} dropped={pf_dropped} skipped={pf_skipped}")

    if dry_run:
        print("🔍 DRY-RUN mode — no writes performed")
    return [e["id"] for e in accepted]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Strategy Ideation — LLM-driven proposal generation"
    )
    parser.add_argument(
        "--max-proposals", type=int, default=_MAX_PROPOSALS_DEFAULT,
        help=f"Max proposals to request (default: {_MAX_PROPOSALS_DEFAULT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print but do not write to backlog"
    )
    args = parser.parse_args()
    run(max_proposals=args.max_proposals, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
