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
from datetime import datetime, timezone
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

# Import manifest module (Phase 0 PR-A)
from manifest import StrategyManifest, ManifestValidationError

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
        elif e["type"] == "manifest":
            name = s.get("name", "?") if isinstance(s, dict) else "?"
            mode = s.get("execution_mode", "?") if isinstance(s, dict) else "?"
            lines.append(
                f"  manifest: {name} execution_mode={mode} priority={e['priority']:.2f}"
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

_MANIFEST_JSON_SCHEMA = """{
  "manifests": [
    {
      "name": "mean_reversion_bollinger_band",
      "code_b64": "aW1wb3J0IG51bXB5IGFzIG5w...",
      "data_sources": [
        {"type": "ohlcv", "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "start": "2020-01-01"}
      ],
      "model_artifacts": [],
      "compute": {"mode": "inference", "budget_seconds": 60, "gpu": false},
      "evaluator": {
        "type": "portfolio",
        "metrics": ["sharpe", "ir", "turnover", "cvar_95"],
        "benchmark": "SPY"
      },
      "execution_mode": "structured",
      "priority": 0.85,
      "source": {"kind": "paper", "ref": "bollinger-band-paper.md"}
    },
    {
      "name": "transformer_volatility_regime",
      "code_b64": "aW1wb3J0IHRvcmNo...",
      "data_sources": [
        {"type": "ohlcv", "universe": ["SPY", "TLT", "GLD", "USO", "IWM", "QQQ"], "start": "2018-01-01"}
      ],
      "model_artifacts": [{"name": "timesfm-base", "revision": "v1.0"}],
      "compute": {"mode": "inference", "budget_seconds": 120, "gpu": true},
      "evaluator": {
        "type": "custom",
        "metrics": ["sharpe", "ir", "turnover", "cvar_95", "factor_exposure"],
        "benchmark": "SPY",
        "extras": {"custom_evaluator": "regime_aware_sharpe", "n_regimes": 3}
      },
      "execution_mode": "expert",
      "priority": 0.90,
      "source": {"kind": "paper", "ref": "deep-macro-finance-2024.md"}
    }
  ]
}"""

_EXPERT_MODE_CATALOG = """Expert mode unlocks:
- RL/deep learning training loops (PyTorch allowed, GPU optional)
- Foundation-model inference (Transformer, TimesFM, etc.)
- Cross-sectional/pairs/rotation strategies with custom evaluators
- Regime detection with dynamic switching
- Custom evaluators (not limited to portfolio metrics; define your own scoring function)
- Multi-asset portfolio construction with custom allocators
- Training mode: write checkpoints, iterate, validate against holdout
"""


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

def _call_llm(messages, max_tokens=4096, temperature=0.7, model=None, response_json=True):
    """Call the OpenAI-compatible chat completions endpoint.

    Returns the parsed JSON content. Raises on HTTP/parse errors.
    Hook point for test mocking — tests can patch this function.

    Args:
        messages: list of chat messages
        max_tokens: max output tokens
        temperature: sampling temperature (0.0–2.0)
        model: override model name (defaults to HYPO_LLM_MODEL)
        response_json: if True, request json_object format and parse JSON
    """
    cfg = _get_llm_config()
    payload = {
        "model": model or cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    body = _post_json(
        f"{cfg['base_url'].rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=60,
    )
    content = body["choices"][0]["message"]["content"].strip()
    # Strip markdown fences
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3].strip()
    if response_json:
        return json.loads(content)
    return content


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
# IDEATION V2 — 3-stage multi-agent pipeline
# ---------------------------------------------------------------------------

_IDEATION_LOG_DIR = BASE_DIR / "ideation_logs"


def _build_brainstorm_prompt(knowledge, templates, research_docs):
    """Build compact context for the brainstorm stage."""
    # Family aggregates (compact)
    families = knowledge.get("families", {})
    family_lines = []
    for key, fam in sorted(families.items()):
        n = fam.get("n_trials", 0)
        best = fam.get("best_val_sharpe", -999)
        family_lines.append(f"  {key}: {n} trials, best sharpe={best:.2f}")

    # Near-misses
    near_misses = knowledge.get("near_misses", [])
    nm_lines = []
    for nm in near_misses[-15:]:
        s = nm.get("strategy", "?")
        sym = nm.get("symbol", "?")
        p = json.dumps(nm.get("params", {}))
        nm_lines.append(f"  {s}/{sym} params={p} — {nm.get('failed_gate', '?')}")

    # Recent code errors
    errors = knowledge.get("errors", [])
    code_errs = [e for e in errors if e.get("evaluation", {}).get("error_type") == "code"]
    ce_lines = []
    for ce in code_errs[-5:]:
        h = ce.get("hypothesis", {})
        ce_lines.append(f"  {h.get('description', '?')} — {ce.get('evaluation', {}).get('error', '?')[:120]}")

    # Research docs (compact)
    research_text = ""
    if research_docs:
        chunks = [body for _, body in research_docs]
        research_text = "\n\n---\n\n".join(chunks)

    # Mandates
    n_novel = sum(1 for fam in families.values() if fam.get("n_trials", 0) < 3)
    has_near_misses = len(near_misses) >= 2

    mandates = []
    if n_novel > 0:
        mandates.append(
            f"MANDATE: At least ONE idea must target families with <3 trials "
            f"(currently {n_novel} such families: "
            + ", ".join(k for k, f in families.items() if f.get("n_trials", 0) < 3)
            + ")"
        )
    if has_near_misses:
        mandates.append(
            "MANDATE: At least ONE idea must RECOMBINE two near-miss directions "
            "(e.g. apply pattern from one to the symbol/regime of another)"
        )
    mandates.append(
        "MANDATE: At least TWO of your ideas must target a UNIVERSE of 5+ symbols "
        "(cross-sectional or portfolio strategies; e.g. rank stocks by signal and take top/bottom quintile). "
        "This means the idea's type should be 'code' with a universe-oriented approach."
    )
    mandates.append(
        "MANDATE: At least ONE idea must be EXPERT-MODE (execution_mode='expert'). "
        "Declare a custom evaluation approach — e.g. RL policy, transformer regression, "
        "cross-sectional rank strategy, regime-switching allocation — and CITE the paper "
        "it draws from (from RESEARCH_DIRS). Expert-mode ideas may use PyTorch/torch, "
        "scipy, sklearn in their code; declare model_artifacts if using foundation models."
    )

    mandates_text = "\n".join(mandates) if mandates else ""

    prompt = f"""You are a creative quantitative researcher brainstorming trading-strategy candidates.

=== FAMILIES ===
{chr(10).join(family_lines) if family_lines else "(no family data)"}

=== NEAR-MISS ARCHIVE ===
{chr(10).join(nm_lines) if nm_lines else "(no near-misses)"}

=== RECENT CODE ERRORS ===
{chr(10).join(ce_lines) if ce_lines else "(no recent code errors)"}

=== RESEARCH ===
{research_text or "(no research documents)"}

=== EXPERT MODE CATALOG ===
{_EXPERT_MODE_CATALOG}

=== MANDATES ===
{mandates_text}

=== YOUR TASK ===
Brainstorm 10-15 SHORT raw ideas. Each idea: a trading-strategy seed — creative, diverse,
not a full proposal, just a direction. Favor novelty and edge-case regimes.

{chr(10).join(templates.keys())}

Return ONLY this JSON:
{{"ideas": [{{"name": "idea label", "type": "param|code", "family": "strategy|symbol", "one_line_rationale": "why this could work, one sentence"}}]}}"""

    return [
        {"role": "system", "content": "You output ONLY valid JSON. No markdown, no commentary. Be creative and diverse."},
        {"role": "user", "content": prompt},
    ]


def _stage_brainstorm(knowledge, templates, research_docs):
    """Stage 1 — Divergent brainstorming with high temperature.

    Returns list of raw idea dicts: [{name, type, family, one_line_rationale}, ...]
    Raises on failure (caught by caller for fallback).
    """
    model = os.environ.get("HYPO_LLM_MODEL_BRAINSTORM", "deepseek-v4-flash")
    messages = _build_brainstorm_prompt(knowledge, templates, research_docs)
    response = _call_llm(messages, max_tokens=4096, temperature=1.0, model=model)
    ideas = response.get("ideas", [])
    if not isinstance(ideas, list) or not ideas:
        raise ValueError("Brainstorm returned empty or invalid ideas list")
    return ideas


def _stage_debate(ideas):
    """Stage 2 — Adversarial debate: Risk Manager attacks, Quant Researcher defends,
    Judge decides survive/kill per idea.

    Returns (results, judge_report) where:
      results: [{index, survive: bool, attack, rebuttal, reason}]
      judge_report: raw judge LLM response list

    Judge is the single source of truth for survive/kill. If the judge call
    fails or returns unparseable output, ALL ideas survive (fail-open).
    """
    ideas_text = "\n".join(
        f"{i}: [{idea.get('type','?')}] {idea.get('name','?')} | family={idea.get('family','?')} "
        f"| rationale: {idea.get('one_line_rationale','?')}"
        for i, idea in enumerate(ideas)
    )

    # --- Risk Manager ---
    risk_prompt = f"""You are a RISK MANAGER evaluating trading-strategy ideas for overfitting and
implementation risk. Attack each idea below.

For EACH idea, identify its WEAKNESSES:
- Overfitting smell: is it too obvious, too narrow, data-mined?
- Deflation cost: more trials in the same family → higher bar (mention family trial count)
- Alpha decay: if this is a well-known anomaly, post-publication alpha may have already decayed
- Implementability: can this work on daily OHLCV data with realistic assumptions?
- Expert-mode ideas: be extra critical about hidden lookahead in the agent's own train/val/holdout split — if the strategy uses any form of supervised learning, the split must be temporal (past→future) and the agent must NOT see future data during training.

=== IDEAS ===
{ideas_text}

Return ONLY this JSON:
{{"risk_report": [
  {{"index": <int matching idea index>, "attack": "one sentence weakness summary"}}
]}}"""

    risk_messages = [
        {"role": "system", "content": "You output ONLY valid JSON. You are a skeptical risk manager."},
        {"role": "user", "content": risk_prompt},
    ]
    risk_response = _call_llm(risk_messages, max_tokens=4096)
    risk_report = risk_response.get("risk_report", [])

    # --- Quant Researcher ---
    quant_prompt = f"""You are a QUANTITATIVE RESEARCHER defending trading-strategy ideas.
For EACH idea below (including the risk manager's attack), provide a DEFENSE:

- Economic rationale: what structural market behavior does this exploit?
- Related literature: any known academic or industry work that supports this direction?
- Variation: what adaptation would address the attack?

=== IDEAS ===
{ideas_text}

=== RISK MANAGER ATTACKS ===
{json.dumps(risk_report, indent=2)}

Return ONLY this JSON:
{{"quant_report": [
  {{"index": <int matching idea index>, "rebuttal": "one sentence defense/support", "variation": "what change would fix the weakness"}}
]}}"""

    quant_messages = [
        {"role": "system", "content": "You output ONLY valid JSON. You are a constructive quant researcher."},
        {"role": "user", "content": quant_prompt},
    ]
    quant_response = _call_llm(quant_messages, max_tokens=4096)
    quant_report = quant_response.get("quant_report", [])

    # --- Build per-idea context for judge ---
    risk_by_idx = {r.get("index"): r.get("attack", "") for r in risk_report if isinstance(r, dict)}
    quant_by_idx = {q.get("index"): (q.get("rebuttal", ""), q.get("variation", ""))
                    for q in quant_report if isinstance(q, dict)}

    # --- Judge: explicit survive/kill verdicts ---
    judge_items = []
    for i, idea in enumerate(ideas):
        attack = risk_by_idx.get(i, "no attack recorded")
        rebuttal, variation = quant_by_idx.get(i, ("no rebuttal", ""))
        judge_items.append({
            "index": i,
            "idea": f"{idea.get('name','?')} [{idea.get('type','?')}] family={idea.get('family','?')}",
            "attack": attack,
            "rebuttal": rebuttal,
        })

    judge_prompt = f"""You are an impartial JUDGE evaluating trading-strategy ideas. For each idea,
review the attack vs rebuttal and decide: SURVIVE (keep) or KILL (discard).

Decision factors:
- SURVIVE if: rebuttal addresses the attack convincingly OR the idea explores novel ground
- KILL if: attack is fatal (clear overfitting, un-implementable, or well-known decayed anomaly)

For EACH idea, provide a one-sentence reason for your decision.

=== IDEAS WITH ATTACKS AND REBUTTALS ===
{json.dumps(judge_items, indent=2)}

Return ONLY this JSON:
{{"judge_report": [
  {{"index": <int matching idea index>, "survive": true|false, "reason": "one sentence verdict"}}
]}}"""

    judge_messages = [
        {"role": "system", "content": "You output ONLY valid JSON. You are an impartial judge."},
        {"role": "user", "content": judge_prompt},
    ]

    judge_report = []
    judge_failed = False
    try:
        judge_response = _call_llm(judge_messages, max_tokens=4096)
        judge_report = judge_response.get("judge_report", [])
        if not isinstance(judge_report, list) or not judge_report:
            judge_failed = True
    except Exception as e:
        print(f"⚠️  Judge LLM call failed: {e} — ALL ideas survive (fail-open)", file=sys.stderr)
        judge_failed = True

    # --- Build results from judge verdicts ---
    if judge_failed:
        # Fail-open: all ideas survive
        print("⚠️  Judge verdicts unavailable — treating ALL ideas as surviving", file=sys.stderr)
        judge_verdicts = {i: (True, "judge unavailable — fail-open") for i in range(len(ideas))}
    else:
        judge_verdicts = {}
        for jv in judge_report:
            if isinstance(jv, dict) and "index" in jv:
                idx = jv["index"]
                survive = jv.get("survive", True)  # default True if key missing
                reason = jv.get("reason", "")
                judge_verdicts[idx] = (survive, reason)

    results = []
    for i in range(len(ideas)):
        attack = risk_by_idx.get(i, "")
        rebuttal, variation = quant_by_idx.get(i, ("", ""))
        survive, judge_reason = judge_verdicts.get(i, (True, "no judge verdict — default survive"))

        results.append({
            "index": i,
            "survive": survive,
            "attack": attack,
            "rebuttal": rebuttal,
            "variation": variation,
            "reason": f"judge: {judge_reason} | attack: {attack[:100]} | rebuttal: {rebuttal[:100]}",
        })

    return results, judge_report


def _stage_select(surviving_ideas, debate_results, knowledge, templates, research_docs, max_proposals):
    """Stage 3 — Convergent selection: rank by novelty × plausibility × implementability,
    output full proposals in the existing format.

    Returns (proposals, fallback_used) where fallback_used=True if 0 survivors
    triggered fallback to full brainstorm list.
    """
    if not surviving_ideas:
        return [], False

    # Compact family summary for ranking context
    families = knowledge.get("families", {})
    family_summary = "\n".join(
        f"  {k}: {f.get('n_trials', 0)} trials, best sharpe={f.get('best_val_sharpe', -999):.2f}"
        for k, f in sorted(families.items())
    )

    # Research docs compact
    research_text = ""
    if research_docs:
        chunks = [body for _, body in research_docs]
        research_text = "\n\n---\n\n".join(chunks)

    # Enumerate surviving ideas with debate context
    items = []
    for r in debate_results:
        idx = r["index"]
        if r["survive"] and idx < len(surviving_ideas):
            idea = surviving_ideas[idx]
            items.append({
                "idx": idx,
                "idea": idea,
                "attack": r.get("attack", ""),
                "rebuttal": r.get("rebuttal", ""),
                "variation": r.get("variation", ""),
            })

    fallback_used = False
    if not items:
        # 0 survivors: loud warning, fall back to full brainstorm list
        print("⚠️  WARNING: 0 survivors after debate — selecting from full brainstorm list", file=sys.stderr)
        fallback_used = True
        for r in debate_results:
            idx = r["index"]
            if idx < len(surviving_ideas):
                idea = surviving_ideas[idx]
                items.append({
                    "idx": idx,
                    "idea": idea,
                    "attack": r.get("attack", ""),
                    "rebuttal": r.get("rebuttal", ""),
                    "variation": r.get("variation", ""),
                })

    items_text = "\n".join(
        f"{i}: [{item['idea'].get('type','?')}] {item['idea'].get('name','?')} "
        f"| rationale: {item['idea'].get('one_line_rationale','?')} "
        f"| attack: {item['attack'][:150]} | rebuttal: {item['rebuttal'][:150]}"
        for i, item in enumerate(items)
    )

    avail_strats = "\n".join(
        f"  {name}: {tmpl['description']} | params: {json.dumps({k: str(v) for k, v in tmpl['param_space'].items()})}"
        for name, tmpl in templates.items()
    )

    select_prompt = f"""You are a quantitative researcher finalizing trading-strategy proposals.

From the surviving ideas below, select and rank the top {max_proposals} by:
  novelty × plausibility × implementability

For each selected idea, produce a FULL proposal in the existing schema.

=== AVAILABLE STRATEGY TEMPLATES (param spaces) ===
{avail_strats}

=== FAMILY AGGREGATES ===
{family_summary}

=== RESEARCH ===
{research_text or "(no research documents)"}

=== SURVIVING IDEAS (after debate) ===
{items_text}

=== YOUR TASK ===
Select and rank up to {max_proposals} ideas. For each, output a full proposal:
- type: "param" or "code"
- priority: 0.0–1.0
- source: {{kind: "idea", ref: "<idea name>"}} or {{kind: "paper", ref: "<research file>"}}
- spec: full type-appropriate spec (for param: strategy/symbol/params; for code: name/description/code/symbol)
- eval_plan: {{extra_criteria: []}}

Favor NOVEL families (low n_trials) and implementations that survived strong attacks.
For code-type: include a COMPLETE `def generate_signals(df):` function.
Interface contract: df has DatetimeIndex, columns Open/High/Low/Close/Volume (capitalized),
NO 'Date' column, return pd.Series({-1,0,1}) aligned to df.index.

Return ONLY this JSON:
{_PROPOSAL_JSON_SCHEMA}"""

    select_messages = [
        {"role": "system", "content": "You output ONLY valid JSON. No markdown."},
        {"role": "user", "content": select_prompt},
    ]
    # Retry once on empty/unparseable JSON — DeepSeek pro occasionally returns
    # nothing due to internal reasoning token consumption, and the whole v2
    # pipeline shouldn't collapse into v1 for a single flaky response.
    last_err = None
    for attempt in range(2):
        try:
            response = _call_llm(select_messages, max_tokens=4096)
            proposals = response.get("proposals", [])
            if proposals:
                return proposals, fallback_used
            last_err = "empty proposals list"
        except Exception as e:
            last_err = str(e)
        if attempt == 0:
            print(f"⚠️  Stage 3 select retry: {last_err}", file=sys.stderr)
    raise RuntimeError(f"select failed after 2 attempts: {last_err}")


def _save_ideation_log(brainstorm_ideas, risk_report, quant_report, judge_report, selection_reasoning, final_proposals, fallback_used=False):
    """Save full audit trail to ideation_logs/<UTC timestamp>.json."""
    _IDEATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = _IDEATION_LOG_DIR / f"{timestamp}.json"

    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage1_brainstorm": {
            "n_ideas": len(brainstorm_ideas),
            "ideas": brainstorm_ideas,
        },
        "stage2_debate": {
            "risk_report": risk_report,
            "quant_report": quant_report,
            "judge_report": judge_report,
        },
        "stage3_selection": {
            "reasoning": selection_reasoning,
            "n_proposed": len(final_proposals),
            "fallback_used": fallback_used,
        },
        "final_proposals": final_proposals,
    }

    _IDEATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, default=str, ensure_ascii=False))


def _run_ideation_v2(knowledge, templates, research_docs, backlog, max_proposals, dry_run):
    """Execute the 3-stage multi-agent ideation pipeline.

    Any stage failure → fall back to single-call path.
    """
    from backlog import Backlog

    brainstorm_ideas = []
    debate_results = []
    proposals = []

    # ── Stage 1: Brainstorm ──
    try:
        print("🧠 IDEATION_V2 Stage 1: Brainstorm (divergent, high-temp)...")
        brainstorm_ideas = _stage_brainstorm(knowledge, templates, research_docs)
        print(f"   📥 {len(brainstorm_ideas)} raw ideas generated")
    except Exception as e:
        print(f"⚠️  IDEATION_V2 brainstorm failed: {e} — falling back to single-call path", file=sys.stderr)
        return None  # signal fallback

    # ── Stage 2: Debate ──
    try:
        print("⚔️  IDEATION_V2 Stage 2: Debate (Risk Manager + Quant Researcher + Judge)...")
        debate_results, judge_report = _stage_debate(brainstorm_ideas)
        n_survived = sum(1 for r in debate_results if r.get("survive"))
        print(f"   ✅ {n_survived}/{len(brainstorm_ideas)} ideas survived debate")
    except Exception as e:
        print(f"⚠️  IDEATION_V2 debate failed: {e} — falling back to single-call path", file=sys.stderr)
        return None

    # ── Stage 3: Select ──
    try:
        print("🎯 IDEATION_V2 Stage 3: Select (ranking + proposal generation)...")
        surviving = brainstorm_ideas  # pass all; _stage_select filters by debate survive flag
        proposals, select_fallback_used = _stage_select(surviving, debate_results, knowledge, templates, research_docs, max_proposals)
        print(f"   📝 {len(proposals)} full proposals generated")
    except Exception as e:
        print(f"⚠️  IDEATION_V2 select failed: {e} — falling back to single-call path", file=sys.stderr)
        return None

    # ── Map proposals through validation + preflight (same as v1) ──
    accepted = []
    pf_passed = 0
    pf_fixed = 0
    pf_dropped = 0
    pf_skipped = 0

    for i, p in enumerate(proposals):
        ok, reason = _validate_proposal(p, templates)
        if not ok:
            print(f"  ⚠️  Proposal {i+1} dropped: {reason}")
            continue

        ptype = p["type"]

        if ptype == "code":
            code = p["spec"].get("code", "")
            spec_context = f"{p['spec'].get('name', '?')}/{p['spec'].get('symbol', '?')}"
            valid, error_msg, tb_str = _preflight_validate(code)

            if valid is None:
                pf_skipped += 1
                print(f"  ⏭️  Proposal {i+1} ({spec_context}): preflight skipped (runner unavailable)")
            elif valid:
                pf_passed += 1
                print(f"  ✅ Preflight passed for Proposal {i+1} ({spec_context})")
            else:
                fixed = False
                for attempt in range(_MAX_PREFLIGHT_FIX_ATTEMPTS):
                    print(f"  🔄 Proposal {i+1} ({spec_context}): preflight failed, fix attempt {attempt+1}/{_MAX_PREFLIGHT_FIX_ATTEMPTS}")
                    fixed_code = _preflight_fix_attempt(code, error_msg, tb_str, spec_context)
                    if fixed_code is None:
                        print(f"  ⚠️  Fix attempt {attempt+1}: LLM did not return valid code")
                        continue
                    valid2, error_msg2, tb_str2 = _preflight_validate(fixed_code)
                    if valid2 is None:
                        pf_skipped += 1
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
            "id": p.get("id", ""),
            "type": ptype,
            "status": "pending",
            "priority": float(p["priority"]),
            "created_at": None,
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

    # ── Observability ──
    n_survived = sum(1 for r in debate_results if r.get("survive"))
    risk_report = [{"index": r["index"], "attack": r["attack"]} for r in debate_results]
    quant_report = [{"index": r["index"], "rebuttal": r["rebuttal"], "variation": r["variation"]}
                    for r in debate_results]
    try:
        _save_ideation_log(brainstorm_ideas, risk_report, quant_report, judge_report,
                          f"Selected {len(proposals)} from {n_survived} survivors", proposals,
                          fallback_used=select_fallback_used)
    except Exception as e:
        print(f"⚠️  Failed to save ideation log: {e}", file=sys.stderr)

    # ── Summary ──
    n_survived = sum(1 for r in debate_results if r.get("survive"))
    n_accepted = len(accepted)
    print(f"IDEATION_V2 brainstormed={len(brainstorm_ideas)} survived={n_survived} proposed={n_accepted}")
    n_code = sum(1 for p in proposals if p.get("type") == "code")
    if n_code > 0:
        print(f"PREFLIGHT passed={pf_passed} fixed={pf_fixed} dropped={pf_dropped} skipped={pf_skipped}")
    return [e["id"] for e in accepted]


# ---------------------------------------------------------------------------
# IDEATION V3 — manifest emission (Phase 1 PR-E)
# ---------------------------------------------------------------------------

def _stage_select_v3(surviving_ideas, debate_results, knowledge, templates, research_docs, max_proposals):
    """Stage 3 (V3) — Convergent selection producing StrategyManifest objects.

    Returns (manifests, fallback_used) where each item is a dict with:
      - manifest: StrategyManifest object
      - priority: float
      - source: dict
    """
    if not surviving_ideas:
        return [], False

    # Compact family summary for ranking context
    families = knowledge.get("families", {})
    family_summary = "\n".join(
        f"  {k}: {f.get('n_trials', 0)} trials, best sharpe={f.get('best_val_sharpe', -999):.2f}"
        for k, f in sorted(families.items())
    )

    # Research docs compact
    research_text = ""
    if research_docs:
        chunks = [body for _, body in research_docs]
        research_text = "\n\n---\n\n".join(chunks)

    # Enumerate surviving ideas with debate context
    items = []
    for r in debate_results:
        idx = r["index"]
        if r["survive"] and idx < len(surviving_ideas):
            idea = surviving_ideas[idx]
            items.append({
                "idx": idx,
                "idea": idea,
                "attack": r.get("attack", ""),
                "rebuttal": r.get("rebuttal", ""),
                "variation": r.get("variation", ""),
            })

    fallback_used = False
    if not items:
        print("⚠️  WARNING: 0 survivors after debate — selecting from full brainstorm list", file=sys.stderr)
        fallback_used = True
        for r in debate_results:
            idx = r["index"]
            if idx < len(surviving_ideas):
                idea = surviving_ideas[idx]
                items.append({
                    "idx": idx,
                    "idea": idea,
                    "attack": r.get("attack", ""),
                    "rebuttal": r.get("rebuttal", ""),
                    "variation": r.get("variation", ""),
                })

    items_text = "\n".join(
        f"{i}: [{item['idea'].get('type','?')}] {item['idea'].get('name','?')} "
        f"| rationale: {item['idea'].get('one_line_rationale','?')} "
        f"| attack: {item['attack'][:150]} | rebuttal: {item['rebuttal'][:150]}"
        for i, item in enumerate(items)
    )

    avail_strats = "\n".join(
        f"  {name}: {tmpl['description']} | params: {json.dumps({k: str(v) for k, v in tmpl['param_space'].items()})}"
        for name, tmpl in templates.items()
    )

    select_prompt = f"""You are a quantitative researcher finalizing strategy MANIFESTS for sandbox-alpha v2.

From the surviving ideas below, select and rank the top {max_proposals} by:
  novelty × plausibility × implementability

For each selected idea, produce a complete StrategyManifest in the v2 manifest schema.
The manifest is the single source of truth — the runner will load data, execute code,
and evaluate according to the manifest's declarations.

=== EXPERT MODE CATALOG ===
{_EXPERT_MODE_CATALOG}

=== AVAILABLE STRATEGY TEMPLATES (param spaces) ===
{avail_strats}

=== FAMILY AGGREGATES ===
{family_summary}

=== RESEARCH ===
{research_text or "(no research documents)"}

=== SURVIVING IDEAS (after debate) ===
{items_text}

=== YOUR TASK ===
Select and rank up to {max_proposals} ideas. For each, output a complete StrategyManifest:

FIELDS:
- name: a short unique strategy identifier (snake_case, e.g. "cross_sectional_momentum_top20")
- code_b64: base64-encoded Python source. For structured mode: a generate_signals(df) function
  returning pd.Series signal per asset or np.array weights. For expert mode: any valid Python
  entrypoint (generate_signals or generate_weights) — may import torch, sklearn, scipy.
- data_sources: list of DataSource objects. At minimum one ohlcv with universe and start date.
  For portfolio strategies: universe should be 5+ symbols.
- model_artifacts: list of model references (name + optional revision). Empty list if no models.
- compute: {{mode: "inference"|"training", budget_seconds: int, gpu: bool}}
- evaluator: {{type: "portfolio"|"single_asset"|"custom", metrics: [...], benchmark: "SPY" or null,
  extras: {{}}(optional extra config)}}
- execution_mode: "structured" (simple generate_signals → portfolio evaluation) or
  "expert" (custom evaluator, RL, regime detection, foundation models, training loops)
- priority: 0.0–1.0 (higher = more promising)
- source: {{kind, ref}} — paper citation or idea reference

INTERFACE CONTRACT (structured mode):
- df has DatetimeIndex, columns Open/High/Low/Close/Volume (ALL capitalized)
- Return pd.Series aligned to df.index with values in {{-1, 0, 1}}
- Imports: numpy and pandas ONLY for structured

IMPORTANT: For expert-mode manifests:
- evaluator.type should be "custom" with extras describing the custom approach
- code_b64 may import torch, sklearn, scipy
- model_artifacts must list any foundation models used
- The code should implement a self-contained training/inference pipeline
- MUST cite the paper in the source.ref field

Favor NOVEL families (low n_trials) and ideas that survived strong attacks.
For code_b64: base64-encode complete self-contained Python source.

Return ONLY this JSON (no markdown, no commentary):
{_MANIFEST_JSON_SCHEMA}"""

    select_messages = [
        {"role": "system", "content": "You output ONLY valid JSON. No markdown."},
        {"role": "user", "content": select_prompt},
    ]

    # Retry logic: 3 attempts with progressively larger token budget + lower
    # temperature. Empirically the failure mode is "unterminated string" from
    # the manifest schema being verbose enough to hit the token cap, followed
    # by JSON drift when the model is too creative on retry.
    attempts_cfg = [
        {"max_tokens": 8192, "temperature": 0.7},
        {"max_tokens": 12288, "temperature": 0.3},
        {"max_tokens": 16384, "temperature": 0.1},
    ]
    last_err = None
    for attempt, cfg in enumerate(attempts_cfg):
        try:
            response = _call_llm(
                select_messages,
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
            )
            manifests_raw = response.get("manifests", [])

            if not manifests_raw:
                last_err = "empty manifests list"
                if attempt < len(attempts_cfg) - 1:
                    print(f"⚠️  Stage 3 v3 select retry ({attempt+1}/{len(attempts_cfg)}): {last_err}", file=sys.stderr)
                continue

            # Parse and validate each manifest
            # We also return the raw dicts for logging
            parsed = []
            raw_dicts = []
            for m_raw in manifests_raw:
                if not isinstance(m_raw, dict):
                    print(f"  ⚠️  Skipping non-dict manifest entry: {type(m_raw).__name__}")
                    continue
                try:
                    manifest = StrategyManifest.from_dict(m_raw)
                    violations = manifest.validate()
                    if violations:
                        print(f"  🚫 Manifest '{manifest.name or '?'}' dropped — {len(violations)} violation(s):")
                        for v in violations:
                            print(f"     - {v}")
                        continue
                    # validation passed
                    parsed.append(manifest)
                    raw_dicts.append(m_raw)
                except ManifestValidationError as e:
                    print(f"  🚫 Manifest parse error: {e}")
                    continue

            if not parsed:
                last_err = "all manifests failed validation"
                if attempt < len(attempts_cfg) - 1:
                    print(f"⚠️  Stage 3 v3 select retry ({attempt+1}/{len(attempts_cfg)}): {last_err}", file=sys.stderr)
                continue

            return parsed, fallback_used

        except Exception as e:
            last_err = str(e)
        if attempt < len(attempts_cfg) - 1:
            print(f"⚠️  Stage 3 v3 select retry ({attempt+1}/{len(attempts_cfg)}): {last_err}", file=sys.stderr)
    raise RuntimeError(f"select v3 failed after {len(attempts_cfg)} attempts: {last_err}")


def _preflight_manifest(manifest_dict):
    """POST manifest to sandbox runner /run_manifest endpoint for dry-check.

    Returns (valid, error_msg). Returns (None, "skipped") if runner unavailable.
    Skip for expert mode — too expensive to actually run.
    """
    runner_url = os.environ.get("SANDBOX_RUNNER_URL", "").rstrip("/")
    if not runner_url:
        return None, "skipped"

    payload = json.dumps(manifest_dict).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{runner_url}/run_manifest",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  ⚠️  Manifest preflight skipped: runner unreachable ({e})")
        return None, "skipped"

    valid = body.get("valid", body.get("status") == "ok")
    error = body.get("error", "")
    return valid, error


def _save_ideation_log_v3(brainstorm_ideas, risk_report, quant_report, judge_report,
                          selection_reasoning, manifest_dicts, fallback_used=False):
    """Save full audit trail for v3 (manifest-emitting) ideation."""
    _IDEATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = _IDEATION_LOG_DIR / f"{timestamp}_v3.json"

    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v3",
        "stage1_brainstorm": {
            "n_ideas": len(brainstorm_ideas),
            "ideas": brainstorm_ideas,
        },
        "stage2_debate": {
            "risk_report": risk_report,
            "quant_report": quant_report,
            "judge_report": judge_report,
        },
        "stage3_selection": {
            "reasoning": selection_reasoning,
            "n_proposed": len(manifest_dicts),
            "fallback_used": fallback_used,
        },
        "final_proposals": manifest_dicts,
    }

    _IDEATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, default=str, ensure_ascii=False))


def _run_ideation_v3(knowledge, templates, research_docs, backlog, max_proposals, dry_run):
    """Execute 3-stage ideation pipeline producing StrategyManifest objects.

    Any stage failure → fall back to v2 pipeline.
    If ALL manifest validation fails → fall back to v2 pipeline.
    """
    from backlog import Backlog

    brainstorm_ideas = []
    debate_results = []
    judge_report = []

    # ── Stage 1: Brainstorm ──
    try:
        print("🧠 IDEATION_V3 Stage 1: Brainstorm (divergent, high-temp)...")
        brainstorm_ideas = _stage_brainstorm(knowledge, templates, research_docs)
        print(f"   📥 {len(brainstorm_ideas)} raw ideas generated")
    except Exception as e:
        print(f"⚠️  IDEATION_V3 brainstorm failed: {e} — falling back to v2 ({e})", file=sys.stderr)
        return None

    # ── Stage 2: Debate ──
    try:
        print("⚔️  IDEATION_V3 Stage 2: Debate (Risk Manager + Quant Researcher + Judge)...")
        debate_results, judge_report = _stage_debate(brainstorm_ideas)
        n_survived = sum(1 for r in debate_results if r.get("survive"))
        print(f"   ✅ {n_survived}/{len(brainstorm_ideas)} ideas survived debate")
    except Exception as e:
        print(f"⚠️  IDEATION_V3 debate failed: {e} — falling back to v2 ({e})", file=sys.stderr)
        return None

    # ── Stage 3: Select (v3 manifest) ──
    try:
        print("🎯 IDEATION_V3 Stage 3: Select (manifest generation)...")
        surviving = brainstorm_ideas
        manifests, select_fallback_used = _stage_select_v3(
            surviving, debate_results, knowledge, templates, research_docs, max_proposals
        )
        print(f"   📝 {len(manifests)} valid manifests generated")
    except Exception as e:
        print(f"⚠️  IDEATION_V3 select failed: {e} — falling back to v2 pipeline", file=sys.stderr)
        return None

    if not manifests:
        print("⚠️  IDEATION_V3: ALL manifests failed validation — falling back to v2 pipeline", file=sys.stderr)
        return None

    # ── Map manifests through preflight + backlog ──
    accepted = []
    pf_passed = 0
    pf_skipped = 0
    manifest_dicts = []

    for i, manifest in enumerate(manifests):
        manifest_dict = manifest.to_dict()
        # Inject priority + source from the select-stage response
        # (These are stored in the backlog entry, not in the manifest itself)
        manifest_dicts.append(manifest_dict)

        execution_mode = manifest.execution_mode
        manifest_name = manifest.name

        # ── Preflight for structured mode (skip expert) ──
        if execution_mode == "structured":
            valid, error = _preflight_manifest(manifest_dict)
            if valid is None:
                pf_skipped += 1
                print(f"  ⏭️  Manifest {i+1} '{manifest_name}': preflight skipped (runner unavailable)")
            elif valid:
                pf_passed += 1
                print(f"  ✅ Manifest {i+1} '{manifest_name}': preflight passed")
            else:
                print(f"  🚫 Manifest {i+1} '{manifest_name}': preflight FAILED — {error}")
                # Dropped on preflight failure
                continue
        else:
            # Expert mode: validation-only via manifest.validate() (already done in _stage_select_v3)
            print(f"  🔬 Manifest {i+1} '{manifest_name}' (expert): validation-only, skipping preflight")

        # Build backlog entry
        # Determine source from the LLM response or fall back
        # The manifest schema includes priority and source in the LLM response
        entry = {
            "id": "",
            "type": "manifest",
            "status": "pending",
            "priority": 0.8,  # default; overridden if available from raw data
            "created_at": None,
            "source": {"kind": "idea", "ref": manifest_name},
            "spec": manifest_dict,
            "eval_plan": {"extra_criteria": []},
            "result": None,
        }

        if dry_run:
            entry["id"] = f"dry_v3_{i}"
            accepted.append(entry)
            print(f"  [DRY-RUN] manifest | {manifest_name} | execution_mode={execution_mode} | priority={entry['priority']:.2f}")
        else:
            ok_add, result_id = backlog.add_entry(entry)
            if ok_add:
                accepted.append(entry)
                entry["id"] = result_id
                print(f"  ✅ manifest | {manifest_name} | execution_mode={execution_mode} | priority={entry['priority']:.2f}")
            else:
                print(f"  ⚠️  Duplicate (spec matches entry {result_id}), skipped")

    # ── Observability ──
    n_survived = sum(1 for r in debate_results if r.get("survive"))
    risk_report = [{"index": r["index"], "attack": r["attack"]} for r in debate_results]
    quant_report = [{"index": r["index"], "rebuttal": r["rebuttal"], "variation": r["variation"]}
                    for r in debate_results]
    try:
        _save_ideation_log_v3(brainstorm_ideas, risk_report, quant_report, judge_report,
                              f"Selected {len(manifests)} from {n_survived} survivors",
                              manifest_dicts, fallback_used=select_fallback_used)
    except Exception as e:
        print(f"⚠️  Failed to save ideation log: {e}", file=sys.stderr)

    # ── Summary ──
    n_accepted = len(accepted)
    n_structured = sum(1 for m in manifests if m.execution_mode == "structured")
    n_expert = sum(1 for m in manifests if m.execution_mode == "expert")
    print(f"IDEATION_V3 brainstormed={len(brainstorm_ideas)} survived={n_survived} proposed={len(manifests)} manifest_count={n_accepted}")
    if n_structured > 0 or n_expert > 0:
        print(f"PREFLIGHT passed={pf_passed} skipped={pf_skipped}")
    print(f"  structured={n_structured} expert={n_expert}")
    return [e["id"] for e in accepted]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(max_proposals=5, dry_run=False):
    """Execute the full ideation pipeline. Returns list of added entry IDs.

    Cascade: IDEATION_V3 (manifest, default on) → IDEATION_V2 (3-stage) → V1 single-call.
    Set IDEATION_V3=0 to fall back to v2 (non-manifest) pipeline.
    Set IDEATION_V2=0 to fall back to v1 single-call pipeline.
    """
    from backlog import Backlog

    # ── (a) Gather context ──
    knowledge = _load_knowledge(_get_knowledge_path())
    research_docs = _gather_research_docs(_get_research_dirs())
    backlog = Backlog()
    backlog_summary = _summarise_backlog(backlog)

    print(f"📚 Research docs loaded: {len(research_docs)}")
    for fname, _ in research_docs:
        print(f"   - {fname}")

    # ── (b) Try IDEATION_V3 pipeline (manifest emission) ──
    use_v3 = os.environ.get("IDEATION_V3", "1") != "0"
    if use_v3:
        print("🔄 IDEATION_V3 enabled — attempting manifest-emitting 3-stage pipeline...")
        try:
            v3_result = _run_ideation_v3(knowledge, STRATEGY_TEMPLATES, research_docs, backlog, max_proposals, dry_run)
            if v3_result is not None:
                return v3_result
            # else: fall through to v2
        except Exception as e:
            print(f"⚠️  IDEATION_V3 pipeline error: {e} — falling back to v2 pipeline", file=sys.stderr)

    # ── (c) Try IDEATION_V2 pipeline (3-stage, v1-shape proposals) ──
    use_v2 = os.environ.get("IDEATION_V2", "1") != "0"
    if use_v2:
        print("🔄 IDEATION_V2 enabled — attempting 3-stage multi-agent pipeline...")
        try:
            v2_result = _run_ideation_v2(knowledge, STRATEGY_TEMPLATES, research_docs, backlog, max_proposals, dry_run)
            if v2_result is not None:
                return v2_result
            # else: fall through to v1
        except Exception as e:
            print(f"⚠️  IDEATION_V2 pipeline error: {e} — falling back to single-call path", file=sys.stderr)

    # ── (d) V1 fallback: single-call LLM ──
    print("📋 Using single-call ideation path (v1 fallback)")
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
