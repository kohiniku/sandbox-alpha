#!/usr/bin/env python3
"""
LLM-driven hypothesis generation for sandbox-alpha.
Calls an OpenAI-compatible chat completions endpoint and strictly validates the response.

Config (all via environment variables):
  HYPO_LLM_BASE_URL     – base URL for chat completions (default: https://api.deepseek.com/v1)
  HYPO_LLM_MODEL        – model name (default: deepseek-v4-pro)
  HYPO_LLM_API_KEY_ENV  – name of env var holding the API key (default: DEEPSEEK_API_KEY)
"""

import json
import os
import re
import time
import random
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_STRATEGY_NAMES = {"sma_crossover", "mean_reversion", "momentum", "rsi"}

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$")


def _get_config():
    return {
        "base_url": os.environ.get("HYPO_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.environ.get("HYPO_LLM_MODEL", "deepseek-v4-pro"),
        "api_key": os.environ.get(
            os.environ.get("HYPO_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY"), ""
        ),
    }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_prompt(knowledge, templates):
    """Build the messages payload for chat completions."""

    # --- Available strategies with param spaces ---
    strategy_lines = []
    for name, tmpl in templates.items():
        desc = tmpl.get("description", name)
        space_desc = _describe_param_space(tmpl["param_space"])
        strategy_lines.append(f"- **{name}**: {desc}\n  param_space: {space_desc}")

    strategies_text = "\n".join(strategy_lines)

    # --- Knowledge summary ---
    knowledge_text = _build_knowledge_summary(knowledge)

    # --- Already tested combos ---
    tested = knowledge.get("tested_combinations", [])
    if tested:
        # Cap at 30 to keep prompt size reasonable
        avoid_combos = tested[-30:]
        avoid_lines = []
        for tc in avoid_combos:
            avoid_lines.append(
                f"  - strategy={tc['strategy']}, symbol={tc['symbol']}, params={json.dumps(tc['params'])}"
            )
        avoid_text = "DO NOT propose any of these already-tested combinations:\n" + "\n".join(avoid_lines)
    else:
        avoid_text = "No combinations have been tested yet."

    system_prompt = (
        "You are a quantitative researcher generating trading strategy hypotheses. "
        "You output ONLY valid JSON — no markdown, no commentary, no code fences."
    )

    user_prompt = f"""Available Strategies with Exact Parameter Spaces:
{strategies_text}

Current Knowledge Base:
{knowledge_text}

Already-Tested Combinations (exact duplicates to avoid):
{avoid_text}

Your task: propose exactly ONE new hypothesis. It must use one of the strategies above,
with parameters strictly within the specified ranges/lists. The symbol must be a valid
ticker like AAPL, MSFT, SPY, BTC-USD, etc.

CRITICAL: Prefer hypotheses that respond to WHY previous ones failed:
- Holdout failures → suspect overfitting; try a different symbol or strategy family.
- Validation/drawdown failures → try lower-risk parameter regions (longer windows, smaller thresholds).
- EXHAUSTED families → do NOT propose params near those already-attempted ranges.

Return ONLY this strict JSON (no other text):
{{"strategy": "<name>", "symbol": "<TICKER>", "params": {{...}}, "rationale": "one sentence explaining why this addresses past failures or explores new ground"}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _describe_param_space(param_space):
    """Describe a param space as a human-readable string."""
    parts = []
    for pname, pvals in param_space.items():
        if isinstance(pvals, range):
            parts.append(f"{pname}=int({pvals.start}..{pvals.stop - 1})")
        elif isinstance(pvals, list):
            parts.append(f"{pname}=one_of{pvals}")
        else:
            parts.append(f"{pname}=?")
    return "{" + ", ".join(parts) + "}"


def _build_knowledge_summary(knowledge):
    """Build a compact summary: families table + last 5 adopted + last 5 rejected with gate reasons."""
    lines = []

    # --- Families table (ALL families, no window) ---
    families = knowledge.get("families", {})
    if families:
        lines.append("Family aggregates (ALL):")
        for fam_key, fam in sorted(families.items()):
            strategy, symbol = fam_key.split("|", 1)
            n = fam.get("n_trials", 0)
            best_sharpe = fam.get("best_val_sharpe", -999)
            gf = fam.get("gate_failures", {})
            v = gf.get("validation", 0)
            d = gf.get("deflation", 0)
            h = gf.get("holdout", 0)
            dc = gf.get("duplicate_cluster", 0)
            ec = gf.get("exhausted_cluster", 0)
            total_fails = v + d + h + dc + ec
            exhausted_marker = ""
            if n >= 3 and best_sharpe < 0:
                exhausted_marker = " EXHAUSTED — do not propose params near previous attempts"
            lines.append(
                f"  {strategy} on {symbol}: {n} trials, best val Sharpe {best_sharpe:.2f}, "
                f"failures: validation={v} deflation={d} holdout={h}{exhausted_marker}"
            )
    else:
        lines.append("Family aggregates: none yet")

    # --- Adopted (last 5) ---
    adopted = knowledge.get("adopted", [])
    if adopted:
        lines.append(f"Adopted ({len(adopted)}, showing last 5):")
        for a in adopted[-5:]:
            hyp = a.get("hypothesis", {})
            ev = a.get("evaluation", {})
            lines.append(
                f"  - {hyp.get('strategy')} on {hyp.get('symbol')} "
                f"params={json.dumps(hyp.get('params', {}))} | "
                f"Val Sharpe={ev.get('sharpe_ratio', '?'):.2f}, "
                f"Holdout Sharpe={ev.get('holdout_sharpe', '?'):.2f}"
            )
    else:
        lines.append("Adopted: none yet")

    # --- Rejected (last 5) WITH gate reasons ---
    rejected = knowledge.get("rejected", [])
    if rejected:
        lines.append(f"Rejected ({len(rejected)}, showing last 5 with failure gates):")
        for r in rejected[-5:]:
            hyp = r.get("hypothesis", {})
            ev = r.get("evaluation", {})
            gate = ev.get("gate_results", {})
            val_sharpe = ev.get("sharpe_ratio", "?")
            holdout_sharpe = ev.get("holdout_sharpe", "?")
            # Build gate-reason string
            gate_parts = []
            if not gate.get("validation", True):
                gate_parts.append(f"validation failed (val Sharpe {val_sharpe})")
            elif not gate.get("holdout", True):
                gate_parts.append(f"holdout failed (val Sharpe {val_sharpe} passed, holdout Sharpe {holdout_sharpe} failed)")
            elif gate.get("cluster") == "duplicate_cluster":
                gate_parts.append(f"duplicate_cluster (holdout Sharpe {holdout_sharpe} ≤ incumbent)")
            elif gate.get("cluster") == "exhausted_cluster":
                gate_parts.append("exhausted_cluster (pre-block skip)")
            else:
                gate_parts.append(f"unknown gate (val Sharpe {val_sharpe})")
            gate_str = ", ".join(gate_parts)
            lines.append(
                f"  - {hyp.get('strategy')} on {hyp.get('symbol')} "
                f"params={json.dumps(hyp.get('params', {}))} | "
                f"rejected at: {gate_str}"
            )
    else:
        lines.append("Rejected: none yet")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_response(data, templates):
    """
    Strict validation of LLM output.
    Returns the validated dict or raises ValueError with a descriptive message.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    required_keys = {"strategy", "symbol", "params", "rationale"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    strategy = data["strategy"]
    symbol = data["symbol"]
    params = data["params"]
    rationale = data.get("rationale", "")

    # Strategy must be in templates
    if strategy not in templates:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Must be one of: {sorted(templates.keys())}"
        )

    # Symbol must match pattern
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(
            f"Invalid symbol '{symbol}'. Must match pattern ^[A-Z0-9][A-Z0-9.\\-]{{0,11}}$"
        )

    # Params must be a dict
    if not isinstance(params, dict):
        raise ValueError(f"Expected params dict, got {type(params).__name__}")

    param_space = templates[strategy]["param_space"]

    # Params keys must exactly match (no extra, no missing)
    expected_keys = set(param_space.keys())
    actual_keys = set(params.keys())
    if actual_keys != expected_keys:
        extra = actual_keys - expected_keys
        missing_keys = expected_keys - actual_keys
        msg = f"Params key mismatch for '{strategy}'."
        if extra:
            msg += f" Unexpected keys: {extra}."
        if missing_keys:
            msg += f" Missing keys: {missing_keys}."
        raise ValueError(msg)

    # Each value must be within the defined range/list
    for pname, pval in params.items():
        pspace = param_space[pname]
        if isinstance(pspace, range):
            if not isinstance(pval, int):
                raise ValueError(
                    f"Param '{pname}' must be int, got {type(pval).__name__} ({pval})"
                )
            if pval not in pspace:
                raise ValueError(
                    f"Param '{pname}' value {pval} out of range {pspace.start}..{pspace.stop - 1}"
                )
        elif isinstance(pspace, list):
            if pval not in pspace:
                raise ValueError(
                    f"Param '{pname}' value {pval} not in allowed list {pspace}"
                )

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(knowledge, templates):
    """
    Call an LLM to generate a trading-strategy hypothesis.

    Args:
        knowledge: the knowledge dict (adopted, rejected, tested_combinations, ...)
        templates: the STRATEGY_TEMPLATES dict

    Returns:
        A hypothesis dict with keys:
          id, strategy, symbol, params, description, generated_at, rationale

    Raises:
        requests.RequestException – network/HTTP errors
        ValueError – invalid or unvalidatable LLM response
        json.JSONDecodeError – malformed JSON in LLM response
    """
    cfg = _get_config()
    messages = _build_prompt(knowledge, templates)

    resp = requests.post(
        f"{cfg['base_url'].rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": cfg["model"],
            "messages": messages,
            "temperature": 0.7,
            # DeepSeek v4系はflashでも隠れreasoningがトークンを消費するため、
            # 512では finish_reason=length で content が空になるケースが多発した
            "max_tokens": int(os.environ.get("HYPO_LLM_MAX_TOKENS", "2048")),
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    resp.raise_for_status()

    body = resp.json()
    content = body["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        # Remove opening fence line
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3].strip()

    parsed = json.loads(content)
    validated = _validate_response(parsed, templates)

    # Build hypothesis dict matching the random generate_hypothesis() shape
    strategy = validated["strategy"]
    symbol = validated["symbol"]
    params = validated["params"]
    rationale = validated["rationale"]

    return {
        "id": f"hyp_{int(time.time())}_{random.randint(1000, 9999)}",
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "description": f"{templates[strategy]['description']} on {symbol}",
        "generated_at": datetime.now().isoformat(),
        "rationale": rationale,
    }
