#!/usr/bin/env python3
"""
Trusted DSL interpreter for strategy logic specs.

Implements a declarative JSON strategy-logic DSL as a fixed-vocabulary
nested-dict node tree — no string parsing, no eval(), no general-purpose
expression language. Every condition is a typed JSON object.

Indicator computations use ONLY trailing/rolling pandas operations
(.rolling, .ewm, .shift(1)) — lookahead is structurally impossible.
"""

from __future__ import annotations

import base64
import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class DslExecutionError(Exception):
    """Raised when a logic_spec cannot be executed (malformed node, missing
    indicator, unknown op, etc.).  Caught by manifest_runner and reported as
    a 'code'-category error, exactly like a code_b64 exception today."""


# ---------------------------------------------------------------------------
# Indicator computation (trailing-only — no future data)
# ---------------------------------------------------------------------------

# ── SMA ──────────────────────────────────────────────────────────────────

def _compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average (trailing window)."""
    return close.rolling(window=window, min_periods=1).mean()


# ── EMA ──────────────────────────────────────────────────────────────────

def _compute_ema(close: pd.Series, window: int) -> pd.Series:
    """Exponential moving average via pandas ewm (span=window, trailing)."""
    return close.ewm(span=window, min_periods=1, adjust=False).mean()


# ── RSI (Wilder smoothing) ──────────────────────────────────────────────

def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """RSI using Wilder's smoothing (reuses logic from _single_name/rsi.py)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ── Bollinger %B ─────────────────────────────────────────────────────────

def _compute_bollinger_pct_b(close: pd.Series, window: int) -> pd.Series:
    """Bollinger %B = (Close - lower) / (upper - lower).

    Uses 2-std bands (standard Bollinger convention).
    """
    sma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    upper = sma + 2.0 * std
    lower = sma - 2.0 * std
    denom = upper - lower
    pct_b = pd.Series(np.nan, index=close.index)
    mask = denom > 0
    pct_b[mask] = (close[mask] - lower[mask]) / denom[mask]
    return pct_b


# ── Rolling Z-score ──────────────────────────────────────────────────────

def _compute_rolling_zscore(close: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (Close - rolling_mean) / rolling_std."""
    sma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std()
    return (close - sma) / std.replace(0, np.nan)


# ── Rolling Volatility ───────────────────────────────────────────────────

def _compute_rolling_vol(close: pd.Series, window: int) -> pd.Series:
    """Annualised rolling volatility of daily log returns."""
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(window=window, min_periods=1).std()
    return vol * np.sqrt(252)


# ── MACD ─────────────────────────────────────────────────────────────────

def _compute_macd(close: pd.Series, window: int) -> pd.Series:
    """MACD = EMA(fast=12) - EMA(slow=window).  Default slow window = 26."""
    fast = close.ewm(span=12, min_periods=1, adjust=False).mean()
    slow = close.ewm(span=window, min_periods=1, adjust=False).mean()
    return fast - slow


# ── dispatcher ───────────────────────────────────────────────────────────

_INDICATOR_FUNCTIONS: Dict[str, Any] = {
    "sma": _compute_sma,
    "ema": _compute_ema,
    "rsi": _compute_rsi,
    "bollinger_pct_b": _compute_bollinger_pct_b,
    "rolling_zscore": _compute_rolling_zscore,
    "rolling_vol": _compute_rolling_vol,
    "macd": _compute_macd,
}

VALID_INDICATOR_TYPES = frozenset(_INDICATOR_FUNCTIONS.keys())


def _compute_indicator(
    df: pd.DataFrame, ind_type: str, window: int
) -> pd.Series:
    """Compute a named indicator on a single-symbol OHLCV DataFrame."""
    fn = _INDICATOR_FUNCTIONS[ind_type]
    return fn(df["Close"], window)


# ---------------------------------------------------------------------------
# Rule evaluation (fixed vocabulary, nested dict nodes, NO string parsing)
# ---------------------------------------------------------------------------

COMPARISON_OPS = frozenset({"gt", "lt", "gte", "lte"})
LOGICAL_OPS = frozenset({"and", "or", "not"})
CROSS_OPS = frozenset({"crosses_above", "crosses_below"})
ALL_OPS = COMPARISON_OPS | LOGICAL_OPS | CROSS_OPS


def _resolve_leaf(
    node: dict, values: Dict[str, pd.Series]
) -> pd.Series:
    """Resolve a leaf node to a numeric Series.

    Leaf is either {"indicator": "<name>"} or {"const": <number>}.
    """
    if "indicator" in node:
        name = node["indicator"]
        s = values.get(name)
        if s is None:
            raise DslExecutionError(
                f"Unknown indicator '{name}'. "
                f"Available: {sorted(values.keys())}"
            )
        return s
    if "const" in node:
        val = node["const"]
        if not isinstance(val, (int, float)):
            raise DslExecutionError(
                f"const value must be numeric, got {type(val).__name__}"
            )
        # Return a constant series matching the shape of the indicator index
        first_series = next(iter(values.values()))
        return pd.Series(val, index=first_series.index)
    raise DslExecutionError(
        "Leaf node must have 'indicator' or 'const' key. "
        f"Got keys: {sorted(node.keys())}"
    )


def _eval_node(
    node: dict, values: Dict[str, pd.Series]
) -> pd.Series:
    """Recursively evaluate a rule node, returning a boolean Series.

    Fixed vocabulary — no free-text expression strings:
      - Comparison: {"op": "gt|lt|gte|lte", "left": ..., "right": ...}
      - Logical:    {"op": "and|or", "children": [...]}
      - Unary:      {"op": "not", "child": ...}
      - Cross:      {"op": "crosses_above|crosses_below", "left": ..., "right": ...}
    """
    if not isinstance(node, dict):
        raise DslExecutionError(
            f"Rule node must be a dict, got {type(node).__name__}"
        )

    op = node.get("op")
    if op is None:
        raise DslExecutionError(
            f"Rule node missing 'op' key. Got keys: {sorted(node.keys())}"
        )

    # ── Logical ──────────────────────────────────────────────────────
    if op in ("and", "or"):
        children = node.get("children")
        if not isinstance(children, list) or len(children) == 0:
            raise DslExecutionError(
                f"'{op}' node requires non-empty 'children' list"
            )
        results = [_eval_node(child, values) for child in children]
        if op == "and":
            out = results[0]
            for r in results[1:]:
                out = out & r
            return out
        else:  # or
            out = results[0]
            for r in results[1:]:
                out = out | r
            return out

    if op == "not":
        child = node.get("child")
        if child is None:
            raise DslExecutionError("'not' node requires a 'child' node")
        return ~_eval_node(child, values)

    # ── Comparison ───────────────────────────────────────────────────
    if op in COMPARISON_OPS:
        left = node.get("left")
        right = node.get("right")
        if left is None or right is None:
            raise DslExecutionError(
                f"'{op}' node requires 'left' and 'right' operands"
            )
        left_s = _resolve_leaf(left, values)
        right_s = _resolve_leaf(right, values)
        if op == "gt":
            return left_s > right_s
        elif op == "lt":
            return left_s < right_s
        elif op == "gte":
            return left_s >= right_s
        else:  # lte
            return left_s <= right_s

    # ── Cross ────────────────────────────────────────────────────────
    if op in CROSS_OPS:
        left = node.get("left")
        right = node.get("right")
        if left is None or right is None:
            raise DslExecutionError(
                f"'{op}' node requires 'left' and 'right' operands"
            )
        left_s = _resolve_leaf(left, values)
        right_s = _resolve_leaf(right, values)

        today = left_s > right_s
        yesterday = (left_s.shift(1) > right_s.shift(1)).fillna(False)

        if op == "crosses_above":
            # crosses above: yesterday was <= threshold, today > threshold
            return today & (~yesterday)
        else:  # crosses_below
            return (~today) & yesterday

    raise DslExecutionError(
        f"Unknown op '{op}'. Supported: {sorted(ALL_OPS)}"
    )


def _eval_rule(
    rule: dict, values: Dict[str, pd.Series]
) -> pd.Series:
    """Evaluate a rule node to a boolean Series (convenience wrapper)."""
    if rule is None:
        raise DslExecutionError("Rule cannot be None")
    return _eval_node(rule, values)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

VALID_POSITION_SIZING = frozenset({
    "long_only_binary",
    "long_short_binary",
})


def _apply_position_sizing(
    entry: pd.Series,
    exit_sig: pd.Series,
    sizing: str,
) -> pd.Series:
    """Apply position-sizing rules to entry/exit boolean signals.

    Returns a Series of position values:
      - long_only_binary:   1 when entry=True, 0 when exit=True
      - long_short_binary:  1 when entry=True, -1 when exit=True, 0 otherwise
    """
    if sizing == "long_only_binary":
        position = pd.Series(0.0, index=entry.index)
        position[entry] = 1.0
        position[exit_sig] = 0.0
        return position
    elif sizing == "long_short_binary":
        position = pd.Series(0.0, index=entry.index)
        position[entry] = 1.0
        # exit_sig in long_short means "go short"
        position[exit_sig] = -1.0
        return position
    else:
        raise DslExecutionError(
            f"Unknown position_sizing '{sizing}'. "
            f"Supported: {sorted(VALID_POSITION_SIZING)}"
        )


# ---------------------------------------------------------------------------
# Top-level runner: single_asset_rule
# ---------------------------------------------------------------------------

def _run_single_asset_rule(
    logic_spec: dict,
    all_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Execute a single_asset_rule logic_spec.

    Returns a signals DataFrame (wide: index=date, columns=symbols)
    with values in {-1, 0, 1}, compatible with _signals_to_weights()
    downstream.
    """
    indicators_spec = logic_spec.get("indicators") or {}
    entry_rule = logic_spec.get("entry_rule")
    exit_rule = logic_spec.get("exit_rule")
    sizing = logic_spec.get("position_sizing", "long_only_binary")

    if sizing not in VALID_POSITION_SIZING:
        raise DslExecutionError(
            f"Unknown position_sizing '{sizing}'. "
            f"Supported: {sorted(VALID_POSITION_SIZING)}"
        )

    # Compute per-symbol signals
    symbol_signals: Dict[str, pd.Series] = {}
    for sym, df in all_data.items():
        if sym.startswith("_"):
            continue  # skip special aux keys

        # Compute indicators for this symbol
        values: Dict[str, pd.Series] = {}
        for name, spec in indicators_spec.items():
            if not isinstance(spec, dict):
                raise DslExecutionError(
                    f"indicator '{name}' spec must be a dict, "
                    f"got {type(spec).__name__}"
                )
            ind_type = spec.get("type")
            window = spec.get("window", 14)
            if ind_type not in VALID_INDICATOR_TYPES:
                raise DslExecutionError(
                    f"Unknown indicator type '{ind_type}' for '{name}'. "
                    f"Supported: {sorted(VALID_INDICATOR_TYPES)}"
                )
            values[name] = _compute_indicator(df, ind_type, window)

        # Evaluate entry and exit rules
        entry_sig = _eval_rule(entry_rule, values) if entry_rule is not None else pd.Series(False, index=df.index)
        exit_sig = _eval_rule(exit_rule, values) if exit_rule is not None else pd.Series(False, index=df.index)

        # Convert to signals
        signal = _apply_position_sizing(entry_sig, exit_sig, sizing)
        symbol_signals[sym] = signal

    return pd.DataFrame(symbol_signals)


# ---------------------------------------------------------------------------
# Top-level runner: cross_sectional_rank
# ---------------------------------------------------------------------------

def _run_cross_sectional_rank(
    logic_spec: dict,
    all_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Execute a cross_sectional_rank logic_spec.

    Returns a wide DataFrame (index=date, columns=symbols) of factor scores
    compatible with the cross-sectional engine (validate_scores shape).
    """
    indicators_spec = logic_spec.get("indicators") or {}
    rank_by = logic_spec.get("rank_by")
    direction = logic_spec.get("direction", "top")
    n_select = logic_spec.get("n_select")
    pct_select = logic_spec.get("pct_select")
    weighting = logic_spec.get("weighting", "equal")

    if rank_by is None:
        raise DslExecutionError("cross_sectional_rank requires 'rank_by'")
    if n_select is None and pct_select is None:
        raise DslExecutionError(
            "cross_sectional_rank requires 'n_select' or 'pct_select'"
        )
    if n_select is not None and pct_select is not None:
        raise DslExecutionError(
            "cross_sectional_rank: use 'n_select' OR 'pct_select', not both"
        )
    if direction not in ("top", "bottom"):
        raise DslExecutionError(
            f"Unknown direction '{direction}'. Use 'top' or 'bottom'"
        )
    if weighting not in ("equal", "score_weighted"):
        raise DslExecutionError(
            f"Unknown weighting '{weighting}'. Use 'equal' or 'score_weighted'"
        )

    # Compute factor scores per symbol
    symbol_scores: Dict[str, pd.Series] = {}
    for sym, df in all_data.items():
        if sym.startswith("_"):
            continue
        ind_spec = indicators_spec.get(rank_by)
        if ind_spec is None:
            raise DslExecutionError(
                f"rank_by indicator '{rank_by}' not defined in indicators"
            )
        ind_type = ind_spec.get("type")
        window = ind_spec.get("window", 14)
        symbol_scores[sym] = _compute_indicator(df, ind_type, window)

    scores_df = pd.DataFrame(symbol_scores)
    return scores_df


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def run_dsl_strategy(
    logic_spec: dict,
    all_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Execute a logic_spec manifest entry, returning a weights/signals/scores
    DataFrame suitable for the downstream manifest_runner pipeline.

    Dispatches on logic_spec['kind']:
      - "single_asset_rule": returns signals DataFrame (values in {-1, 0, 1})
      - "cross_sectional_rank": returns scores DataFrame (wide, date × symbol)
    """
    if not isinstance(logic_spec, dict):
        raise DslExecutionError(
            f"logic_spec must be a dict, got {type(logic_spec).__name__}"
        )

    kind = logic_spec.get("kind")
    if kind is None:
        raise DslExecutionError("logic_spec missing 'kind' field")

    if kind == "single_asset_rule":
        return _run_single_asset_rule(logic_spec, all_data)
    elif kind == "cross_sectional_rank":
        return _run_cross_sectional_rank(logic_spec, all_data)
    else:
        raise DslExecutionError(
            f"Unknown logic_spec kind '{kind}'. "
            "Supported: 'single_asset_rule', 'cross_sectional_rank'"
        )


# ---------------------------------------------------------------------------
# DSL → Python compilation (issue #99)
# ---------------------------------------------------------------------------
#
# The remote runner (sandbox-runner) validates code_b64 and has no
# logic_spec support, so DSL manifests never reach the engine.  The engine
# *does* support DSL locally (_run_dsl_mode), but the live loop posts to the
# remote runner only.  Instead of waiting on a runner change (that repo is a
# trust anchor), we compile a logic_spec deterministically into equivalent
# structured-mode Python at preflight time.  The generated code:
#   - uses ONLY the sandbox globals (pd, np, data) — no imports;
#   - implements exactly the same indicator math and rule evaluation as the
#     interpreter above (trailing-only ops, lookahead structurally
#     impossible);
#   - defines the standard entrypoint: generate_signals (single_asset_rule)
#     or generate_cross_signal (cross_sectional_rank), so the engine's
#     existing structured-mode dispatch runs it unchanged.

_GEN_INDICATOR_FN = '''\
def _indicator(close, ind_type, window):
    if ind_type == "sma":
        return close.rolling(window=window, min_periods=1).mean()
    if ind_type == "ema":
        return close.ewm(span=window, min_periods=1, adjust=False).mean()
    if ind_type == "rsi":
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    if ind_type == "bollinger_pct_b":
        sma = close.rolling(window=window, min_periods=1).mean()
        std = close.rolling(window=window, min_periods=1).std()
        upper = sma + 2.0 * std
        lower = sma - 2.0 * std
        denom = upper - lower
        pct_b = pd.Series(np.nan, index=close.index)
        mask = denom > 0
        pct_b[mask] = (close[mask] - lower[mask]) / denom[mask]
        return pct_b
    if ind_type == "rolling_zscore":
        sma = close.rolling(window=window, min_periods=1).mean()
        std = close.rolling(window=window, min_periods=1).std()
        return (close - sma) / std.replace(0, np.nan)
    if ind_type == "rolling_vol":
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.rolling(window=window, min_periods=1).std()
        return vol * np.sqrt(252)
    if ind_type == "macd":
        fast = close.ewm(span=12, min_periods=1, adjust=False).mean()
        slow = close.ewm(span=window, min_periods=1, adjust=False).mean()
        return fast - slow
    raise ValueError("unknown indicator type: " + str(ind_type))
'''

_GEN_RULE_EVAL = '''\
def _leaf(node, values):
    if "indicator" in node:
        return values[node["indicator"]]
    if "const" in node:
        first = next(iter(values.values()))
        return pd.Series(node["const"], index=first.index)
    raise ValueError("leaf must have 'indicator' or 'const' key")

def _eval_node(node, values):
    op = node["op"]
    if op in ("and", "or"):
        children = node["children"]
        out = _eval_node(children[0], values)
        for ch in children[1:]:
            if op == "and":
                out = out & _eval_node(ch, values)
            else:
                out = out | _eval_node(ch, values)
        return out
    if op == "not":
        return ~_eval_node(node["child"], values)
    if op in ("gt", "lt", "gte", "lte"):
        left = _leaf(node["left"], values)
        right = _leaf(node["right"], values)
        if op == "gt":
            return left > right
        if op == "lt":
            return left < right
        if op == "gte":
            return left >= right
        return left <= right
    if op in ("crosses_above", "crosses_below"):
        left = _leaf(node["left"], values)
        right = _leaf(node["right"], values)
        today = left > right
        yesterday = (left.shift(1) > right.shift(1)).fillna(False)
        if op == "crosses_above":
            return today & (~yesterday)
        return (~today) & yesterday
    raise ValueError("unknown op: " + str(op))
'''


def _validate_rule_tree(node, indicators):
    """Structural validation of a rule node (compile-time mirror of the
    interpreter's runtime checks). Raises DslExecutionError on any node the
    interpreter would reject."""
    if not isinstance(node, dict):
        raise DslExecutionError(f"Rule node must be a dict, got {type(node).__name__}")
    op = node.get("op")
    if op is None:
        raise DslExecutionError(
            f"Rule node missing 'op' key. Got keys: {sorted(node.keys())}"
        )
    if op in ("and", "or"):
        children = node.get("children")
        if not isinstance(children, list) or len(children) == 0:
            raise DslExecutionError(
                f"'{op}' node requires non-empty 'children' list"
            )
        for ch in children:
            _validate_rule_tree(ch, indicators)
        return
    if op == "not":
        if node.get("child") is None:
            raise DslExecutionError("'not' node requires a 'child' node")
        _validate_rule_tree(node["child"], indicators)
        return
    if op in COMPARISON_OPS or op in CROSS_OPS:
        for side in ("left", "right"):
            leaf = node.get(side)
            if leaf is None:
                raise DslExecutionError(
                    f"'{op}' node requires '{side}' operand"
                )
            if not isinstance(leaf, dict):
                raise DslExecutionError(
                    f"'{op}' operand must be a dict, got {type(leaf).__name__}"
                )
            if "indicator" in leaf:
                if leaf["indicator"] not in indicators:
                    raise DslExecutionError(
                        f"Unknown indicator '{leaf['indicator']}'. "
                        f"Available: {sorted(indicators)}"
                    )
            elif "const" in leaf:
                if not isinstance(leaf["const"], (int, float)):
                    raise DslExecutionError(
                        f"const value must be numeric, got {type(leaf['const']).__name__}"
                    )
            else:
                raise DslExecutionError(
                    "Leaf node must have 'indicator' or 'const' key. "
                    f"Got keys: {sorted(leaf.keys())}"
                )
        return
    raise DslExecutionError(
        f"Unknown op '{op}'. Supported: {sorted(ALL_OPS)}"
    )


def _indent(text: str, prefix: str) -> str:
    """Indent every line of ``text`` by ``prefix``.

    Used to embed the module-level code templates (_GEN_INDICATOR_FN,
    _GEN_RULE_EVAL) inside the generated entrypoint function body — the
    templates are written at column 0 and must be re-indented where they
    are interpolated, otherwise the generated source is a SyntaxError.
    """
    return "".join(prefix + line for line in text.splitlines(keepends=True))


def compile_logic_spec_to_code(logic_spec, name="dsl_strategy"):
    """Deterministically compile a logic_spec into self-contained Python
    source for the structured-mode sandbox.

    The remote runner executes code_b64 only — it has no logic_spec support.
    This compiler translates a validated logic_spec into equivalent Python
    (same indicator math, same rule evaluation) defining the standard
    structured entrypoint, so DSL manifests can flow through the existing
    runner unchanged. The DSL is retained in the manifest for review; the
    compiled code is the executable form.

    Returns the Python source string. Raises DslExecutionError for any spec
    the interpreter would reject.
    """
    if not isinstance(logic_spec, dict):
        raise DslExecutionError(
            f"logic_spec must be a dict, got {type(logic_spec).__name__}"
        )
    kind = logic_spec.get("kind")
    if kind is None:
        raise DslExecutionError("logic_spec missing 'kind' field")
    if kind not in ("single_asset_rule", "cross_sectional_rank"):
        raise DslExecutionError(
            f"Unknown logic_spec kind '{kind}'. "
            "Supported: 'single_asset_rule', 'cross_sectional_rank'"
        )

    indicators = logic_spec.get("indicators")
    if not isinstance(indicators, dict) or len(indicators) == 0:
        raise DslExecutionError(
            "logic_spec.indicators must be a non-empty dict"
        )
    for ind_name, ind_spec in indicators.items():
        if not isinstance(ind_spec, dict):
            raise DslExecutionError(
                f"indicator '{ind_name}' spec must be a dict, "
                f"got {type(ind_spec).__name__}"
            )
        ind_type = ind_spec.get("type")
        if ind_type not in VALID_INDICATOR_TYPES:
            raise DslExecutionError(
                f"Unknown indicator type '{ind_type}' for '{ind_name}'. "
                f"Supported: {sorted(VALID_INDICATOR_TYPES)}"
            )

    header = (
        "# Compiled from logic_spec by strategy_dsl.compile_logic_spec_to_code "
        f"(issue #99) for manifest '{name}'.\n"
        "# Semantics identical to the trusted DSL interpreter; trailing-only "
        "indicator ops, no lookahead.\n"
    )

    if kind == "single_asset_rule":
        entry_rule = logic_spec.get("entry_rule")
        exit_rule = logic_spec.get("exit_rule")
        if entry_rule is not None:
            _validate_rule_tree(entry_rule, indicators)
        if exit_rule is not None:
            _validate_rule_tree(exit_rule, indicators)
        sizing = logic_spec.get("position_sizing", "long_only_binary")
        if sizing not in VALID_POSITION_SIZING:
            raise DslExecutionError(
                f"Unknown position_sizing '{sizing}'. "
                f"Supported: {sorted(VALID_POSITION_SIZING)}"
            )
        body = f'''def generate_signals(data):
{_indent(_GEN_INDICATOR_FN, "    ")}
{_indent(_GEN_RULE_EVAL, "    ")}
    _INDICATORS = {indicators!r}
    _ENTRY_RULE = {entry_rule!r}
    _EXIT_RULE = {exit_rule!r}
    _SIZING = {sizing!r}
    signals = {{}}
    for sym, df in data.items():
        if sym.startswith("_"):
            continue
        values = {{}}
        for ind_name, ind_spec in _INDICATORS.items():
            values[ind_name] = _indicator(df["Close"], ind_spec["type"], ind_spec.get("window", 14))
        entry = _eval_node(_ENTRY_RULE, values) if _ENTRY_RULE is not None else pd.Series(False, index=df.index)
        exit_sig = _eval_node(_EXIT_RULE, values) if _EXIT_RULE is not None else pd.Series(False, index=df.index)
        if _SIZING == "long_only_binary":
            position = pd.Series(0.0, index=df.index)
            position[entry] = 1.0
            position[exit_sig] = 0.0
        else:
            position = pd.Series(0.0, index=df.index)
            position[entry] = 1.0
            position[exit_sig] = -1.0
        signals[sym] = position
    return signals
'''
        return header + body

    # cross_sectional_rank — validation mirrors _run_cross_sectional_rank so
    # any spec the interpreter would reject also fails at compile time.
    rank_by = logic_spec.get("rank_by")
    if rank_by is None:
        raise DslExecutionError("cross_sectional_rank requires 'rank_by'")
    if rank_by not in indicators:
        raise DslExecutionError(
            f"rank_by indicator '{rank_by}' not defined in indicators"
        )
    n_select = logic_spec.get("n_select")
    pct_select = logic_spec.get("pct_select")
    if n_select is None and pct_select is None:
        raise DslExecutionError(
            "cross_sectional_rank requires 'n_select' or 'pct_select'"
        )
    if n_select is not None and pct_select is not None:
        raise DslExecutionError(
            "cross_sectional_rank: use 'n_select' OR 'pct_select', not both"
        )
    direction = logic_spec.get("direction", "top")
    if direction not in ("top", "bottom"):
        raise DslExecutionError(
            f"Unknown direction '{direction}'. Use 'top' or 'bottom'"
        )
    weighting = logic_spec.get("weighting", "equal")
    if weighting not in ("equal", "score_weighted"):
        raise DslExecutionError(
            f"Unknown weighting '{weighting}'. Use 'equal' or 'score_weighted'"
        )
    body = f'''def generate_cross_signal(data, extras=None):
{_indent(_GEN_INDICATOR_FN, "    ")}
    _INDICATORS = {indicators!r}
    _RANK_BY = {rank_by!r}
    scores = {{}}
    for sym, df in data.items():
        if sym.startswith("_"):
            continue
        ind_spec = _INDICATORS[_RANK_BY]
        scores[sym] = _indicator(df["Close"], ind_spec["type"], ind_spec.get("window", 14))
    return pd.DataFrame(scores)
'''
    return header + body


def prepare_spec_for_runner(manifest_spec):
    """Return a copy of ``manifest_spec`` that the remote runner will accept.

    The runner validates ``code_b64`` only — it has no ``logic_spec`` support
    (trust-anchor repo; must not be modified here).  The engine additionally
    enforces ``code_b64`` XOR ``logic_spec``.  So every manifest sent to a
    runner endpoint (/run_manifest, /validate_manifest) must carry code and
    nothing else.

    Behaviour (issue #99):
      - ``logic_spec`` present, ``code_b64`` empty  -> compile the DSL
        deterministically (compile_logic_spec_to_code), set ``code_b64`` to
        the compiled source, delete ``logic_spec``.  The original dict is
        NOT mutated; the DSL stays in the backlog/ideation log for review.
      - ``code_b64`` present, no ``logic_spec``     -> shallow-ish pass-through
        (deep copy, unchanged).
      - both present, or neither                    -> (None, error): the
        manifest violates the XOR contract and must be dropped, not sent.
      - ``logic_spec`` present but un-compilable    -> (None, error).

    Returns (prepared_spec, None) on success, (None, error_msg) on failure.
    Never raises.
    """
    try:
        if not isinstance(manifest_spec, dict):
            return None, f"manifest spec must be a dict, got {type(manifest_spec).__name__}"
        has_code = bool(manifest_spec.get("code_b64"))
        logic_spec = manifest_spec.get("logic_spec")
        if has_code and logic_spec is not None:
            return None, ("code_b64 and logic_spec are mutually exclusive — "
                          "the manifest must carry exactly one")
        if not has_code and logic_spec is None:
            return None, ("manifest carries neither code_b64 nor logic_spec — "
                          "nothing to compile or run")
        if has_code:
            return copy.deepcopy(manifest_spec), None
        # logic_spec path: compile deterministically
        try:
            source = compile_logic_spec_to_code(
                logic_spec, name=manifest_spec.get("name", "dsl_strategy"))
        except DslExecutionError as e:
            return None, f"logic_spec compile failed: {e}"
        except Exception as e:  # defensive: compile must never crash the loop
            return None, f"logic_spec compile failed: {type(e).__name__}: {e}"
        prepared = copy.deepcopy(manifest_spec)
        prepared["code_b64"] = base64.b64encode(source.encode("utf-8")).decode("ascii")
        prepared.pop("logic_spec", None)
        return prepared, None
    except Exception as e:  # last-resort guard: callers treat as drop
        return None, f"prepare_spec_for_runner failed: {type(e).__name__}: {e}"
