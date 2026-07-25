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
