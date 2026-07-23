#!/usr/bin/env python3
"""
Trusted strategy harness for LLM-generated signal code.

Usage:
  python3 strategy_harness.py --code-b64 <base64> --symbol SYM --data-dir /data

Pipeline:
  (a) base64-decode, reject > 64 KB
  (b) AST safety check
  (c) Exec in restricted namespace, extract generate_signals(df) -> pd.Series
  (d) Causality (lookahead) detection
  (e) Apply cost model + 3-way split metrics
  (f) Output JSON
"""
import argparse
import ast
import base64
import hashlib
import os
import sys
import json
import io
import numpy as np
import pandas as pd

try:  # package import (pytest) / script import (container)
    from .metrics import (
    COST_BPS,
    load_cached_data,
    split_walkforward,
    apply_trading_cost,
    compute_split_metrics,
)
except ImportError:
    from metrics import (
    COST_BPS,
    load_cached_data,
    split_walkforward,
    apply_trading_cost,
    compute_split_metrics,
)

MAX_CODE_BYTES = 64 * 1024  # 64 KB

# ---------------------------------------------------------------------------
# (b) AST safety check — best-effort hygiene
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
FORBIDDEN_BUILTINS = {"__import__", "eval", "exec", "compile", "open",
                       "getattr", "setattr", "delattr"}


class SafetyVisitor(ast.NodeVisitor):
    """Walk the AST and reject forbidden constructs."""

    def __init__(self):
        self.errors = []

    def _reject(self, node, what):
        self.errors.append(what)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.name.split(".")[0]
            if name not in ALLOWED_IMPORTS:
                self._reject(node, f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module is None:
            self._reject(node, "relative import (no module)")
        else:
            name = node.module.split(".")[0]
            if name not in ALLOWED_IMPORTS:
                self._reject(node, f"from {node.module} import ...")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for forbidden builtin calls like eval(...), exec(...), etc.
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
            self._reject(node, f"call to {node.func.id}()")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Reject attribute access whose name starts and ends with __
        # EXCEPT __init__ in class defs (checked via context — we just ban all
        # dunder access for simplicity since strategy code shouldn't need any).
        if isinstance(node.attr, str):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                self._reject(node, f"dunder attribute: .{node.attr}")
        self.generic_visit(node)

    def visit_Global(self, node):
        # global at module level is fine, but in functions it's suspicious
        # We'll allow it; it's not a security concern for this context.
        self.generic_visit(node)


def check_safety(code_str):
    """Run AST safety check. Returns None if OK, else error string."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return f"syntax error: {e}"

    visitor = SafetyVisitor()
    visitor.visit(tree)

    if visitor.errors:
        return f"forbidden construct: {visitor.errors[0]}"
    return None


# ---------------------------------------------------------------------------
# (c) Exec in restricted namespace
# ---------------------------------------------------------------------------

# Allow-listed builtins
class _PrintCatcher:
    """A print replacement that writes to the harness-suppressed stdout stream.
    User print() calls go here and never reach the harness's real stdout."""
    def __call__(self, *args, **kwargs):
        # Write to the currently redirected stdout (or real stdout if none)
        pass  # Accept and discard — real stdout is redirected during exec/signal calls
    def write(self, s):
        pass
    def flush(self):
        pass

SAFE_BUILTINS = {
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "enumerate": enumerate,
    "zip": zip,
    "float": float,
    "int": int,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "sorted": sorted,
    "round": round,
    "isinstance": isinstance,
    "Exception": Exception,
    "ValueError": ValueError,
    "print": _PrintCatcher(),
}


def _safe_import(name, *args, **kwargs):
    # AST 検査の許可リストと同一 — import 文を実行時にも同じ集合に制限する
    if name.split(".")[0] in ("numpy", "pandas", "math"):
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import not allowed: {name}")


SAFE_BUILTINS["__import__"] = _safe_import


def exec_and_extract(code_str, df):
    """Exec user code in restricted namespace, return generate_signals(df)."""
    namespace = {
        "np": np,
        "pd": pd,
        "math": __import__("math"),
        "__builtins__": SAFE_BUILTINS,
    }

    # Capture stdout during exec and signal generation
    with io.StringIO() as buf, __import__("contextlib").redirect_stdout(buf):
        exec(code_str, namespace)

    if "generate_signals" not in namespace:
        raise RuntimeError("generate_signals function not defined")

    generate_signals = namespace["generate_signals"]
    if not callable(generate_signals):
        raise RuntimeError("generate_signals is not callable")

    return generate_signals


def _call_signals(generate_signals, df):
    """Call generate_signals(df) with stdout suppressed. Returns validated Series."""
    with io.StringIO() as buf, __import__("contextlib").redirect_stdout(buf):
        result = generate_signals(df.copy())

    if not isinstance(result, pd.Series):
        raise RuntimeError(f"generate_signals must return pd.Series, got {type(result).__name__}")

    # Fill NaN with 0
    result = result.fillna(0)

    # Validate values are in {-1, 0, 1}
    unique_vals = set(result.unique())
    if not unique_vals.issubset({-1.0, 0.0, 1.0, -1, 0, 1}):
        raise RuntimeError(
            f"generate_signals returned invalid values: {unique_vals}. Must be in {{-1, 0, 1}}"
        )

    # Validate index matches
    if not result.index.equals(df.index):
        raise RuntimeError("generate_signals returned Series with index != df.index")

    return result


# ---------------------------------------------------------------------------
# (d) Causality (lookahead) check
# ---------------------------------------------------------------------------

def check_lookahead(generate_signals, df):
    """
    Pick 3 deterministic truncation points and verify signals at t do not
    depend on future data.  Allow warm-up: compare indices >= 50.
    """
    n = len(df)
    if n < 100:
        # Too short for meaningful check — skip (test data should be longer)
        return

    # Deterministic truncation points: at 50%, 70%, 85% of len(df)
    fractions = [0.50, 0.70, 0.85]
    full_signals = _call_signals(generate_signals, df)

    for frac in fractions:
        k = int(n * frac)
        truncated = df.iloc[:k]
        partial_signals = _call_signals(generate_signals, truncated)

        # Compare indices >= 50 (warm-up allowance)
        compare_start = 50
        if k <= compare_start:
            continue  # slice too small

        full_slice = full_signals.iloc[compare_start:k]
        partial_slice = partial_signals.iloc[compare_start:k]

        # Both must be the same length
        if len(full_slice) != len(partial_slice):
            raise RuntimeError("lookahead detected: signals at t depend on future data")

        # Use allclose for float comparison
        if not np.allclose(full_slice.values, partial_slice.values):
            raise RuntimeError("lookahead detected: signals at t depend on future data")


# ---------------------------------------------------------------------------
# (e) Execution model + metrics
# ---------------------------------------------------------------------------

def compute_position_returns(df, signals):
    """positions = signals.shift(1).fillna(0); strategy_returns = pos * daily_ret"""
    positions = signals.shift(1).fillna(0)
    daily_returns = df["Close"].pct_change()
    strategy_returns = positions * daily_returns
    return strategy_returns.dropna(), signals


def build_synthetic_df(n_days=250, seed=42):
    """Build a deterministic synthetic OHLCV DataFrame for preflight validation.

    Returns a DataFrame with DatetimeIndex, columns Open/High/Low/Close/Volume,
    using seeded random-walk prices (numpy default_rng).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")
    daily_ret = rng.normal(0.0003, 0.012, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_ret)
    # Derive O/H/L/V from close with small noise
    open_ = close * (1.0 + rng.uniform(-0.002, 0.002, size=n_days))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.005, size=n_days))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.005, size=n_days))
    volume = rng.integers(500_000, 5_000_000, size=n_days).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


def run_preflight(code_b64_str, cost_bps=None):
    """Preflight validation: decode → safety → exec → signal contract → backtest on synthetic data.

    Returns a JSON-serialisable dict:
      success: {"valid": true, "n_signals": <int>}
      failure: {"valid": false, "error": "<msg>", "traceback": "<last ~15 lines>"}
    """
    import traceback as tb_module

    # (a) decode
    try:
        decoded = base64.b64decode(code_b64_str)
    except Exception as e:
        return {"valid": False, "error": f"base64 decode failed: {e}", "traceback": ""}

    if len(decoded) > MAX_CODE_BYTES:
        return {"valid": False, "error": f"code too large: {len(decoded)} bytes (max {MAX_CODE_BYTES})", "traceback": ""}

    try:
        code_str_decoded = decoded.decode("utf-8")
    except Exception as e:
        return {"valid": False, "error": f"utf-8 decode failed: {e}", "traceback": ""}

    # (b) AST safety check
    safety_err = check_safety(code_str_decoded)
    if safety_err:
        return {"valid": False, "error": safety_err, "traceback": ""}

    # Build synthetic data
    df = build_synthetic_df()

    # (c) Exec and extract
    try:
        generate_signals = exec_and_extract(code_str_decoded, df)
    except Exception as e:
        tb_lines = tb_module.format_exc().splitlines()[-15:]
        return {"valid": False, "error": f"exec failed: {e}", "traceback": "\n".join(tb_lines)}

    # Validate signals on full data
    try:
        signals = _call_signals(generate_signals, df)
    except Exception as e:
        tb_lines = tb_module.format_exc().splitlines()[-15:]
        return {"valid": False, "error": f"signal validation failed: {e}", "traceback": "\n".join(tb_lines)}

    # Fast backtest pass on synthetic data to catch downstream errors
    try:
        strat_returns, _ = compute_position_returns(df, signals)
        strat_returns_net = apply_trading_cost(strat_returns, signals.reindex(strat_returns.index), cost_bps=cost_bps)
        _ = compute_split_metrics(strat_returns_net, signals, len(df), cost_bps=cost_bps)
    except Exception as e:
        tb_lines = tb_module.format_exc().splitlines()[-15:]
        return {"valid": False, "error": f"backtest pass failed: {e}", "traceback": "\n".join(tb_lines)}

    n_signals = int((signals != 0).sum())
    return {"valid": True, "n_signals": n_signals}


def run_harness(code_str, symbol, data_dir, cost_bps=None):
    """Full harness pipeline: decode → safety → exec → lookahead → metrics → JSON."""

    # (a) decode
    try:
        decoded = base64.b64decode(code_str)
    except Exception as e:
        return {"error": f"base64 decode failed: {e}"}

    if len(decoded) > MAX_CODE_BYTES:
        return {"error": f"code too large: {len(decoded)} bytes (max {MAX_CODE_BYTES})"}

    code_str_decoded = decoded.decode("utf-8")

    # code_hash
    code_hash = hashlib.sha256(decoded).hexdigest()

    # (b) AST safety check
    safety_err = check_safety(code_str_decoded)
    if safety_err:
        return {"error": safety_err}

    # Load data
    try:
        df = load_cached_data(symbol, data_dir)
    except SystemExit:
        return {"error": f"data not cached: {symbol}"}
    if df.empty:
        return {"error": f"No data for {symbol}"}

    # (c) Exec and extract
    try:
        generate_signals = exec_and_extract(code_str_decoded, df)
    except Exception as e:
        return {"error": f"exec failed: {e}"}

    # Validate signals on full data
    try:
        signals = _call_signals(generate_signals, df)
    except Exception as e:
        return {"error": f"signal validation failed: {e}"}

    # (d) Causality check
    try:
        check_lookahead(generate_signals, df)
    except Exception as e:
        return {"error": str(e)}

    # (e) Compute metrics: 3-way chronological split
    train_df, val_df, holdout_df = split_walkforward(df)

    def _segment_metrics(seg_df):
        strat_returns, sig = compute_position_returns(seg_df, signals.reindex(seg_df.index))
        strat_returns_net = apply_trading_cost(strat_returns, sig.reindex(strat_returns.index), cost_bps=cost_bps)
        return compute_split_metrics(strat_returns_net, sig, len(seg_df), cost_bps=cost_bps)

    is_metrics = _segment_metrics(train_df)
    val_metrics = _segment_metrics(val_df)
    holdout_metrics = _segment_metrics(holdout_df)

    result = {
        "strategy": "codegen",
        "code_hash": code_hash,
        "symbol": symbol,
        "data_points": len(df),
        "date_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        "in_sample": is_metrics,
        "out_of_sample": val_metrics,
        "holdout": holdout_metrics,
        "walkforward": {"enabled": True, "train_ratio": 0.6, "val_ratio": 0.2, "holdout_ratio": 0.2},
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trusted Strategy Harness")
    parser.add_argument("--code-b64", required=True, help="Base64-encoded strategy code")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="Ticker symbol (normal backtest mode)")
    group.add_argument("--synthetic", action="store_true",
                       help="Preflight mode: validate against synthetic OHLCV data")

    parser.add_argument("--data-dir", help="Directory for cached CSV files (normal mode)")
    parser.add_argument("--cost-bps", type=float, default=None,
                       help="Override trading cost in bps (0.0-100.0, default 5.0)")
    args = parser.parse_args()

    # Validate --cost-bps if provided
    if args.cost_bps is not None:
        if args.cost_bps < 0 or args.cost_bps > 100:
            parser.error(f"--cost-bps must be in [0.0, 100.0], got {args.cost_bps}")

    if args.synthetic:
        result = run_preflight(args.code_b64, cost_bps=args.cost_bps)
        print(json.dumps(result, default=str))
        sys.exit(0)  # validity is in the payload, always exit 0
    else:
        if not args.data_dir:
            parser.error("--data-dir is required in normal mode (without --synthetic)")
        result = run_harness(args.code_b64, args.symbol, args.data_dir, cost_bps=args.cost_bps)

        if "error" in result:
            print(json.dumps({"error": result["error"]}))
            sys.exit(1)

        print(json.dumps(result, indent=2, default=str))
