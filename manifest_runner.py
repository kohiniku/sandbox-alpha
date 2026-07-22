#!/usr/bin/env python3
"""Manifest runner for sandbox-alpha v2.

Executes a strategy manifest end-to-end:
  parse -> validate -> load data -> exec user code -> aggregate portfolio
  returns -> evaluate -> print JSON.

Pipeline
--------
1. Decode + validate manifest (manifest.py).
2. Load OHLCV data per OhlcvSource (data_adapters.ohlcv).
3. Execute user code (generate_signals or generate_weights).
4. Compute portfolio returns from weights + asset returns.
5. Evaluate via evaluators.dispatch.evaluate.
6. Print exactly one JSON (always exit 0).

Error taxonomy
--------------
- 'manifest': schema/validation failure.
- 'infra': data loading failure (MissingDataError, I/O).
- 'code': user code failure (missing entrypoint, runtime exception).
"""
import argparse
import base64
import inspect
import io
import json
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from manifest import (
    OhlcvSource, NewsSentimentSource, InsiderSource, MacroSource, Sec13FSource,
    StrategyManifest, ManifestValidationError,
)
from data_adapters.ohlcv import MissingDataError, align_universe, load_ohlcv
from data_adapters.news_sentiment import load_news_sentiment
from data_adapters.sec_13f import load_sec_13f
from data_adapters.insider import load_insider_trades
from data_adapters.macro import load_macro
from evaluators.dispatch import evaluate

# Cross-sectional contract validators — dual-import for container flat-layout
try:
    from backtests.strategies.cross_sectional._contract import (
        validate_weights,
        validate_signals,
        validate_scores,
    )
except ImportError:
    try:
        # Container flat layout: Dockerfile does `COPY backtests/ /backtest/`,
        # so backtests/strategies/cross_sectional/ lands at /backtest/strategies/
        # cross_sectional/, NOT at /backtest/cross_sectional/ (which is the
        # separate PR 4c engine package). The `strategies.` prefix matters.
        from strategies.cross_sectional._contract import (
            validate_weights,
            validate_signals,
            validate_scores,
        )
    except ImportError:
        # Validators are optional — only needed when generate_cross_signal is used.
        # If the cross_sectional package isn't available (e.g. older container
        # images), dispatch falls through gracefully.
        validate_weights = None  # type: ignore[assignment]
        validate_signals = None  # type: ignore[assignment]
        validate_scores = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Allowed imports inside user code (sandbox)
# ---------------------------------------------------------------------------

_STRUCTURED_MODULES = frozenset({"pandas", "numpy", "pd", "np"})

_EXPERT_MODULES = frozenset({
    "pandas", "numpy", "scipy", "sklearn", "statsmodels",
    # torch is INTENTIONALLY EXCLUDED — the current backtest image does not
    # ship CUDA/torch (~5GB). Re-enable when the sandbox-alpha-backtest:ml
    # image lands. Without this, LLM code doing `import torch` currently
    # crashes with ModuleNotFoundError for every proposal.
    "math", "statistics", "dataclasses", "typing", "collections",
    "functools", "itertools", "json",
})


def _safe_import(name: str, allowlist: frozenset, *args: Any, **kwargs: Any) -> Any:
    """Restricted __import__ that only allows specified modules."""
    top = name.split(".")[0]
    if top not in allowlist:
        raise ImportError(
            f"import '{name}' is not allowed. Allowed modules: {sorted(allowlist)}"
        )
    return __builtins__["__import__"](name, *args, **kwargs) if isinstance(
        __builtins__, dict
    ) else __builtins__.__import__(name, *args, **kwargs)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_with_extras(
    fn: Any, data: Dict[str, pd.DataFrame], extras: Dict[str, Any]
) -> Any:
    """Call fn(data) or fn(data, extras) depending on its signature.

    If fn accepts 2+ parameters, extras is passed as the second argument.
    Otherwise, only data is passed (backward compatible with single-arg
    generate_signals / generate_weights).
    """
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        if len(params) >= 2 and extras:
            return fn(data, extras)
    except (ValueError, TypeError):
        pass
    return fn(data)


def _error_json(error_type: str, error: str, tb: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "error": error,
    }
    if tb:
        out["traceback"] = tb
    return json.dumps(out)


def _signals_to_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """Convert signals in {-1, 0, 1} to equal-weight-normalized portfolio weights.

    Per row:
    - Long positions (signal=1): weight = 1/n_active_long
    - Short positions (signal=-1): weight = -1/n_active_short
    - Flat (signal=0): weight = 0
    If all signals are zero, all weights are zero (flat).
    """
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

    for idx in signals.index:
        row = signals.loc[idx]
        n_long = (row == 1).sum()
        n_short = (row == -1).sum()
        if n_long > 0:
            weights.loc[idx, row == 1] = 1.0 / n_long
        if n_short > 0:
            weights.loc[idx, row == -1] = -1.0 / n_short

    return weights


def _dict_signals_to_wide(signals_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Convert {symbol: Series} signals to a wide DataFrame."""
    return pd.DataFrame(signals_dict)


def _walk_forward_split(
    index: pd.DatetimeIndex, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Split DatetimeIndex into train/val/holdout (60/20/20 by default).
    
    Returns (train_end, val_end) timestamps. Holdout starts after val_end.
    """
    n = len(index)
    train_end_idx = int(n * train_frac) - 1
    val_end_idx = int(n * (train_frac + val_frac)) - 1
    return index[train_end_idx], index[val_end_idx]


# ---------------------------------------------------------------------------
# Required expert metrics
# ---------------------------------------------------------------------------

REQUIRED_EXPERT_METRICS = frozenset({
    "val_sharpe", "val_max_drawdown_pct", "val_total_return_pct",
    "holdout_sharpe", "holdout_max_drawdown_pct", "holdout_total_return_pct",
})


def _validate_expert_metrics(result: dict) -> Optional[str]:
    """Validate expert mode return dict. Returns error message or None if valid.
    
    Non-finite values (NaN/inf) are NOT treated as errors here — those are
    legitimate "no edge" outcomes and are flagged separately by
    _find_nonfinite_metrics for the degenerate-metrics path.
    """
    if not isinstance(result, dict):
        return f"run() must return dict, got {type(result).__name__}"
    
    missing = REQUIRED_EXPERT_METRICS - set(result.keys())
    if missing:
        return f"run() missing required metrics: {sorted(missing)}"
    
    for key in REQUIRED_EXPERT_METRICS:
        val = result[key]
        if not isinstance(val, (int, float)):
            return f"run() metric '{key}' must be numeric, got {type(val).__name__}"
    
    return None


def _find_nonfinite_metrics(result: dict) -> list:
    """Return sorted list of REQUIRED_EXPERT_METRICS keys whose values are
    numeric but non-finite (NaN/inf).  A strategy that produces zero trades
    in a segment will naturally yield zero-variance returns → NaN Sharpe.
    This is not a code bug — it is an honest "no edge" outcome.
    """
    nonfinite = []
    for key in REQUIRED_EXPERT_METRICS:
        val = result.get(key)
        if isinstance(val, (int, float)) and not np.isfinite(val):
            nonfinite.append(key)
    return sorted(nonfinite)


def _check_pathological(result: dict) -> list:
    """Check for pathological metric values. Returns list of warnings."""
    warnings = []
    for key in ["val_sharpe", "holdout_sharpe"]:
        if key in result and abs(result[key]) > 10:
            warnings.append(f"{key}={result[key]}: |sharpe| > 10 is suspicious")
    for key in ["val_max_drawdown_pct", "holdout_max_drawdown_pct"]:
        if key in result and result[key] > 100:
            warnings.append(f"{key}={result[key]}: drawdown > 100% is suspicious")
    for key in ["val_total_return_pct", "holdout_total_return_pct"]:
        if key in result and abs(result[key]) > 10000:
            warnings.append(f"{key}={result[key]}: |return| > 10000% is suspicious")
    return warnings


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_manifest(manifest: StrategyManifest, data_dir: str) -> str:
    """Execute the full manifest pipeline. Returns JSON string."""

    # --- Step 1: Load data ---
    all_data: Dict[str, pd.DataFrame] = {}
    news_df: Optional[pd.DataFrame] = None
    sec13f_df: Optional[pd.DataFrame] = None
    insider_df: Optional[pd.DataFrame] = None
    macro_df: Optional[pd.DataFrame] = None

    for ds in manifest.data_sources:
        if isinstance(ds, OhlcvSource):
            try:
                loaded = load_ohlcv(
                    universe=ds.universe,
                    start=ds.start,
                    end=ds.end,
                    data_dir=data_dir,
                )
                all_data.update(loaded)
            except MissingDataError as e:
                return _error_json("infra", str(e))
        elif isinstance(ds, NewsSentimentSource):
            try:
                news_df = load_news_sentiment(
                    universe=ds.universe,
                    start=ds.start,
                    end=ds.end,
                    source=ds.source,
                    min_relevance=ds.min_relevance,
                    data_dir=data_dir,
                )
            except Exception as e:
                return _error_json("infra", f"News sentiment loading failed: {e}")
        elif isinstance(ds, Sec13FSource):
            try:
                sec13f_df = load_sec_13f(
                    universe=ds.universe,
                    start=ds.start,
                    end=ds.end,
                    filers=ds.filers,
                    min_position_pct=ds.min_position_pct,
                    data_dir=data_dir,
                )
            except Exception as e:
                return _error_json("infra", f"SEC 13F loading failed: {e}")
        elif isinstance(ds, InsiderSource):
            try:
                insider_df = load_insider_trades(
                    universe=ds.universe,
                    start=ds.start,
                    end=ds.end,
                    min_transaction_usd=ds.min_transaction_usd,
                    roles=ds.roles,
                    data_dir=data_dir,
                )
            except Exception as e:
                return _error_json("infra", f"Insider loading failed: {e}")
        elif isinstance(ds, MacroSource):
            try:
                macro_df = load_macro(
                    series=ds.series,
                    start=ds.start,
                    end=ds.end,
                    frequency=ds.frequency,
                    data_dir=data_dir,
                )
            except Exception as e:
                return _error_json("infra", f"Macro loading failed: {e}")

    # Store news under special key if loaded
    if news_df is not None and not news_df.empty:
        all_data["_news_sentiment"] = news_df

    # Store sec_13f under special key if loaded
    if sec13f_df is not None and not sec13f_df.empty:
        all_data["_sec_13f"] = sec13f_df

    # Store insider trades under special key if loaded
    if insider_df is not None and not insider_df.empty:
        all_data["_insider_trades"] = insider_df

    # Store macro under special key if loaded
    if macro_df is not None and not macro_df.empty:
        all_data["_macro"] = macro_df

    if not all_data:
        return _error_json("infra", "No OHLCV data sources declared in manifest")

    # --- Step 2: Execute user code ---
    try:
        code_bytes = base64.b64decode(manifest.code_b64)
        code_str = code_bytes.decode("utf-8")
    except Exception as e:
        return _error_json("code", f"Failed to decode code_b64: {e}")

    # Build sandbox namespace
    allowlist = _STRUCTURED_MODULES if manifest.execution_mode == "structured" else _EXPERT_MODULES
    
    sandbox: Dict[str, Any] = {
        "pd": pd,
        "np": np,
        "pandas": pd,
        "numpy": np,
        "data": all_data,
        "__builtins__": {
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
            "print": lambda *a, **kw: None,  # silence
            "sorted": sorted,
            "reversed": reversed,
            "any": any,
            "all": all,
            "round": round,
            "iter": iter,
            "next": next,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "callable": callable,
            "NotImplementedError": NotImplementedError,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "RuntimeError": RuntimeError,
            "AttributeError": AttributeError,
            "Exception": Exception,
            "True": True,
            "False": False,
            "None": None,
            "__import__": lambda name, *a, **k: _safe_import(name, allowlist, *a, **k),
        },
    }

    try:
        exec(code_str, sandbox)  # noqa: S102
    except Exception as e:
        tb = traceback.format_exc()[-2000:]
        return _error_json("code", f"User code raised {type(e).__name__}: {e}", tb)

    # --- Step 3: Align and compute returns ---
    # Filter out special internal keys (_news_sentiment etc.) for OHLCV alignment
    ohlcv_data = {k: v for k, v in all_data.items() if not k.startswith("_")}
    if not ohlcv_data:
        ohlcv_data = all_data  # fallback if all keys are special (shouldn't happen)

    panel = align_universe(ohlcv_data)
    if panel.empty:
        return _error_json("infra", "align_universe returned empty panel (no common dates)")

    # Extract Close prices from MultiIndex panel
    close_panel = panel.xs("Close", level="field", axis=1)
    asset_returns = close_panel.pct_change()

    # Compute walk-forward split
    train_end, val_end = _walk_forward_split(close_panel.index)

    # --- Step 5: Benchmark ---
    benchmark_symbol = manifest.evaluator.benchmark
    benchmark_series: Optional[pd.Series] = None
    benchmark_warning = None

    if benchmark_symbol:
        if benchmark_symbol in close_panel.columns:
            bm_returns = close_panel[benchmark_symbol].pct_change()
            benchmark_series = bm_returns.reindex(close_panel.index).fillna(0.0)
        else:
            benchmark_warning = (
                f"Benchmark '{benchmark_symbol}' not in universe {list(close_panel.columns)}; "
                f"IR will be skipped."
            )

    # --- Step 6: Route by execution_mode ---
    if manifest.execution_mode == "expert":
        return _run_expert_mode(
            manifest=manifest,
            sandbox=sandbox,
            all_data=all_data,
            close_panel=close_panel,
            train_end=train_end,
            val_end=val_end,
            benchmark_series=benchmark_series,
            benchmark_warning=benchmark_warning,
        )
    else:
        return _run_structured_mode(
            manifest=manifest,
            sandbox=sandbox,
            all_data=all_data,
            close_panel=close_panel,
            asset_returns=asset_returns,
            train_end=train_end,
            val_end=val_end,
            benchmark_series=benchmark_series,
            benchmark_warning=benchmark_warning,
            news_df=news_df,
            sec13f_df=sec13f_df,
        )


def _run_structured_mode(
    manifest: StrategyManifest,
    sandbox: Dict[str, Any],
    all_data: Dict[str, pd.DataFrame],
    close_panel: pd.DataFrame,
    asset_returns: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    benchmark_series: Optional[pd.Series],
    benchmark_warning: Optional[str],
    news_df: Optional[pd.DataFrame] = None,
    sec13f_df: Optional[pd.DataFrame] = None,
    extras_in: Optional[Dict[str, Any]] = None,
) -> str:
    """Execute structured mode: generate_signals/generate_weights entrypoints."""
    
    # Build extras dict for functions that accept additional arguments
    extras: Dict[str, Any] = {}
    if news_df is not None and not news_df.empty:
        extras["news_sentiment"] = news_df
    if sec13f_df is not None and not sec13f_df.empty:
        extras["sec_13f"] = sec13f_df
    # Merge caller-provided extras (e.g. cross_return_type for dispatch routing)
    if extras_in:
        extras.update(extras_in)

    # Execute user code — check for all supported entrypoints
    has_signals = callable(sandbox.get("generate_signals"))
    has_weights = callable(sandbox.get("generate_weights"))
    has_cross_signal = callable(sandbox.get("generate_cross_signal"))

    if not has_signals and not has_weights and not has_cross_signal:
        return _error_json(
            "code",
            "User code must define generate_signals(data), generate_weights(data), "
            "or generate_cross_signal(data, extras). "
            "None was found after exec.",
        )

    # ── Cross-sectional dispatch ──────────────────────────────────────
    # Contract validation (PR 4b), engine wiring (PR 4c).
    if has_cross_signal:
        try:
            result = _call_with_extras(
                sandbox["generate_cross_signal"], all_data, extras
            )
        except Exception as e:
            tb = traceback.format_exc()[-2000:]
            return _error_json(
                "code", f"generate_cross_signal raised {type(e).__name__}: {e}", tb
            )

        if not isinstance(result, pd.DataFrame):
            return _error_json(
                "code",
                f"generate_cross_signal must return DataFrame, got {type(result).__name__}",
            )

        # Infer universe from the loaded data (exclude special internal keys)
        universe = sorted(k for k in all_data if not k.startswith("_"))

        # Determine return type from extras, default to "scores"
        return_type = extras.get("cross_return_type", "scores")

        # Validate against the matching contract
        if validate_weights is None:
            return _error_json(
                "infra",
                "Cross-sectional validators are not available in this environment. "
                "The cross_sectional package must be present in the backtest image.",
            )

        try:
            if return_type == "weights":
                validate_weights(result, universe)
            elif return_type == "signals":
                validate_signals(result, universe)
            elif return_type == "scores":
                validate_scores(result, universe)
            else:
                return _error_json(
                    "code",
                    f"Unknown cross_return_type '{return_type}'. "
                    "Must be one of: weights, signals, scores.",
                )
        except ValueError as e:
            return _error_json("cross_contract_violation", str(e))

        # ── Cross-sectional engine (PR 4c) ────────────────────────────
        try:
            from backtests.cross_sectional.engine import run_cross_sectional_backtest
        except ImportError:
            from cross_sectional.engine import run_cross_sectional_backtest  # type: ignore[no-redef]  # flat-container fallback

        try:
            result_dict = run_cross_sectional_backtest(
                raw_output=result,
                return_type=return_type,
                universe=universe,
                panel=all_data,
                config={
                    **extras,
                    "train_end": train_end,
                    "val_end": val_end,
                },
            )
            return json.dumps(result_dict, default=str)
        except ValueError as e:
            return _error_json("cross_engine_error", str(e))

    use_weights_fn = has_weights
    weighting_label = "generate_weights" if use_weights_fn else "equal_active_signals"

    try:
        if use_weights_fn:
            raw_weights = _call_with_extras(
                sandbox["generate_weights"], all_data, extras
            )
            if isinstance(raw_weights, dict):
                weights_df = pd.DataFrame(raw_weights)
            elif isinstance(raw_weights, pd.DataFrame):
                weights_df = raw_weights
            else:
                return _error_json(
                    "code",
                    f"generate_weights must return DataFrame or dict, got {type(raw_weights).__name__}",
                )
        else:
            raw_signals = _call_with_extras(
                sandbox["generate_signals"], all_data, extras
            )
            if isinstance(raw_signals, dict):
                signals_df = _dict_signals_to_wide(raw_signals)
            elif isinstance(raw_signals, pd.DataFrame):
                signals_df = raw_signals
            else:
                return _error_json(
                    "code",
                    f"generate_signals must return DataFrame or dict, got {type(raw_signals).__name__}",
                )
            weights_df = _signals_to_weights(signals_df)
    except Exception as e:
        tb = traceback.format_exc()[-2000:]
        return _error_json("code", f"Entrypoint raised {type(e).__name__}: {e}", tb)

    # Align weights to asset_returns index/symbols
    symbols = list(close_panel.columns)
    weights_aligned = weights_df.reindex(index=asset_returns.index, columns=symbols, fill_value=0.0)
    weights_aligned = weights_aligned.ffill().fillna(0.0)

    # Portfolio return: (weights.shift(1) * asset_returns).sum(axis=1), drop first
    portfolio_ret = (weights_aligned.shift(1) * asset_returns).sum(axis=1).iloc[1:]
    weights_for_eval = weights_aligned.shift(1).iloc[1:]

    if len(portfolio_ret) < 2:
        return _error_json("code", "Portfolio return series has fewer than 2 rows after alignment")

    # Split into val and holdout periods
    val_returns = portfolio_ret[train_end < portfolio_ret.index]
    val_returns = val_returns[val_returns.index <= val_end]
    holdout_returns = portfolio_ret[portfolio_ret.index > val_end]

    val_weights = weights_for_eval.reindex(val_returns.index)
    holdout_weights = weights_for_eval.reindex(holdout_returns.index)

    # Evaluate val and holdout separately
    returns_df = asset_returns.reindex(portfolio_ret.index)

    try:
        # Val metrics
        val_metrics = evaluate(
            spec=manifest.evaluator,
            returns=returns_df.reindex(val_returns.index),
            weights=val_weights,
            benchmark=benchmark_series.reindex(val_returns.index) if benchmark_series is not None else None,
            config=manifest.evaluator.extras if manifest.evaluator.extras else None,
        )
        # Holdout metrics
        holdout_metrics = evaluate(
            spec=manifest.evaluator,
            returns=returns_df.reindex(holdout_returns.index),
            weights=holdout_weights,
            benchmark=benchmark_series.reindex(holdout_returns.index) if benchmark_series is not None else None,
            config=manifest.evaluator.extras if manifest.evaluator.extras else None,
        )
    except Exception as e:
        tb = traceback.format_exc()[-2000:]
        return _error_json("infra", f"Evaluator failed: {e}", tb)

    # Prefix metrics with val_ and holdout_
    metrics = {}
    for k, v in val_metrics.items():
        metrics[f"val_{k}"] = v
    for k, v in holdout_metrics.items():
        metrics[f"holdout_{k}"] = v

    # Add convenience metrics
    if "val_sharpe" not in metrics:
        metrics["val_sharpe"] = np.nan
    if "val_max_drawdown_pct" not in metrics:
        metrics["val_max_drawdown_pct"] = np.nan
    if "val_total_return_pct" not in metrics:
        metrics["val_total_return_pct"] = float((1 + val_returns).prod() - 1) * 100
    if "holdout_sharpe" not in metrics:
        metrics["holdout_sharpe"] = np.nan
    if "holdout_max_drawdown_pct" not in metrics:
        metrics["holdout_max_drawdown_pct"] = np.nan
    if "holdout_total_return_pct" not in metrics:
        metrics["holdout_total_return_pct"] = float((1 + holdout_returns).prod() - 1) * 100

    result: Dict[str, Any] = {
        "status": "ok",
        "execution_mode": "structured",
        "manifest_name": manifest.name,
        "universe_size": len(symbols),
        "n_days": len(portfolio_ret),
        "metrics": metrics,
        "config": {
            "benchmark": manifest.evaluator.benchmark,
            "weighting": weighting_label,
            "train_end": train_end.isoformat(),
            "val_end": val_end.isoformat(),
        },
    }
    if benchmark_warning:
        result["warning"] = benchmark_warning

    return json.dumps(result)


def _run_expert_mode(
    manifest: StrategyManifest,
    sandbox: Dict[str, Any],
    all_data: Dict[str, pd.DataFrame],
    close_panel: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    benchmark_series: Optional[pd.Series],
    benchmark_warning: Optional[str],
) -> str:
    """Execute expert mode: run() entrypoint returns metrics dict directly."""
    
    # Check for run() entrypoint
    if not callable(sandbox.get("run")):
        return _error_json(
            "code",
            "Expert mode requires run(data, train_end, val_end, benchmark, config) entrypoint. "
            "Function not found after exec.",
        )

    # Prepare config dict
    config = manifest.evaluator.extras if manifest.evaluator.extras else {}

    # Call run()
    try:
        result_dict = sandbox["run"](
            data=all_data,
            train_end=train_end,
            val_end=val_end,
            benchmark=benchmark_series,
            config=config,
        )
    except Exception as e:
        tb = traceback.format_exc()[-2000:]
        return _error_json("code", f"run() raised {type(e).__name__}: {e}", tb)

    # Validate return value
    error_msg = _validate_expert_metrics(result_dict)
    if error_msg:
        return _error_json("code", error_msg)

    # Degenerate metrics: numeric but non-finite (e.g. NaN Sharpe from
    # zero-variance returns caused by no trades).  This is NOT a code error —
    # it is an honest "no edge" outcome.  Report as a success with degenerate
    # flag so the autonomous loop can short-circuit evaluation without marking
    # it as an error.
    nonfinite = _find_nonfinite_metrics(result_dict)
    if nonfinite:
        metrics = {k: result_dict[k] for k in REQUIRED_EXPERT_METRICS}
        extras = {k: v for k, v in result_dict.items() if k not in REQUIRED_EXPERT_METRICS}
        result: Dict[str, Any] = {
            "status": "ok",
            "execution_mode": "expert",
            "manifest_name": manifest.name,
            "universe_size": len(close_panel.columns),
            "n_days": len(close_panel),
            "metrics": metrics,
            "degenerate": True,
            "degenerate_reason": f"metrics not finite: {nonfinite} (likely no trades in a segment)",
            "config": {
                "benchmark": manifest.evaluator.benchmark,
                "entrypoint": "run",
                "train_end": train_end.isoformat(),
                "val_end": val_end.isoformat(),
            },
        }
        if extras:
            result["expert_extras"] = extras
        if benchmark_warning:
            result["warning"] = benchmark_warning
        return json.dumps(result)

    # Check for pathological values
    warnings = _check_pathological(result_dict)
    
    # Separate required vs extra metrics
    metrics = {k: result_dict[k] for k in REQUIRED_EXPERT_METRICS}
    extras = {k: v for k, v in result_dict.items() if k not in REQUIRED_EXPERT_METRICS}

    result: Dict[str, Any] = {
        "status": "ok",
        "execution_mode": "expert",
        "manifest_name": manifest.name,
        "universe_size": len(close_panel.columns),
        "n_days": len(close_panel),
        "metrics": metrics,
        "config": {
            "benchmark": manifest.evaluator.benchmark,
            "entrypoint": "run",
            "train_end": train_end.isoformat(),
            "val_end": val_end.isoformat(),
        },
    }
    if extras:
        result["expert_extras"] = extras
    if benchmark_warning:
        result["warning"] = benchmark_warning
    if warnings:
        result["pathological_warnings"] = warnings

    return json.dumps(result)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(symbols, n_days=250, seed=42):
    """Deterministic multi-symbol OHLCV for preflight execution.

    Same shape/columns as load_ohlcv output so the manifest code sees a
    realistic input without touching the runner cache.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp("2026-07-01"), periods=n_days)
    data = {}
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0005, 0.015, size=n_days)
        close = 100 * (1 + rets).cumprod()
        # give each symbol its own drift so cross-sectional strategies see variance
        close = close * (1 + i * 0.02)
        df = pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.002, size=n_days)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, size=n_days))),
            "Low":  close * (1 - np.abs(rng.normal(0, 0.005, size=n_days))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, size=n_days),
        }, index=idx)
        df.index.name = "Date"
        data[sym] = df
    return data


def _validate_manifest_synthetic(manifest):
    """Run the manifest against synthetic data to catch runtime bugs.

    Returns a JSON string with {valid: bool, error, error_type, traceback}.
    Never raises. Used by the runner's /validate_manifest endpoint via
    manifest_runner --synthetic-run.
    """
    # Extract universe from the first ohlcv source, defaulting to a small set
    universe = None
    for ds in manifest.data_sources:
        if getattr(ds, "type", None) == "ohlcv":
            universe = list(getattr(ds, "universe", []) or [])
            break
    if not universe:
        universe = ["SPY", "QQQ", "AAPL"]
    universe = universe[:5]  # cap for preflight speed
    data = _synthetic_ohlcv(universe)

    # Decode user code
    try:
        code_str = base64.b64decode(manifest.code_b64).decode("utf-8")
    except Exception as e:
        return json.dumps({"valid": False, "error_type": "code",
                           "error": f"code_b64 decode failed: {e}"})

    allowlist = _STRUCTURED_MODULES if manifest.execution_mode == "structured" else _EXPERT_MODULES
    sandbox = {
        "pd": pd, "np": np, "pandas": pd, "numpy": np, "data": data,
        "__builtins__": {
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "map": map, "filter": filter, "sum": sum, "min": min, "max": max,
            "abs": abs, "int": int, "float": float, "str": str, "list": list,
            "dict": dict, "set": set, "tuple": tuple, "bool": bool, "type": type,
            "print": print, "isinstance": isinstance, "hasattr": hasattr,
            "getattr": getattr, "sorted": sorted, "reversed": reversed,
            "round": round, "any": any, "all": all,
            "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "AttributeError": AttributeError,
            "Exception": Exception, "True": True, "False": False, "None": None,
            "__import__": lambda name, *a, **k: _safe_import(name, allowlist, *a, **k),
        },
    }

    # Compile+exec (SyntaxError caught here even though caller usually pre-checks)
    try:
        exec(code_str, sandbox)
    except Exception as e:
        return json.dumps({"valid": False, "error_type": "code",
                           "error": f"exec raised {type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-1500:]})

    # Call the appropriate entrypoint on synthetic data
    idx = next(iter(data.values())).index
    train_end = idx[int(len(idx) * 0.6)]
    val_end = idx[int(len(idx) * 0.8)]

    try:
        if manifest.execution_mode == "expert":
            fn = sandbox.get("run")
            if not callable(fn):
                return json.dumps({"valid": False, "error_type": "code",
                                   "error": "expert mode requires def run(data, train_end, val_end, benchmark, config)"})
            bench = data[universe[0]]["Close"].pct_change()
            result = fn(data, train_end, val_end, bench, manifest.evaluator.extras or {})
            if not isinstance(result, dict):
                return json.dumps({"valid": False, "error_type": "code",
                                   "error": f"run() returned {type(result).__name__}, expected dict"})
            required = {"val_sharpe", "val_max_drawdown_pct", "val_total_return_pct",
                        "holdout_sharpe", "holdout_max_drawdown_pct", "holdout_total_return_pct"}
            missing = sorted(required - set(result))
            if missing:
                return json.dumps({"valid": False, "error_type": "code",
                                   "error": f"run() dict missing required keys: {missing}"})
            # Type check (non-numeric values like Series) — reuse real-run validation
            type_error = _validate_expert_metrics(result)
            if type_error:
                return json.dumps({"valid": False, "error_type": "code",
                                   "error": type_error})
            # Non-finite metrics on synthetic data: warn, don't block.
            # Synthetic no-trade does not imply real-data no-trade.
            nf = _find_nonfinite_metrics(result)
            if nf:
                warnings = [f"run() metric '{k}' is not finite on synthetic data" for k in nf]
                return json.dumps({"valid": True, "warnings": warnings})
        else:
            fn = sandbox.get("generate_weights") or sandbox.get("generate_signals")
            if not callable(fn):
                return json.dumps({"valid": False, "error_type": "code",
                                   "error": "structured mode requires def generate_signals(data) or def generate_weights(data)"})
            _out = fn(data)
            # Basic shape check
            if _out is None:
                return json.dumps({"valid": False, "error_type": "code",
                                   "error": "generate_signals/weights returned None"})
    except Exception as e:
        return json.dumps({"valid": False, "error_type": "code",
                           "error": f"entrypoint raised {type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-1500:]})

    return json.dumps({"valid": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-b64", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--synthetic-run", action="store_true",
                        help="Preflight mode: run against synthetic data, no data-dir needed.")
    args = parser.parse_args()

    # Decode manifest
    try:
        raw = base64.b64decode(args.manifest_b64, validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(_error_json("infra", f"failed to decode manifest: {e}"))
        return 0

    # Parse manifest
    try:
        manifest = StrategyManifest.from_dict(payload)
    except ManifestValidationError as e:
        print(_error_json("manifest", str(e)))
        return 0
    except Exception as e:
        print(_error_json("infra", f"unexpected error parsing manifest: {e}",
                          traceback.format_exc()[-2000:]))
        return 0

    # Validate
    violations = manifest.validate()
    if violations:
        print(json.dumps({
            "status": "error",
            "error_type": "manifest",
            "error": "manifest validation failed",
            "violations": violations,
        }))
        return 0

    # Synthetic preflight branch
    if args.synthetic_run:
        print(_validate_manifest_synthetic(manifest))
        return 0

    if not args.data_dir:
        print(_error_json("infra", "--data-dir required in normal mode"))
        return 0
    # Run
    output = run_manifest(manifest, args.data_dir)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
