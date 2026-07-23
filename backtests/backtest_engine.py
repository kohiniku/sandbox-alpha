#!/usr/bin/env python3
"""
Minimal backtest engine for sandbox testing.
Usage:
  python backtest_engine.py --strategy sma_crossover --symbol AAPL --params '{"fast_window": 10, "slow_window": 30}'
  python backtest_engine.py --fetch-only --symbol AAPL --data-dir /cache
  python backtest_engine.py --strategy momentum --symbol AAPL --data-dir /cache
"""
import argparse
import json
import os
import sys
import pandas as pd
from datetime import datetime

try:  # package import (pytest) / script import (container)
    from .metrics import (
    COST_BPS,
    load_cached_data,
    split_walkforward,
    apply_trading_cost,
    calculate_metrics,
    compute_split_metrics,
)
    from .strategies import (
    STRATEGIES,
    run_mean_reversion_strategy,
    run_momentum_strategy,
    run_rsi_strategy,
    run_sma_crossover_strategy,
)
except ImportError:
    from metrics import (
    COST_BPS,
    load_cached_data,
    split_walkforward,
    apply_trading_cost,
    calculate_metrics,
    compute_split_metrics,
)
    from strategies import (
    STRATEGIES,
    run_mean_reversion_strategy,
    run_momentum_strategy,
    run_rsi_strategy,
    run_sma_crossover_strategy,
)


def fetch_data(symbol, period="5y"):
    """Fetch historical data from yfinance (lazy import — only when needed)."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    return df


def fetch_and_cache(symbol, data_dir):
    """Fetch 5y daily OHLCV and save to CSV. Returns JSON info dict."""
    df = fetch_data(symbol)
    path = os.path.join(data_dir, f"{symbol}.csv")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(path, index=True)
    info = {"fetched": symbol, "rows": len(df), "path": path}
    return info


def run_strategy_on_segment(df, strategy_fn, params):
    """
    Run a strategy on a data segment and return (returns, signal) series.
    Signal is the raw (un-lagged) position signal for cost/num_trades calculation.
    """
    result_df = strategy_fn(df.copy(), **params)
    returns = result_df["Strategy_Returns"].dropna()
    signal = result_df.get("Signal")
    return returns, signal


def run_backtest(strategy_name, symbol, params, walkforward=True, data_dir=None,
                 metrics_since=None, cv_folds=None, embargo_days=21, cost_bps=None):
    """
    Run full backtest with optional walk-forward validation.
    data_dir: if set, load cached CSV from this dir instead of calling yfinance.
    metrics_since: if set (YYYY-MM-DD string or datetime), compute since_metrics
                   over rows with index >= that date (full data, no splitting).
    cost_bps: optional per-run cost override (None → module default COST_BPS).
    """
    if data_dir:
        df = load_cached_data(symbol, data_dir)
    else:
        print(f"Fetching data for {symbol}...", file=sys.stderr)
        df = fetch_data(symbol)

    if df.empty:
        return {"error": f"No data for {symbol}"}

    strategy_fn = STRATEGIES.get(strategy_name)
    if strategy_fn is None:
        return {"error": f"Unknown strategy: {strategy_name}"}

    if not walkforward:
        # Full-sample backtest
        print(f"Running {strategy_name} strategy (full sample)...", file=sys.stderr)
        returns, signal = run_strategy_on_segment(df, strategy_fn, params)
        returns_net = apply_trading_cost(returns, signal, cost_bps=cost_bps)
        metrics = calculate_metrics(returns_net, signal, cost_bps=cost_bps)
        metrics["strategy"] = strategy_name
        metrics["params"] = params
        metrics["symbol"] = symbol
        metrics["data_points"] = len(df)
        metrics["date_range"] = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
        metrics["walkforward"] = {"enabled": False}
        # Optional: compute metrics over rows with index >= metrics_since
        if metrics_since is not None:
            since_dt = pd.Timestamp(metrics_since, tz=df.index.tz if hasattr(df.index, 'tz') else None)
            since_df = df[df.index >= since_dt]
            if len(since_df) == 0:
                metrics["since_metrics"] = {"n_days": 0}
            else:
                since_returns, since_signal = run_strategy_on_segment(since_df, strategy_fn, params)
                since_returns_net = apply_trading_cost(since_returns, since_signal, cost_bps=cost_bps)
                since_m = compute_split_metrics(since_returns_net, since_signal, len(since_df), cost_bps=cost_bps)
                # Rename num_days to n_days for spec compliance
                since_m["n_days"] = since_m.pop("num_days")
                metrics["since_metrics"] = since_m
        return metrics

    # Walk-forward: 60% train / 20% validation / 20% holdout
    print(f"Running {strategy_name} strategy (walk-forward 60/20/20)...", file=sys.stderr)
    train_df, val_df, holdout_df = split_walkforward(df)

    # In-sample (train)
    is_returns, is_signal = run_strategy_on_segment(train_df, strategy_fn, params)
    is_returns_net = apply_trading_cost(is_returns, is_signal, cost_bps=cost_bps)
    is_metrics = compute_split_metrics(is_returns_net, is_signal, len(train_df), cost_bps=cost_bps)

    # Validation (out-of-sample)
    val_returns, val_signal = run_strategy_on_segment(val_df, strategy_fn, params)
    val_returns_net = apply_trading_cost(val_returns, val_signal, cost_bps=cost_bps)
    val_metrics = compute_split_metrics(val_returns_net, val_signal, len(val_df), cost_bps=cost_bps)

    # Holdout
    holdout_returns, holdout_signal = run_strategy_on_segment(holdout_df, strategy_fn, params)
    holdout_returns_net = apply_trading_cost(holdout_returns, holdout_signal, cost_bps=cost_bps)
    holdout_metrics = compute_split_metrics(holdout_returns_net, holdout_signal, len(holdout_df), cost_bps=cost_bps)

    result = {
        "strategy": strategy_name,
        "params": params,
        "symbol": symbol,
        "data_points": len(df),
        "date_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        "in_sample": is_metrics,
        "out_of_sample": val_metrics,
        "holdout": holdout_metrics,
        "walkforward": {"enabled": True, "train_ratio": 0.6, "val_ratio": 0.2, "holdout_ratio": 0.2},
    }
    # --- CV folds (opt-in, additive — does NOT modify v1 output) ---
    if cv_folds is not None:
        try:  # package import (pytest) / script import (container flat layout)
            from .splitter import WalkForwardCV
        except ImportError:
            from splitter import WalkForwardCV
        cv = WalkForwardCV(
            n_folds=cv_folds, embargo_days=embargo_days,
            train_frac=0.6, val_frac=0.2,
        )
        folds = cv.split(df)

        # Holdout: identical across folds — compute once
        holdout_cv_df = folds[0][2]
        holdout_cv_ret, holdout_cv_sig = run_strategy_on_segment(
            holdout_cv_df, strategy_fn, params)
        holdout_cv_net = apply_trading_cost(holdout_cv_ret, holdout_cv_sig, cost_bps=cost_bps)
        holdout_cv_metrics = compute_split_metrics(
            holdout_cv_net, holdout_cv_sig, len(holdout_cv_df), cost_bps=cost_bps)
        holdout_cv_ret_list = [float(v) for v in holdout_cv_net.values]
        holdout_cv_dates = [d.strftime('%Y-%m-%d') for d in holdout_cv_net.index]

        cv_folds_data = []
        for k, (train_cv_df, val_cv_df, _) in enumerate(folds):
            # Train metrics
            tr_ret, tr_sig = run_strategy_on_segment(
                train_cv_df, strategy_fn, params)
            tr_net = apply_trading_cost(tr_ret, tr_sig, cost_bps=cost_bps)
            train_m = compute_split_metrics(tr_net, tr_sig, len(train_cv_df), cost_bps=cost_bps)

            # Val metrics + raw net daily returns
            vr_ret, vr_sig = run_strategy_on_segment(
                val_cv_df, strategy_fn, params)
            vr_net = apply_trading_cost(vr_ret, vr_sig, cost_bps=cost_bps)
            val_m = compute_split_metrics(vr_net, vr_sig, len(val_cv_df), cost_bps=cost_bps)
            val_ret_list = [float(v) for v in vr_net.values]
            val_date_list = [d.strftime('%Y-%m-%d') for d in vr_net.index]

            cv_folds_data.append({
                "fold": k,
                "n_train": len(train_cv_df),
                "n_val": len(val_ret_list),
                "train_metrics": train_m,
                "val_metrics": val_m,
                "val_daily_returns": val_ret_list,
                "val_dates": val_date_list,
            })

        result["cv"] = {
            "config": {
                "n_folds": cv_folds,
                "embargo_days": embargo_days,
                "train_frac": 0.6,
                "val_frac": 0.2,
            },
            "folds": cv_folds_data,
            "holdout": {
                "n_days": len(holdout_cv_ret_list),
                "metrics": holdout_cv_metrics,
                "daily_returns": holdout_cv_ret_list,
                "dates": holdout_cv_dates,
            },
        }

    # Optional: compute metrics over rows with index >= metrics_since
    if metrics_since is not None:
        since_dt = pd.Timestamp(metrics_since, tz=df.index.tz if hasattr(df.index, 'tz') else None)
        since_df = df[df.index >= since_dt]
        if len(since_df) == 0:
            result["since_metrics"] = {"n_days": 0}
        else:
            since_returns, since_signal = run_strategy_on_segment(since_df, strategy_fn, params)
            since_returns_net = apply_trading_cost(since_returns, since_signal, cost_bps=cost_bps)
            since_m = compute_split_metrics(since_returns_net, since_signal, len(since_df), cost_bps=cost_bps)
            # Rename num_days to n_days for spec compliance
            since_m["n_days"] = since_m.pop("num_days")
            result["since_metrics"] = since_m
    return result


if __name__ == "__main__":

    def _validate_cv_folds(value):
        ivalue = int(value)
        if ivalue < 2 or ivalue > 5:
            raise argparse.ArgumentTypeError(
                f"--cv-folds must be in [2, 5], got {ivalue}")
        return ivalue

    def _validate_embargo_days(value):
        ivalue = int(value)
        if ivalue < 0 or ivalue > 60:
            raise argparse.ArgumentTypeError(
                f"--embargo-days must be in [0, 60], got {ivalue}")
        return ivalue

    def _validate_cost_bps(value):
        fvalue = float(value)
        # Reject bool-ish (True/False cast to 1.0/0.0), NaN, inf
        if isinstance(value, bool) or fvalue != fvalue or fvalue in (float("inf"), float("-inf")):
            raise argparse.ArgumentTypeError(
                f"--cost-bps must be a real number in [0.0, 100.0], got {value!r}")
        if fvalue < 0.0 or fvalue > 100.0:
            raise argparse.ArgumentTypeError(
                f"--cost-bps must be in [0.0, 100.0], got {fvalue}")
        return fvalue

    parser = argparse.ArgumentParser(description="Backtest Engine")
    parser.add_argument("--strategy", default=None, help="Strategy name")
    parser.add_argument("--symbol", required=True, help="Ticker symbol")
    parser.add_argument("--params", default="{}", help="JSON string of strategy parameters")
    parser.add_argument(
        "--no-walkforward", action="store_true", help="Disable walk-forward (full sample only)"
    )
    parser.add_argument(
        "--walkforward", dest="walkforward", action="store_true", default=True,
        help="Enable walk-forward validation (default)"
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Fetch 5y daily OHLCV and save to --data-dir, then exit"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory for cached CSV files (used with --fetch-only or as data source)"
    )
    parser.add_argument(
        "--metrics-since", default=None,
        help="YYYY-MM-DD date: compute since_metrics over rows with index >= this date"
    )
    parser.add_argument(
        "--cv-folds", type=_validate_cv_folds, default=None,
        help="Number of CV folds [2-5] for walk-forward CV (opt-in)"
    )
    parser.add_argument(
        "--embargo-days", type=_validate_embargo_days, default=21,
        help="Embargo gap in trading days between train and val [0-60] (default: 21)"
    )
    parser.add_argument(
        "--cost-bps", type=_validate_cost_bps, default=None,
        help="Per-run trading cost override in bps [0.0-100.0] (default: module COST_BPS=5.0)"
    )
    args = parser.parse_args()

    # --- fetch-only mode: no strategy execution ---
    if args.fetch_only:
        if not args.data_dir:
            print(json.dumps({"error": "--fetch-only requires --data-dir DIR"}))
            sys.exit(1)
        info = fetch_and_cache(args.symbol, args.data_dir)
        print(json.dumps(info, default=str))
        sys.exit(0)

    # --- strategy mode: require --strategy ---
    if not args.strategy:
        parser.error("--strategy is required (or use --fetch-only)")

    params = json.loads(args.params)
    walkforward = not args.no_walkforward
    result = run_backtest(
        args.strategy, args.symbol, params,
        walkforward=walkforward,
        data_dir=args.data_dir,
        metrics_since=args.metrics_since,
        cv_folds=args.cv_folds,
        embargo_days=args.embargo_days,
        cost_bps=args.cost_bps,
    )
    print(json.dumps(result, indent=2, default=str))
