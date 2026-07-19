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
import numpy as np
import pandas as pd
from datetime import datetime

from .metrics import (
    COST_BPS,
    load_cached_data,
    split_walkforward,
    apply_trading_cost,
    calculate_metrics,
    compute_split_metrics,
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


def run_sma_crossover_strategy(df, fast_window=10, slow_window=30):
    """SMA Crossover Strategy"""
    df["SMA_Fast"] = df["Close"].rolling(window=fast_window).mean()
    df["SMA_Slow"] = df["Close"].rolling(window=slow_window).mean()

    df["Signal"] = 0
    df.loc[df["SMA_Fast"] > df["SMA_Slow"], "Signal"] = 1
    df.loc[df["SMA_Fast"] < df["SMA_Slow"], "Signal"] = -1

    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]

    return df


def run_mean_reversion_strategy(df, window=20, threshold=2.0):
    """Mean Reversion Strategy"""
    df["SMA"] = df["Close"].rolling(window=window).mean()
    df["Std"] = df["Close"].rolling(window=window).std()
    df["Z_Score"] = (df["Close"] - df["SMA"]) / df["Std"]

    df["Signal"] = 0
    df.loc[df["Z_Score"] < -threshold, "Signal"] = 1
    df.loc[df["Z_Score"] > threshold, "Signal"] = -1

    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]

    return df


def run_momentum_strategy(df, lookback=20, hold_period=5):
    """Momentum Strategy"""
    df["Momentum"] = df["Close"].pct_change(lookback)

    df["Signal"] = 0
    df.loc[df["Momentum"] > 0, "Signal"] = 1
    df.loc[df["Momentum"] < 0, "Signal"] = -1

    df["Position"] = df["Signal"].rolling(window=hold_period).mean()
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Position"].shift(1) * df["Returns"]

    return df


def run_rsi_strategy(df, rsi_window=14, oversold=30, overbought=70):
    """
    RSI Mean-Reversion Strategy
    Implements Wilder's smoothing method for RSI (the original Welles Wilder formula).
    Logic: long when RSI < oversold (mean-reversion: buy oversold),
           short when RSI > overbought (sell overbought).
    """
    # Step 1: price changes
    delta = df["Close"].diff()

    # Step 2: separate gains and losses
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Step 3: Wilder smoothing — first value is SMA, subsequent values use exponential smoothing
    avg_gain = gain.copy()
    avg_loss = loss.copy()

    # Initial SMA for the first window
    avg_gain.iloc[rsi_window] = gain.iloc[1:rsi_window+1].mean()
    avg_loss.iloc[rsi_window] = loss.iloc[1:rsi_window+1].mean()

    # Wilder's smoothing for the rest
    for i in range(rsi_window + 1, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (rsi_window - 1) + gain.iloc[i]) / rsi_window
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (rsi_window - 1) + loss.iloc[i]) / rsi_window

    # Step 4: RS and RSI
    # Avoid division by zero: where avg_loss is 0, RSI = 100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    df["RSI"] = df["RSI"].fillna(100.0)  # avg_loss=0 → RSI=100

    # Step 5: generate signals
    df["Signal"] = 0
    df.loc[df["RSI"] < oversold, "Signal"] = 1    # oversold → buy
    df.loc[df["RSI"] > overbought, "Signal"] = -1  # overbought → sell

    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Signal"].shift(1) * df["Returns"]

    return df


STRATEGIES = {
    "sma_crossover": run_sma_crossover_strategy,
    "mean_reversion": run_mean_reversion_strategy,
    "momentum": run_momentum_strategy,
    "rsi": run_rsi_strategy,
}


def run_backtest(strategy_name, symbol, params, walkforward=True, data_dir=None):
    """
    Run full backtest with optional walk-forward validation.
    data_dir: if set, load cached CSV from this dir instead of calling yfinance.
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
        returns_net = apply_trading_cost(returns, signal)
        metrics = calculate_metrics(returns_net, signal)
        metrics["strategy"] = strategy_name
        metrics["params"] = params
        metrics["symbol"] = symbol
        metrics["data_points"] = len(df)
        metrics["date_range"] = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
        metrics["walkforward"] = {"enabled": False}
        return metrics

    # Walk-forward: 60% train / 20% validation / 20% holdout
    print(f"Running {strategy_name} strategy (walk-forward 60/20/20)...", file=sys.stderr)
    train_df, val_df, holdout_df = split_walkforward(df)

    # In-sample (train)
    is_returns, is_signal = run_strategy_on_segment(train_df, strategy_fn, params)
    is_returns_net = apply_trading_cost(is_returns, is_signal)
    is_metrics = compute_split_metrics(is_returns_net, is_signal, len(train_df))

    # Validation (out-of-sample)
    val_returns, val_signal = run_strategy_on_segment(val_df, strategy_fn, params)
    val_returns_net = apply_trading_cost(val_returns, val_signal)
    val_metrics = compute_split_metrics(val_returns_net, val_signal, len(val_df))

    # Holdout
    holdout_returns, holdout_signal = run_strategy_on_segment(holdout_df, strategy_fn, params)
    holdout_returns_net = apply_trading_cost(holdout_returns, holdout_signal)
    holdout_metrics = compute_split_metrics(holdout_returns_net, holdout_signal, len(holdout_df))

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
    return result


if __name__ == "__main__":
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
    )
    print(json.dumps(result, indent=2, default=str))
