#!/usr/bin/env python3
"""
Minimal backtest engine for sandbox testing
Usage: python backtest_engine.py --strategy sma_crossover --symbol AAPL --params '{"fast_window": 10, "slow_window": 30}'
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# 取引コスト: 片道 bps（デフォルト 5.0 = 0.05%）
COST_BPS = 5.0


def fetch_data(symbol, period="2y"):
    """Fetch historical data"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    return df


def split_walkforward(df, train_ratio=0.7):
    """Split data into in-sample (train) and out-of-sample (test)"""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def apply_trading_cost(strategy_returns, signal):
    """Apply per-trade cost: COST_BPS bps per side on each position change"""
    if len(strategy_returns) == 0 or len(signal) == 0:
        return strategy_returns
    cost_frac = COST_BPS / 10000.0  # bps -> decimal
    # Align signal with returns index (signal is lagged by 1)
    sig_aligned = signal.shift(1).reindex(strategy_returns.index)
    # Position changes (entry + exit = 2 sides per change)
    pos_changes = sig_aligned.diff().abs().fillna(0)
    cost_series = pos_changes * cost_frac
    # Only apply cost on days where we actually have returns
    cost_aligned = cost_series.reindex(strategy_returns.index, fill_value=0)
    return strategy_returns - cost_aligned


def calculate_metrics(returns, signal=None):
    """
    Calculate performance metrics.
    - num_trades: counts position changes, not trading days
    - drawdown: equity-curve (1+r).cumprod() based, compound-consistent with total_return
    """
    if len(returns) == 0:
        return {"error": "No trades executed"}

    # Compound total return
    equity_curve = (1 + returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1

    # Sharpe ratio (annualized)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown from equity curve (compound-based)
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = drawdowns.min()

    # num_trades: count actual position changes (not trading days)
    if signal is not None and len(signal) > 0:
        sig_changes = signal.diff().abs().fillna(0)
        num_trades = int((sig_changes > 0).sum())
    else:
        num_trades = len(returns)

    return {
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "num_trades": num_trades,
        "avg_daily_return_pct": round(returns.mean() * 100, 4),
        "cost_bps": COST_BPS,
    }


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


def run_backtest(strategy_name, symbol, params, walkforward=True):
    """
    Run full backtest with optional walk-forward validation.
    Returns dict with IS and OOS metrics.
    """
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

    # Walk-forward: 70% train / 30% test
    print(f"Running {strategy_name} strategy (walk-forward 70/30)...", file=sys.stderr)
    train_df, test_df = split_walkforward(df)

    # In-sample
    is_returns, is_signal = run_strategy_on_segment(train_df, strategy_fn, params)
    is_returns_net = apply_trading_cost(is_returns, is_signal)
    is_metrics = calculate_metrics(is_returns_net, is_signal)

    # Out-of-sample (same params, unseen data)
    oos_returns, oos_signal = run_strategy_on_segment(test_df, strategy_fn, params)
    oos_returns_net = apply_trading_cost(oos_returns, oos_signal)
    oos_metrics = calculate_metrics(oos_returns_net, oos_signal)

    result = {
        "strategy": strategy_name,
        "params": params,
        "symbol": symbol,
        "data_points": len(df),
        "date_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        "in_sample": is_metrics,
        "out_of_sample": oos_metrics,
        "walkforward": {"enabled": True, "train_ratio": 0.7},
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Engine")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--symbol", required=True, help="Ticker symbol")
    parser.add_argument("--params", default="{}", help="JSON string of strategy parameters")
    parser.add_argument(
        "--no-walkforward", action="store_true", help="Disable walk-forward (full sample only)"
    )
    parser.add_argument(
        "--walkforward", dest="walkforward", action="store_true", default=True,
        help="Enable walk-forward validation (default)"
    )
    args = parser.parse_args()

    params = json.loads(args.params)

    # Backward compat: also accept positional-style args for direct CLI use
    # e.g. python backtest_engine.py sma_crossover AAPL 10 30
    # If --strategy not explicitly set, try positional fallback
    # (argparse handles --strategy as required, so this is just for
    #  the case where someone pipes the old positional format)
    if not params:
        # Maybe positional args were appended after --?
        pass

    walkforward = not args.no_walkforward
    result = run_backtest(args.strategy, args.symbol, params, walkforward=walkforward)
    print(json.dumps(result, indent=2, default=str))
