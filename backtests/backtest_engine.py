#!/usr/bin/env python3
"""
Minimal backtest engine for sandbox testing
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import sys

def fetch_data(symbol, period="2y"):
    """Fetch historical data"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    return df

def calculate_metrics(returns):
    """Calculate performance metrics"""
    if len(returns) == 0:
        return {"error": "No trades executed"}
    
    total_return = (1 + returns).prod() - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    
    return {
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "num_trades": len(returns),
        "avg_daily_return_pct": round(returns.mean() * 100, 4)
    }

def run_sma_crossover_strategy(df, fast_window=10, slow_window=30):
    """SMA Crossover Strategy"""
    df['SMA_Fast'] = df['Close'].rolling(window=fast_window).mean()
    df['SMA_Slow'] = df['Close'].rolling(window=slow_window).mean()
    
    df['Signal'] = 0
    df.loc[df['SMA_Fast'] > df['SMA_Slow'], 'Signal'] = 1
    df.loc[df['SMA_Fast'] < df['SMA_Slow'], 'Signal'] = -1
    
    df['Position'] = df['Signal'].diff()
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    
    strategy_returns = df['Strategy_Returns'].dropna()
    return calculate_metrics(strategy_returns)

def run_mean_reversion_strategy(df, window=20, threshold=2.0):
    """Mean Reversion Strategy"""
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['Std'] = df['Close'].rolling(window=window).std()
    df['Z_Score'] = (df['Close'] - df['SMA']) / df['Std']
    
    df['Signal'] = 0
    df.loc[df['Z_Score'] < -threshold, 'Signal'] = 1
    df.loc[df['Z_Score'] > threshold, 'Signal'] = -1
    
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    
    strategy_returns = df['Strategy_Returns'].dropna()
    return calculate_metrics(strategy_returns)

def run_momentum_strategy(df, lookback=20, hold_period=5):
    """Momentum Strategy"""
    df['Momentum'] = df['Close'].pct_change(lookback)
    
    df['Signal'] = 0
    df.loc[df['Momentum'] > 0, 'Signal'] = 1
    df.loc[df['Momentum'] < 0, 'Signal'] = -1
    
    df['Position'] = df['Signal'].rolling(window=hold_period).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    
    strategy_returns = df['Strategy_Returns'].dropna()
    return calculate_metrics(strategy_returns)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python backtest_engine.py <strategy_name> <symbol> [params...]")
        sys.exit(1)
    
    strategy_name = sys.argv[1]
    symbol = sys.argv[2]
    
    print(f"Fetching data for {symbol}...", file=sys.stderr)
    df = fetch_data(symbol)
    
    if df.empty:
        print(json.dumps({"error": f"No data for {symbol}"}))
        sys.exit(1)
    
    print(f"Running {strategy_name} strategy...", file=sys.stderr)
    
    if strategy_name == "sma_crossover":
        fast = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        slow = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        results = run_sma_crossover_strategy(df, fast, slow)
        results['strategy'] = 'sma_crossover'
        results['params'] = {'fast_window': fast, 'slow_window': slow}
    
    elif strategy_name == "mean_reversion":
        window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
        results = run_mean_reversion_strategy(df, window, threshold)
        results['strategy'] = 'mean_reversion'
        results['params'] = {'window': window, 'threshold': threshold}
    
    elif strategy_name == "momentum":
        lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        hold = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        results = run_momentum_strategy(df, lookback, hold)
        results['strategy'] = 'momentum'
        results['params'] = {'lookback': lookback, 'hold_period': hold}
    
    else:
        print(json.dumps({"error": f"Unknown strategy: {strategy_name}"}))
        sys.exit(1)
    
    results['symbol'] = symbol
    results['data_points'] = len(df)
    results['date_range'] = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
    
    print(json.dumps(results, indent=2))
