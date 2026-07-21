"""
Shared backtest metrics: cost model, data splitting, metric computation.
Trusted code — extracted from backtest_engine.py so both the engine
and the strategy harness can use identical metric logic.
"""
import json
import os
import sys
import numpy as np
import pandas as pd

try:  # package import (pytest) / script import (container flat layout)
    from .splitter import WalkForwardCV  # re-export for gate-v2 CV use
except ImportError:
    from splitter import WalkForwardCV  # noqa: F401

# Trading cost: one-way bps (default 5.0 = 0.05%)
COST_BPS = 5.0


# ---------------------------------------------------------------------------
# Data loading (cached CSV — no network)
# ---------------------------------------------------------------------------

def load_cached_data(symbol, data_dir):
    """Load data from a cached CSV file. No network dependency."""
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.isfile(path):
        print(json.dumps({"error": f"data not cached: {symbol}"}))
        sys.exit(1)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date")
    return df


# ---------------------------------------------------------------------------
# Walk-forward splitting
# ---------------------------------------------------------------------------

def split_walkforward(df, train_ratio=0.6, val_ratio=0.2, holdout_ratio=0.2):
    """Split data into train (in-sample), validation, and holdout segments chronologically.

    Note: WalkForwardCV (imported above) supersedes this single-shot split for
    gate-v2 use, providing expanding-window CV folds with an embargo gap.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ---------------------------------------------------------------------------
# Trading cost model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-split metrics
# ---------------------------------------------------------------------------

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


def compute_split_metrics(returns, signal, num_days):
    """Build per-split metrics dict including num_days."""
    metrics = calculate_metrics(returns, signal)
    metrics["num_days"] = num_days
    return metrics
