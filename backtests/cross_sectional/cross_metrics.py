"""Cross-sectional portfolio metrics.

Sharpe, IR, turnover, max drawdown, hit rate — plus a composite
metrics_dict that includes both v1-standard keys and cross additions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_sharpe(
    returns: pd.Series, annualization: int = 252
) -> float:
    """Annualized Sharpe ratio.

    Sharpe = mean(returns) / std(returns) × sqrt(annualization).
    Returns 0.0 if std(returns) == 0.
    """
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    sharpe = returns.mean() / std * np.sqrt(annualization)
    return float(0.0 if not np.isfinite(sharpe) else sharpe)


def portfolio_ir(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization: int = 252,
) -> float:
    """Information ratio: mean(active_returns) / std(active_returns) × sqrt(ann).

    active = portfolio - benchmark.
    Returns 0.0 when portfolio == benchmark (active returns are all zero).
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    active = returns - benchmark_returns.reindex(returns.index)
    active = active.dropna()
    if len(active) < 2:
        return 0.0
    std = active.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    ir = active.mean() / std * np.sqrt(annualization)
    return float(0.0 if not np.isfinite(ir) else ir)


def portfolio_turnover(weights: pd.DataFrame) -> float:
    """Mean daily gross turnover (unannualized).

    turnover = mean(|weights.diff().abs().sum(axis=1)|)
    For constant (static) weights, returns 0.0.
    """
    if weights.empty or len(weights) < 2:
        return 0.0
    turnover = weights.diff().abs().sum(axis=1)
    return float(turnover.iloc[1:].mean())


def portfolio_max_drawdown(returns: pd.Series) -> float:
    """Max peak-to-trough drawdown on cumulative returns.

    Returns a negative number (e.g. -0.25 = 25% drawdown).
    If no drawdown, returns 0.0.
    """
    if len(returns) < 2:
        return 0.0
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    dd = drawdowns.min()
    return float(0.0 if dd >= 0 else dd)


def portfolio_hit_rate(returns: pd.Series) -> float:
    """Fraction of positive-return days.

    Returns 0.0 if no returns.
    """
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def metrics_dict(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    weights: pd.DataFrame,
    num_days: int,
) -> dict:
    """Composite metrics block: v1-standard keys + cross-sectional additions.

    Returns dict with:
      sharpe_ratio, total_return, max_drawdown, num_days, num_trades
      ir, turnover, hit_rate
    """
    total_return = float((1 + returns).prod() - 1)
    sharpe = portfolio_sharpe(returns)
    max_dd = portfolio_max_drawdown(returns)
    hit = portfolio_hit_rate(returns)
    turnover_val = portfolio_turnover(weights) if weights is not None else 0.0

    # num_trades: count days where weights changed
    if weights is not None and len(weights) >= 2:
        weight_changes = weights.diff().abs().sum(axis=1)
        num_trades = int((weight_changes > 1e-10).sum())
    else:
        num_trades = len(returns)

    result = {
        "sharpe_ratio": round(float(sharpe), 3),
        "total_return": round(total_return, 6),
        "max_drawdown": round(float(max_dd), 6),
        "num_days": num_days,
        "num_trades": num_trades,
        "ir": 0.0,
        "turnover": round(float(turnover_val), 6),
        "hit_rate": round(float(hit), 4),
    }

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        result["ir"] = round(
            portfolio_ir(returns, benchmark_returns), 3
        )

    return result
