"""Built-in portfolio metrics registered on Evaluator."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from evaluators.base import Evaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRADING_DAYS = 252
_MIN_ROWS = 20


def _portfolio_returns(
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame],
) -> pd.Series:
    """Compute portfolio daily returns.

    If *weights* is None, assume equal weight across all columns.
    Weights are aligned to returns index.
    """
    if weights is None:
        return returns.mean(axis=1)
    # Align weights to returns index
    w = weights.reindex(returns.index).ffill().fillna(0.0)
    # Element-wise multiply then sum across columns
    return (returns * w.reindex(columns=returns.columns, fill_value=0.0)).sum(axis=1)


# ---------------------------------------------------------------------------
# Registered metrics
# ---------------------------------------------------------------------------


@Evaluator.register("sharpe")
def sharpe(
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame],
    benchmark: Optional[pd.Series],
    config: dict,
) -> float:
    """Annualized Sharpe ratio of portfolio daily returns."""
    pr = _portfolio_returns(returns, weights)
    if len(pr) < _MIN_ROWS:
        raise ValueError(f"sharpe requires >= {_MIN_ROWS} rows, got {len(pr)}")
    mu = pr.mean()
    sigma = pr.std(ddof=1)
    if sigma < 1e-15 or np.isnan(sigma):
        return np.nan
    return float(mu / sigma * np.sqrt(_TRADING_DAYS))


@Evaluator.register("ir")
def ir(
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame],
    benchmark: Optional[pd.Series],
    config: dict,
) -> float:
    """Information ratio vs benchmark."""
    if benchmark is None:
        return np.nan
    pr = _portfolio_returns(returns, weights)
    bm = benchmark.reindex(pr.index).fillna(0.0)
    active = pr - bm
    if len(active) < _MIN_ROWS:
        raise ValueError(f"ir requires >= {_MIN_ROWS} rows, got {len(active)}")
    mu = active.mean()
    sigma = active.std(ddof=1)
    if sigma < 1e-15 or np.isnan(sigma):
        return np.nan
    return float(mu / sigma * np.sqrt(_TRADING_DAYS))


@Evaluator.register("turnover")
def turnover(
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame],
    benchmark: Optional[pd.Series],
    config: dict,
) -> float:
    """Annualized portfolio turnover.

    sum(|w_t - w_{t-1}|) / T * 252.  Zero if weights=None.
    """
    if weights is None:
        return 0.0
    w = weights.reindex(returns.index).ffill().fillna(0.0)
    if len(w) < 2:
        return 0.0
    diff = w.diff().abs().sum(axis=1)
    # Exclude the first NaN row
    diff = diff.iloc[1:]
    T = len(diff)
    if T == 0:
        return 0.0
    return float(diff.sum() / T * _TRADING_DAYS)


@Evaluator.register("cvar_95")
def cvar_95(
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame],
    benchmark: Optional[pd.Series],
    config: dict,
) -> float:
    """Conditional VaR at 95%: -mean of worst 5% of daily portfolio returns.

    Positive number = expected shortfall.
    """
    pr = _portfolio_returns(returns, weights)
    if len(pr) < _MIN_ROWS:
        raise ValueError(f"cvar_95 requires >= {_MIN_ROWS} rows, got {len(pr)}")
    cutoff = int(max(1, np.floor(len(pr) * 0.05)))
    worst = pr.nsmallest(cutoff)
    return float(-worst.mean())


@Evaluator.register("max_drawdown")
def max_drawdown(
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame],
    benchmark: Optional[pd.Series],
    config: dict,
) -> float:
    """Worst peak-to-trough of cumulative return series (percent, e.g. -18.5)."""
    pr = _portfolio_returns(returns, weights)
    if len(pr) < _MIN_ROWS:
        raise ValueError(f"max_drawdown requires >= {_MIN_ROWS} rows, got {len(pr)}")
    cum = (1.0 + pr).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return float(drawdown.min() * 100.0)


@Evaluator.register("total_return")
def total_return(
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame],
    benchmark: Optional[pd.Series],
    config: dict,
) -> float:
    """Cumulative product of (1+r) - 1, in percent."""
    pr = _portfolio_returns(returns, weights)
    if len(pr) < _MIN_ROWS:
        raise ValueError(f"total_return requires >= {_MIN_ROWS} rows, got {len(pr)}")
    cum = (1.0 + pr).prod() - 1.0
    return float(cum * 100.0)


@Evaluator.register("factor_exposure")
def factor_exposure(
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame],
    benchmark: Optional[pd.Series],
    config: dict,
) -> Dict[str, float]:
    """OLS regression of portfolio returns on factor matrix.

    config['factors'] should be a DataFrame of factor returns aligned to
    portfolio index.  Returns expanded dict like
    {factor_exposure_alpha, factor_exposure_MKT, ...}.
    If factors not provided, all values are NaN.
    """
    pr = _portfolio_returns(returns, weights)
    factors = config.get("factors")
    if factors is None or not isinstance(factors, pd.DataFrame) or factors.empty:
        # Return a single NaN key; dispatch will handle expansion
        return {"factor_exposure_alpha": np.nan}

    if len(pr) < _MIN_ROWS:
        raise ValueError(f"factor_exposure requires >= {_MIN_ROWS} rows, got {len(pr)}")

    # Align
    common_idx = pr.index.intersection(factors.index)
    y = pr.loc[common_idx].values
    X = factors.loc[common_idx].values

    # Add constant column for alpha
    X_with_const = np.column_stack([np.ones(len(y)), X])

    # OLS via least-squares
    result, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)

    factor_names = list(factors.columns)
    out: Dict[str, float] = {"factor_exposure_alpha": float(result[0])}
    for i, name in enumerate(factor_names):
        out[f"factor_exposure_{name}"] = float(result[i + 1])
    return out
