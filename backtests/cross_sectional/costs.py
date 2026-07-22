"""Transaction cost model for cross-sectional portfolio.

Computes per-date cost as bps × turnover, with per-symbol or uniform
cost rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_transaction_costs(
    weights: pd.DataFrame,
    cost_bps_map: dict | float = 5.0,
    default_bps: float = 5.0,
) -> pd.Series:
    """Compute per-date cost series in decimal units.

    Turnover_t  = sum_i |w_{i,t} - w_{i,t-1}|  for i in universe
    Cost_t      = sum_i |w_{i,t} - w_{i,t-1}| × cost_bps_i / 10000

    Parameters
    ----------
    weights : DataFrame (date × symbol)
    cost_bps_map : float | dict[symbol → bps]
        Uniform scalar or per-symbol tiered rates.
    default_bps : float
        Fallback for symbols missing from the map (only used when
        cost_bps_map is a dict).

    Returns
    -------
    Series of per-date costs, index = weights.index.  First row ≈ 0
    (no prior weights to compare).  Values are in decimal units
    (e.g. 5 bps → 0.0005 per unit turnover).
    """
    if weights.empty:
        return pd.Series(0.0, index=weights.index)

    # Build per-symbol bps Series
    if isinstance(cost_bps_map, (int, float)):
        bps = pd.Series(cost_bps_map, index=weights.columns, dtype=float)
    else:
        bps = pd.Series(default_bps, index=weights.columns, dtype=float)
        for sym, val in cost_bps_map.items():
            if sym in bps.index:
                bps[sym] = float(val)

    # Turnover: abs(weight change), sum across symbols
    turnover = weights.diff().abs().fillna(0.0)
    cost_series = (turnover * bps / 10000.0).sum(axis=1)

    # First row cost = 0 explicitly (diff().abs() already makes it 0,
    # but ensure we never accidentally charge entry cost)
    if len(cost_series) > 0:
        cost_series.iloc[0] = 0.0

    return cost_series
