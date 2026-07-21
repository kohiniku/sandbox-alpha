"""Cross-sectional momentum: reference strategy for contract tests.

Returns SCORES (not weights). Engine (PR 4c) is responsible for z-scoring,
top-k selection, and turnover management.
"""
from __future__ import annotations

import pandas as pd


def compute_cross_signal(
    panel: dict[str, pd.DataFrame],
    universe: list[str],
    extras: dict | None = None,
) -> pd.DataFrame:
    """12-month lookback momentum: close.pct_change(252) per symbol.

    Parameters
    ----------
    panel : dict[str, pd.DataFrame]
        {symbol: OHLCV DataFrame with aligned DateIndex and 'Close' column}
    universe : list[str]
        Ordered symbol list for output column alignment.
    extras : dict | None
        Optional extra data (unused by this strategy).

    Returns
    -------
    pd.DataFrame
        Index = date, columns = symbols (ordered per ``universe``).
        Values are **scores** — raw momentum factor values.
        First 252 rows are NaN (insufficient lookback).

    Notes
    -----
    Returns SCORES (not weights). Engine (PR 4c) z-scores, selects top-k,
    and manages turnover.  This function is intentionally minimal — its
    primary purpose is exercising the contract validators.
    """
    # Collect per-symbol score series
    series = {}
    for sym in universe:
        if sym not in panel:
            continue
        df = panel[sym]
        close = df.get("Close")
        if close is None:
            continue
        series[sym] = close.pct_change(252)  # 12-month momentum

    if not series:
        # All symbols missing — return empty DataFrame with correct structure
        return pd.DataFrame(columns=universe)

    # Build wide DataFrame aligned on date index
    result = pd.DataFrame(series)
    result.index.name = "Date"

    # Reindex to ensure columns match universe order (fill missing with NaN)
    result = result.reindex(columns=universe)

    return result


# Register in cross-sectional strategy dictionary
CROSS_SECTIONAL_STRATEGIES: dict[str, type(compute_cross_signal)] = {}
CROSS_SECTIONAL_STRATEGIES["xs_momentum"] = compute_cross_signal
