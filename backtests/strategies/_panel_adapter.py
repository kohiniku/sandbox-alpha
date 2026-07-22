"""Adapt a single-name compute_signal into the cross-sectional interface.

Used by the ideation/backtest engine to route single-name strategies through
the cross-sectional path on demand (PR 4e+).
"""

import pandas as pd


def wrap_single_as_cross(compute_signal_fn, name: str = ""):
    """Adapt a single-name compute_signal into a cross_signal function.

    The returned function has signature (panel, universe, extras) and
    yields a wide DataFrame (index=Date, columns=universe) of signals in
    {-1, 0, 1}. The engine treats this as return_type='signals'.
    """
    def _cross(panel, universe, extras=None):
        extras = extras or {}
        signals = {}
        for sym in universe:
            if sym in panel:
                df_out, position_col = compute_signal_fn(
                    panel[sym].copy(), **extras
                )
                signals[sym] = df_out[position_col]
        return pd.DataFrame(signals).reindex(columns=universe)

    _cross.__name__ = f"cross_{name}" if name else "cross_single_adapter"
    return _cross
