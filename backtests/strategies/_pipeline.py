"""Shared position -> returns pipeline (trusted code).

Strategy modules compute positions only. Converting a position series into
daily strategy returns is identical across strategies and lives here exactly
once, mirroring the split used for LLM-generated strategies (strategy code
emits signals; the trusted harness computes returns/costs/metrics).
"""


def attach_returns(df, position_col):
    """Append Returns and Strategy_Returns columns derived from position_col.

    The position is lagged by one day (trade at next close) — identical to the
    original per-strategy implementations.
    """
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df[position_col].shift(1) * df["Returns"]
    return df
