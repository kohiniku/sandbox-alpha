"""Time-series momentum with a rolling-average holding position.

Position (not Signal) drives returns: the raw Signal is smoothed over
hold_period days, so the traded position is fractional. Costs and trade
counts are still computed from the raw Signal by the engine.
"""

NAME = "momentum"


def compute_signal(df, lookback=20, hold_period=5):
    """Momentum Strategy"""
    df["Momentum"] = df["Close"].pct_change(lookback)

    df["Signal"] = 0
    df.loc[df["Momentum"] > 0, "Signal"] = 1
    df.loc[df["Momentum"] < 0, "Signal"] = -1

    df["Position"] = df["Signal"].rolling(window=hold_period).mean()

    return df, "Position"
