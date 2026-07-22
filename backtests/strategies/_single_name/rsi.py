"""RSI mean reversion using Wilder's original smoothing formula."""

import numpy as np

NAME = "rsi"


def compute_signal(df, rsi_window=14, oversold=30, overbought=70):
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

    return df, "Signal"
