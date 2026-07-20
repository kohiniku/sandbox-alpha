"""Bollinger-style mean reversion on rolling z-score of Close."""

NAME = "mean_reversion"


def compute_signal(df, window=20, threshold=2.0):
    """Mean Reversion Strategy"""
    df["SMA"] = df["Close"].rolling(window=window).mean()
    df["Std"] = df["Close"].rolling(window=window).std()
    df["Z_Score"] = (df["Close"] - df["SMA"]) / df["Std"]

    df["Signal"] = 0
    df.loc[df["Z_Score"] < -threshold, "Signal"] = 1
    df.loc[df["Z_Score"] > threshold, "Signal"] = -1

    return df, "Signal"
