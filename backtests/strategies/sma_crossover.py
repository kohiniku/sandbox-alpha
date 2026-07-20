"""SMA crossover: long when fast SMA above slow SMA, short when below."""

NAME = "sma_crossover"


def compute_signal(df, fast_window=10, slow_window=30):
    """SMA Crossover Strategy"""
    df["SMA_Fast"] = df["Close"].rolling(window=fast_window).mean()
    df["SMA_Slow"] = df["Close"].rolling(window=slow_window).mean()

    df["Signal"] = 0
    df.loc[df["SMA_Fast"] > df["SMA_Slow"], "Signal"] = 1
    df.loc[df["SMA_Fast"] < df["SMA_Slow"], "Signal"] = -1

    return df, "Signal"
