"""
Circular block bootstrap utility for computing lower confidence bounds (LCB)
on the annualized Sharpe ratio.

Purely stateless — all methods are @staticmethod.  No instance state.
"""

import math
import numpy as np
import pandas as pd


class BootstrapLCB:
    """Stateless circular block bootstrap for Sharpe LCB computation."""

    @staticmethod
    def default_block_len(n: int) -> int:
        """Block length heuristic: max(21, sqrt(n)).

        21-day floor matches the embargo gap and is conservative for daily
        returns where return autocorrelation is near zero (Politis & White 2004).
        """
        return max(21, int(math.sqrt(n)))

    @staticmethod
    def compute(
        returns: pd.Series,
        block_len: int | None = None,
        n_resample: int = 2000,
        alpha: float = 0.05,
        annualization: int = 252,
        seed: int | None = None,
    ) -> float:
        """Lower α-quantile of the Sharpe distribution from circular block bootstrap.

        Args:
            returns: Daily strategy returns.
            block_len: Block length (default: max(21, sqrt(n))).  Clipped to n
                       if the series is shorter.
            n_resample: Number of bootstrap resamples (B).
            alpha: Significance level for the lower bound (default 0.05 = 5% LCB).
            annualization: Annualization factor (252 for daily).
            seed: RNG seed for reproducibility.

        Returns:
            The α-quantile of the resampled annualized-Sharpe distribution.

        Raises:
            ValueError: If *returns* is empty.
        """
        n = len(returns)
        if n == 0:
            raise ValueError("returns series is empty")
        if np.isclose(returns.std(), 0.0):
            return 0.0

        if block_len is None:
            block_len = BootstrapLCB.default_block_len(n)
        block_len = min(block_len, n)  # clip to series length for short series

        rng = np.random.default_rng(seed)
        n_blocks = int(math.ceil(n / block_len))

        # Vectorized: generate (B, n_blocks) random start positions
        starts = rng.integers(0, n, size=(n_resample, n_blocks))  # (B, n_blocks)

        # Circular indices: (B, n_blocks, block_len)
        offsets = np.arange(block_len)  # (block_len,)
        indices = (starts[:, :, np.newaxis] + offsets) % n  # (B, n_blocks, block_len)

        # Flatten and truncate to n
        flat_indices = indices.reshape(n_resample, -1)[:, :n]  # (B, n)

        # Gather returns and compute per-sample Sharpe
        samples = returns.values[flat_indices]  # (B, n)
        sample_means = samples.mean(axis=1)
        sample_stds = samples.std(axis=1, ddof=1)

        boot_sharpes = np.zeros(n_resample)
        mask = sample_stds > 0
        boot_sharpes[mask] = (
            sample_means[mask] / sample_stds[mask] * np.sqrt(annualization)
        )

        lcb = float(np.quantile(boot_sharpes, alpha))
        return lcb
