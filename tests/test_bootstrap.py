"""
Tests for backtests/bootstrap.py — BootstrapLCB.

Coverage test (#8) is the honesty check: if the bootstrap produces
LCBs that aren't actually lower bounds, this test fails.
"""
import math

import numpy as np
import pandas as pd
import pytest

from backtests.bootstrap import BootstrapLCB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annualized_sharpe(returns: pd.Series) -> float:
    """Point-estimate annualized Sharpe."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(252))


def _generate_series(n: int, mu: float, sigma: float, seed: int = 42) -> pd.Series:
    """Generate i.i.d. normal daily returns."""
    rng = np.random.default_rng(seed)
    data = rng.normal(mu, sigma, size=n)
    return pd.Series(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lcb_below_point_estimate():
    """For a series with positive mean, LCB(5%) < point Sharpe."""
    returns = _generate_series(500, mu=0.001, sigma=0.01, seed=1)
    point = _annualized_sharpe(returns)
    lcb = BootstrapLCB.compute(returns, n_resample=2000, alpha=0.05, seed=42)
    assert lcb < point, f"Expected LCB {lcb:.4f} < point Sharpe {point:.4f}"


def test_lcb_equal_zero_when_zero_variance():
    """Constant series → LCB = 0.0 (no signal, no variance)."""
    returns = pd.Series(np.ones(100) * 0.001)
    lcb = BootstrapLCB.compute(returns, seed=42)
    assert lcb == 0.0, f"Expected 0.0, got {lcb}"


def test_lcb_negative_when_series_is_random_noise():
    """For np.random.normal(0, 0.01, 500), LCB should be < 0."""
    returns = _generate_series(500, mu=0.0, sigma=0.01, seed=2)
    lcb = BootstrapLCB.compute(returns, n_resample=2000, alpha=0.05, seed=42)
    assert lcb < 0, f"Expected negative LCB for noise, got {lcb:.4f}"


def test_lcb_stable_across_seeds():
    """Two seeds should give LCBs within 0.15 of each other for B=2000."""
    returns = _generate_series(500, mu=0.001, sigma=0.01, seed=3)
    lcb1 = BootstrapLCB.compute(returns, n_resample=2000, alpha=0.05, seed=10)
    lcb2 = BootstrapLCB.compute(returns, n_resample=2000, alpha=0.05, seed=20)
    assert abs(lcb1 - lcb2) < 0.15, (
        f"LCBs differ by {abs(lcb1 - lcb2):.4f} — "
        f"seed 10: {lcb1:.4f}, seed 20: {lcb2:.4f}"
    )


def test_block_len_default():
    """default_block_len returns max(21, int(sqrt(n)))."""
    assert BootstrapLCB.default_block_len(400) == 21  # sqrt(400)=20, floored to 21
    assert BootstrapLCB.default_block_len(1000) == 31  # sqrt(1000)=31(.62)
    assert BootstrapLCB.default_block_len(100) == 21  # sqrt(100)=10, floored


def test_empty_series_raises():
    """Empty series raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        BootstrapLCB.compute(pd.Series([], dtype=float))


def test_short_series_clips_block_len():
    """Series of length 15 with block_len=21 → clipped to 15, no crash."""
    returns = _generate_series(15, mu=0.001, sigma=0.01, seed=4)
    lcb = BootstrapLCB.compute(returns, block_len=21, n_resample=2000, alpha=0.05, seed=42)
    # Just checking it doesn't crash; result doesn't matter
    assert isinstance(lcb, float)


def test_coverage_property_on_known_process():
    """CRITICAL honesty test.

    Generate 200 synthetic AR(0) return series with true Sharpe = 1.5,
    compute 5% LCB for each.  At least 90% of the 200 LCBs must be
    below the true Sharpe of 1.5 (i.e., LCB is genuinely a lower bound).
    """
    # Theoretical Sharpe = 1.5 ⇒ mu/sigma * sqrt(252) = 1.5
    # With sigma = 0.01: mu = 1.5 * 0.01 / sqrt(252)
    TRUE_SHARPE = 1.5
    sigma = 0.01
    mu = TRUE_SHARPE * sigma / math.sqrt(252)  # ~0.000945

    n_series = 200
    n_days = 500
    master_seed = 12345
    rng = np.random.default_rng(master_seed)

    n_below = 0
    for i in range(n_series):
        data = rng.normal(mu, sigma, size=n_days)
        returns = pd.Series(data)
        lcb = BootstrapLCB.compute(returns, n_resample=2000, alpha=0.05, seed=master_seed + i)
        if lcb < TRUE_SHARPE:
            n_below += 1

    coverage = n_below / n_series
    assert coverage >= 0.90, (
        f"Coverage: {n_below}/{n_series} = {coverage:.1%} < 90%. "
        f"Bootstrap LCB is not an honest lower bound!"
    )
