"""
Tests for gate v2 functions (CV + bootstrap LCB) in autonomous_loop.py.

These functions are additive — they are NOT wired into evaluate_result() yet.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from autonomous_loop import (
    _eval_val_gate_cv,
    _eval_holdout_gate_cv,
    compute_effective_min_sharpe,
    MAX_DRAWDOWN_LIMIT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(mu: float, sigma: float, n: int, seed: int) -> pd.Series:
    """Generate i.i.d. normal daily returns."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, size=n))


def _make_fold_returns(mu: float, sigma: float, n_per_fold: int,
                       n_folds: int = 3, seed: int = 0) -> list:
    """Generate list of per-fold pd.Series."""
    folds = []
    for i in range(n_folds):
        folds.append(_make_returns(mu, sigma, n_per_fold, seed + i))
    return folds


# ---------------------------------------------------------------------------
# Validation gate tests
# ---------------------------------------------------------------------------

def test_val_gate_cv_passes_strong_strategy():
    """Per-fold returns with high true Sharpe → pass, reasons contain '✅' for LCB Sharpe."""
    # High Sharpe: mu=0.002, sigma=0.01 → Sharpe ≈ 3.17
    folds = _make_fold_returns(mu=0.002, sigma=0.01, n_per_fold=130, n_folds=3, seed=100)
    passed, lcb, reasons = _eval_val_gate_cv(folds, N_family=2, n_resample=2000)
    assert passed, f"Should pass, got reasons: {reasons}"
    assert any("✅ LCB Sharpe" in r for r in reasons), f"Missing ✅ for LCB Sharpe: {reasons}"
    assert lcb > 0


def test_val_gate_cv_rejects_noise():
    """Per-fold returns are pure noise → LCB negative → rejected."""
    # Zero-mean noise
    folds = _make_fold_returns(mu=0.0, sigma=0.01, n_per_fold=130, n_folds=3, seed=200)
    passed, lcb, reasons = _eval_val_gate_cv(folds, N_family=2, n_resample=2000)
    assert not passed, f"Should reject noise, got passed with reasons: {reasons}"
    assert any("❌ LCB Sharpe" in r for r in reasons), f"Missing ❌ for LCB Sharpe: {reasons}"


def test_val_gate_cv_rejects_negative_return():
    """High Sharpe but net negative return → rejected on 'Val Return' criterion."""
    # Series with mild negative drift: mu=-0.0005, sigma=0.005
    # Sharpe = -0.0005/0.005 * sqrt(252) = -1.59 (negative)
    # But LCB of negative Sharpe is also negative, so it will fail both LCB and return.
    # Let me construct a case where LCB is positive but total return is negative.
    # Hard to do since positive LCB implies positive mean. Instead, use a series
    # that has strong positive days but ends net negative.
    rng = np.random.default_rng(300)
    # Start with positive returns, then a big crash near the end
    data = np.concatenate([
        rng.normal(0.003, 0.01, size=120),   # strong positive
        rng.normal(-0.05, 0.02, size=10),     # crash
    ])
    # This should have decent Sharpe but negative total return from the crash
    folds = [pd.Series(data[:65]), pd.Series(data[65:])]
    passed, lcb, reasons = _eval_val_gate_cv(folds, N_family=2, n_resample=2000)
    assert not passed, f"Should reject negative return, got: {reasons}"
    assert any("❌ Val Return" in r for r in reasons), (
        f"Expected '❌ Val Return' in reasons: {reasons}"
    )


def test_val_gate_cv_rejects_deep_drawdown():
    """Sharpe passes but drawdown < MAX_DRAWDOWN_LIMIT → rejected on 'Val Drawdown'."""
    # Construct a series with decent mean but a single catastrophic day
    rng = np.random.default_rng(400)
    data = np.concatenate([
        rng.normal(0.002, 0.01, size=120),  # good days
        [-0.40],                             # catastrophic 40% single-day crash
        rng.normal(0.002, 0.01, size=9),
    ])
    folds = [pd.Series(data)]
    passed, lcb, reasons = _eval_val_gate_cv(folds, N_family=2, n_resample=2000)
    assert not passed, f"Should reject deep drawdown, got: {reasons}"
    assert any("❌ Val Drawdown" in r for r in reasons), (
        f"Expected '❌ Val Drawdown' in reasons: {reasons}"
    )


def test_val_gate_cv_uses_total_T_for_deflation():
    """With N_family=100, threshold uses T_val_cv = sum of fold lengths."""
    n_per = 130
    n_folds = 3
    folds = _make_fold_returns(mu=0.001, sigma=0.01,
                               n_per_fold=n_per, n_folds=n_folds, seed=500)
    T_expected = n_per * n_folds  # 390
    threshold_expected = compute_effective_min_sharpe(100, T_expected)

    # The function internally calls compute_effective_min_sharpe(N_family, T_val_cv)
    # where T_val_cv = sum(len(r) for r in folds).  Since our folds each have
    # length T_expected/3, the sum is T_expected.  We verify by checking that
    # the reason string contains the correct T_cv value.
    _, _, reasons = _eval_val_gate_cv(folds, N_family=100, n_resample=2000)
    reason_line = [r for r in reasons if "T_cv" in r][0]
    assert f"T_cv={T_expected}" in reason_line, (
        f"Expected T_cv={T_expected} in reason, got: {reason_line}"
    )


# ---------------------------------------------------------------------------
# Holdout gate tests
# ---------------------------------------------------------------------------

def test_holdout_gate_cv_passes():
    """Strong holdout series + reasonable lcb_val_sharpe → pass."""
    # Use high mu and large n so LCB is reliably > 0.5.
    holdout = _make_returns(mu=0.002, sigma=0.01, n=500, seed=601)
    lcb_val = 1.0  # strong validation LCB
    passed, lcb_ho, reasons = _eval_holdout_gate_cv(holdout, lcb_val,
                                                     n_resample=2000)
    # threshold = min(0.5, 0.5*1.0) = 0.5
    # With mu=0.002, sigma=0.01, n=500, Sharpe ≈ 4.47
    assert passed, f"Should pass, got reasons: {reasons}"


def test_holdout_gate_cv_uses_min_half_threshold():
    """Threshold = min(0.5, 0.5 * lcb_val_sharpe)."""
    holdout = _make_returns(mu=0.002, sigma=0.01, n=200, seed=700)
    lcb_val = 2.0  # so threshold = min(0.5, 1.0) = 0.5
    passed, lcb_ho, reasons = _eval_holdout_gate_cv(holdout, lcb_val,
                                                     n_resample=2000)
    reason_line = [r for r in reasons if "Holdout Sharpe" in r][0]
    # threshold should be 0.5 (capped)
    assert "threshold=min(0.5, 0.5*val)" in reason_line

    # Also test with lcb_val=0.4 → threshold = min(0.5, 0.2) = 0.2
    passed2, _, reasons2 = _eval_holdout_gate_cv(holdout, 0.4, n_resample=2000)
    reason_line2 = [r for r in reasons2 if "Holdout Sharpe" in r][0]
    assert "0.20" in reason_line2  # threshold displayed as 0.20


# ---------------------------------------------------------------------------
# Format consistency
# ---------------------------------------------------------------------------

def test_reasons_format_matches_v1_style():
    """Reason strings use ✅/❌ prefix as v1 does (visual consistency)."""
    folds = _make_fold_returns(mu=0.002, sigma=0.01, n_per_fold=130, n_folds=3, seed=800)
    _, _, reasons = _eval_val_gate_cv(folds, N_family=2, n_resample=2000)

    # Every reason should start with either ✅ or ❌
    for r in reasons:
        assert r.startswith("✅") or r.startswith("❌"), (
            f"Reason does not have prefix: {r!r}"
        )

    # Holdout gate too
    holdout = _make_returns(mu=0.001, sigma=0.01, n=200, seed=900)
    _, _, ho_reasons = _eval_holdout_gate_cv(holdout, 1.0, n_resample=2000)
    for r in ho_reasons:
        assert r.startswith("✅") or r.startswith("❌"), (
            f"Holdout reason does not have prefix: {r!r}"
        )
