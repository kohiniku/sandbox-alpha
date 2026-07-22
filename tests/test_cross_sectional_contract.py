"""Tests for cross-sectional strategy contract validators.

Covers validate_weights, validate_signals, validate_scores, and the
xs_momentum reference strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Dual-import for container flat-layout compatibility
try:
    from backtests.strategies.cross_sectional._contract import (
        validate_weights,
        validate_signals,
        validate_scores,
    )
except ImportError:
    from cross_sectional._contract import (
        validate_weights,
        validate_signals,
        validate_scores,
    )

try:
    from backtests.strategies.cross_sectional.xs_momentum import compute_cross_signal
except ImportError:
    from cross_sectional.xs_momentum import compute_cross_signal


# ── helpers ────────────────────────────────────────────────────────────────

UNIVERSE = ["AAPL", "MSFT", "GOOGL"]

_dates = pd.bdate_range("2024-01-01", periods=10)
_dates_dt = pd.DatetimeIndex(_dates)


def _mk_weights(values, dates=None, symbols=None):
    if dates is None:
        dates = _dates_dt
    if symbols is None:
        symbols = UNIVERSE
    return pd.DataFrame(values, index=dates[:len(values)], columns=symbols, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# validate_weights
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateWeights:
    def test_accepts_long_only_summing_to_one(self):
        w = _mk_weights([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        validate_weights(w, UNIVERSE)  # no raise

    def test_accepts_long_short_summing_to_zero(self):
        w = _mk_weights([[0.3, -0.1, -0.2], [0.5, 0.2, -0.7]])
        for s in w.sum(axis=1):
            assert abs(s) < 1e-6
        validate_weights(w, UNIVERSE)  # no raise

    def test_rejects_row_summing_to_bogus_number(self):
        w = _mk_weights([[0.5, 0.2, 0.2]])  # sums to 0.9
        with pytest.raises(ValueError, match=r"Row sum.*0\.9"):
            validate_weights(w, UNIVERSE)

    def test_rejects_columns_outside_universe(self):
        w = _mk_weights([[0.3, 0.7]], symbols=["AAPL", "TSLA"])
        with pytest.raises(ValueError, match="universe"):
            validate_weights(w, UNIVERSE)

    def test_rejects_non_datetime_index(self):
        w = pd.DataFrame({"AAPL": [1.0]}, index=[1, 2, 3])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_weights(w, UNIVERSE)

    def test_rejects_nan_values(self):
        w = _mk_weights([[0.5, np.nan, 0.5]])
        with pytest.raises(ValueError, match="NaN"):
            validate_weights(w, UNIVERSE)

    def test_accepts_empty_weights(self):
        w = pd.DataFrame(columns=UNIVERSE, dtype=float)
        w.index = pd.DatetimeIndex([])
        validate_weights(w, UNIVERSE)  # no raise


# ═══════════════════════════════════════════════════════════════════════════
# validate_signals
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateSignals:
    def test_accepts_ternary_values(self):
        s = _mk_weights([[1, 0, -1], [-1, 1, 0]], symbols=UNIVERSE)
        validate_signals(s, UNIVERSE)  # no raise

    def test_rejects_out_of_domain_values(self):
        s = _mk_weights([[1, 0.5, -1]], symbols=UNIVERSE)
        with pytest.raises(ValueError, match=r"\{-1, 0, 1\}"):
            validate_signals(s, UNIVERSE)

    def test_rejects_value_2(self):
        s = _mk_weights([[1, 2, -1]], symbols=UNIVERSE)
        with pytest.raises(ValueError, match=r"\{-1, 0, 1\}"):
            validate_signals(s, UNIVERSE)

    def test_rejects_nan_values(self):
        s = _mk_weights([[1, np.nan, -1]], symbols=UNIVERSE)
        with pytest.raises(ValueError, match=r"\{-1, 0, 1\}"):
            validate_signals(s, UNIVERSE)

    def test_rejects_non_datetime_index(self):
        s = pd.DataFrame({"AAPL": [1]}, index=[1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_signals(s, UNIVERSE)

    def test_rejects_columns_outside_universe(self):
        s = _mk_weights([[1, 0, -1]], symbols=["AAPL", "TSLA", "GOOGL"])
        with pytest.raises(ValueError, match="universe"):
            validate_signals(s, UNIVERSE)

    def test_accepts_empty_signals(self):
        s = pd.DataFrame(columns=UNIVERSE, dtype=float)
        s.index = pd.DatetimeIndex([])
        validate_signals(s, UNIVERSE)  # no raise


# ═══════════════════════════════════════════════════════════════════════════
# validate_scores
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateScores:
    def test_accepts_arbitrary_numeric_range(self):
        # Scores can be any finite numeric — engine z-scores downstream
        s = _mk_weights([[1.5, -3.2, 0.0], [100.0, -50.0, 0.001]])
        validate_scores(s, UNIVERSE)  # no raise

    def test_rejects_nan_values(self):
        s = _mk_weights([[1.0, np.nan, 2.0]])
        with pytest.raises(ValueError, match="NaN"):
            validate_scores(s, UNIVERSE)

    def test_rejects_inf_values(self):
        s = _mk_weights([[1.0, np.inf, 2.0]])
        with pytest.raises(ValueError, match="finite"):
            validate_scores(s, UNIVERSE)

    def test_rejects_non_datetime_index(self):
        s = pd.DataFrame({"AAPL": [1.0]}, index=[1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_scores(s, UNIVERSE)

    def test_accepts_integer_scores(self):
        s = _mk_weights([[1, 2, 3]], symbols=UNIVERSE)
        s = s.astype(int)
        validate_scores(s, UNIVERSE)  # no raise

    def test_accepts_empty_scores(self):
        s = pd.DataFrame(columns=UNIVERSE, dtype=float)
        s.index = pd.DatetimeIndex([])
        validate_scores(s, UNIVERSE)  # no raise


# ═══════════════════════════════════════════════════════════════════════════
# xs_momentum reference strategy
# ═══════════════════════════════════════════════════════════════════════════


class TestXsMomentumReference:
    def test_returns_scores_of_correct_shape(self):
        """Synthetic 3-symbol panel, 300 days.

        Output: DataFrame with DatetimeIndex, columns = universe symbols.
        First ~252 rows are NaN (insufficient lookback for 12-month momentum);
        later rows contain finite score values.
        """
        n_days = 300
        rng = np.random.default_rng(42)
        idx = pd.bdate_range(end="2026-07-01", periods=n_days)

        panel = {}
        for sym in UNIVERSE:
            rets = rng.normal(0.0, 0.015, size=n_days)
            close = 100 * (1 + rets).cumprod()
            panel[sym] = pd.DataFrame({"Close": close}, index=idx)

        result = compute_cross_signal(panel, UNIVERSE)

        # Shape
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert list(result.columns) == UNIVERSE
        assert len(result) == n_days

        # First 252 rows should be all NaN (12-month lookback window)
        assert result.iloc[:252].isna().all().all(), (
            "First 252 rows must be NaN (insufficient lookback for 12-month "
            "momentum). Got non-NaN values."
        )

        # Later rows should have finite values
        assert result.shape[0] > 252, "Need >252 rows to test post-lookback values"
        assert result.iloc[252:].notna().any().any(), (
            "Expected at least some non-NaN values after lookback period."
        )

        # Passes scores validator
        validate_scores(result.dropna(how="all"), UNIVERSE)
