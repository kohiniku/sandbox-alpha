"""Tests for cross-sectional portfolio construction (PortfolioBuilder)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Dual-import for container flat-layout compatibility
try:
    from backtests.cross_sectional.portfolio import PortfolioBuilder
except ImportError:
    from cross_sectional.portfolio import PortfolioBuilder  # type: ignore[no-redef]

try:
    from backtests.cross_sectional.costs import apply_transaction_costs
except ImportError:
    from cross_sectional.costs import apply_transaction_costs  # type: ignore[no-redef]


# ── helpers ────────────────────────────────────────────────────────────────

def _make_scores(n_symbols=20, n_days=50, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    return pd.DataFrame(rng.normal(0, 1, (n_days, n_symbols)), index=dates, columns=symbols)


def _make_weights_from_rows(rows, dates=None):
    """Build a weights DataFrame from a list of rows (dicts or arrays)."""
    if dates is None:
        dates = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=dates)


# ═══════════════════════════════════════════════════════════════════════════
# top_k_weights
# ═══════════════════════════════════════════════════════════════════════════


class TestTopKWeights:
    def test_top_k_selects_correct_count_per_row(self):
        """k=5 → 5 non-zero weights per row summing to 1."""
        scores = _make_scores(20, 10)
        k = 5
        w = PortfolioBuilder.top_k_weights(scores, k=k, long_only=True)
        for idx in w.index:
            non_zero = (w.loc[idx] != 0).sum()
            assert non_zero == k, f"Row {idx}: expected {k} non-zero, got {non_zero}"
            assert abs(w.loc[idx].sum() - 1.0) < 1e-10, (
                f"Row {idx}: sum={w.loc[idx].sum()}, expected 1.0"
            )

    def test_top_k_ls_shorts_bottom_k(self):
        """long_only=False, k=5 → 5 positive + 5 negative, sum to 0."""
        scores = _make_scores(30, 10)
        k = 5
        w = PortfolioBuilder.top_k_weights(scores, k=k, long_only=False)
        for idx in w.index:
            pos = (w.loc[idx] > 0).sum()
            neg = (w.loc[idx] < 0).sum()
            assert pos == k, f"Row {idx}: expected {k} positive, got {pos}"
            assert neg == k, f"Row {idx}: expected {k} negative, got {neg}"
            assert abs(w.loc[idx].sum()) < 1e-10, (
                f"Row {idx}: sum={w.loc[idx].sum()}, expected ~0"
            )
            assert abs(w.loc[idx][w.loc[idx] > 0].sum() - 1.0) < 1e-10
            assert abs(w.loc[idx][w.loc[idx] < 0].sum() + 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# quintile_ls_weights
# ═══════════════════════════════════════════════════════════════════════════


class TestQuintileLSWeights:
    def test_quintile_ls_long_top_short_bottom(self):
        """20 symbols → 4 per quintile, top quintile long, bottom quintile short."""
        scores = _make_scores(20, 10, seed=123)
        w = PortfolioBuilder.quintile_ls_weights(scores, quintiles=5)
        # With 20 symbols and 5 quintiles → 4 per quintile
        for idx in w.index:
            pos = (w.loc[idx] > 0).sum()
            neg = (w.loc[idx] < 0).sum()
            assert pos == 4, f"Row {idx}: expected 4 long, got {pos}"
            assert neg == 4, f"Row {idx}: expected 4 short, got {neg}"
            assert abs(w.loc[idx].sum()) < 1e-10, (
                f"Row {idx}: sum should be ~0, got {w.loc[idx].sum()}"
            )
            assert abs(w.loc[idx][w.loc[idx] > 0].sum() - 1.0) < 1e-10
            assert abs(w.loc[idx][w.loc[idx] < 0].sum() + 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# zscore_continuous_weights
# ═══════════════════════════════════════════════════════════════════════════


class TestZscoreContinuousWeights:
    def test_zscore_continuous_zeros_below_threshold(self):
        """threshold=0.5: symbols with z-score < 0.5 get weight 0."""
        # Create scores where some are clearly below average
        dates = pd.bdate_range("2024-01-02", periods=3)
        # Row where S00 and S01 are high, S02-S04 are low
        scores = pd.DataFrame(
            [
                [10.0, 9.0, 0.0, -1.0, -2.0],
                [10.0, 9.0, 0.0, -1.0, -2.0],
                [10.0, 9.0, 0.0, -1.0, -2.0],
            ],
            index=dates,
            columns=["S00", "S01", "S02", "S03", "S04"],
        )
        w = PortfolioBuilder.zscore_continuous_weights(scores, threshold=0.5, long_only=True)

        for idx in w.index:
            # S00 and S01 should have non-zero weight (z-score well above 0.5)
            assert w.loc[idx, "S00"] > 0
            assert w.loc[idx, "S01"] > 0
            # S02-S04 should be zero (z-score < 0.5)
            assert w.loc[idx, "S02"] == 0.0
            assert w.loc[idx, "S03"] == 0.0
            assert w.loc[idx, "S04"] == 0.0
            assert abs(w.loc[idx].sum() - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# apply_rebalance_calendar
# ═══════════════════════════════════════════════════════════════════════════


class TestRebalanceCalendar:
    def test_rebalance_monthly_freezes_weights_between_month_ends(self):
        """Weekly weights → monthly rebalance → same weight within the month."""
        dates = pd.bdate_range("2024-01-02", periods=60)
        # Create weights that change every day
        w = pd.DataFrame(
            np.arange(60 * 3).reshape(60, 3),
            index=dates,
            columns=["A", "B", "C"],
            dtype=float,
        )
        reb = PortfolioBuilder.apply_rebalance_calendar(w, cadence="monthly")

        # Within each month, weights should be constant (same as first day of that month)
        months = reb.index.to_series().dt.month
        for month in months.unique():
            month_slice = reb[months == month]
            first_row = month_slice.iloc[0]
            for i in range(len(month_slice)):
                assert (month_slice.iloc[i] == first_row).all(), (
                    f"Month {month}: weights changed within month at position {i}"
                )

    def test_rebalance_weekly_freezes_between_mondays(self):
        """Daily weights → weekly rebalance → same weight Monday-Friday."""
        dates = pd.bdate_range("2024-01-08", periods=15)  # starts on Monday
        w = pd.DataFrame(
            np.arange(15 * 2).reshape(15, 2),
            index=dates,
            columns=["A", "B"],
            dtype=float,
        )
        reb = PortfolioBuilder.apply_rebalance_calendar(w, cadence="weekly")

        # Group by week (ISO week number)
        weeks = reb.index.isocalendar().week
        for week in weeks.unique():
            week_slice = reb[weeks == week]
            first_row = week_slice.iloc[0]
            for i in range(len(week_slice)):
                assert (week_slice.iloc[i] == first_row).all(), (
                    f"Week {week}: weights changed within week at position {i}"
                )

    def test_rebalance_daily_is_identity(self):
        """Daily cadence returns input unchanged."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        w = pd.DataFrame(
            np.random.RandomState(42).randn(10, 3),
            index=dates,
            columns=["A", "B", "C"],
        )
        reb = PortfolioBuilder.apply_rebalance_calendar(w, cadence="daily")
        pd.testing.assert_frame_equal(w, reb)


# ═══════════════════════════════════════════════════════════════════════════
# Transaction costs
# ═══════════════════════════════════════════════════════════════════════════


class TestTransactionCosts:
    def test_transaction_costs_zero_on_first_row(self):
        """Initial weights have no prior → first row cost = 0."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        w = pd.DataFrame(
            np.random.RandomState(42).randn(10, 5),
            index=dates,
            columns=["A", "B", "C", "D", "E"],
        )
        costs = apply_transaction_costs(w, cost_bps_map=5.0)
        assert costs.iloc[0] == 0.0, f"First row cost should be 0, got {costs.iloc[0]}"
        # Later rows should have costs (non-zero turnover)
        assert (costs.iloc[1:] >= 0).all(), "Costs should be non-negative"

    def test_transaction_costs_equals_bps_times_turnover(self):
        """Known weights delta → cost = turnover × bps / 10000."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        w = pd.DataFrame(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            index=dates,
            columns=["A", "B"],
        )
        costs = apply_transaction_costs(w, cost_bps_map=5.0)

        # Row 1 (index 1): turnover = |0.5-1.0| + |0.5-0.0| = 1.0
        expected_t1 = 1.0 * 5.0 / 10000.0
        assert abs(costs.iloc[1] - expected_t1) < 1e-12, (
            f"Expected {expected_t1}, got {costs.iloc[1]}"
        )

        # Row 2 (index 2): turnover = |0.0-0.5| + |1.0-0.5| = 1.0
        expected_t2 = 1.0 * 5.0 / 10000.0
        assert abs(costs.iloc[2] - expected_t2) < 1e-12, (
            f"Expected {expected_t2}, got {costs.iloc[2]}"
        )

    def test_transaction_costs_per_symbol_map(self):
        """Different bps per symbol → cost = weighted sum of turnovers."""
        dates = pd.bdate_range("2024-01-02", periods=3)
        w = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.4, 0.3, 0.3]],
            index=dates,
            columns=["A", "B", "C"],
        )
        # A costs 10 bps, B costs 5 bps, C costs 1 bps
        costs = apply_transaction_costs(
            w, cost_bps_map={"A": 10.0, "B": 5.0, "C": 1.0}, default_bps=5.0
        )

        # Row 1: turnover = |0.5-1.0|=0.5 for A, |0.5-0.0|=0.5 for B, 0 for C
        expected = (0.5 * 10.0 + 0.5 * 5.0 + 0.0 * 1.0) / 10000.0
        assert abs(costs.iloc[1] - expected) < 1e-12, (
            f"Expected {expected}, got {costs.iloc[1]}"
        )

        # Row 2: turnover = |0.4-0.5|=0.1 for A, |0.3-0.5|=0.2 for B, |0.3-0.0|=0.3 for C
        expected2 = (0.1 * 10.0 + 0.2 * 5.0 + 0.3 * 1.0) / 10000.0
        assert abs(costs.iloc[2] - expected2) < 1e-12, (
            f"Expected {expected2}, got {costs.iloc[2]}"
        )
