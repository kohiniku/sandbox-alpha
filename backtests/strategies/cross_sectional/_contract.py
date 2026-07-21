"""Cross-sectional strategy contract: validators for weights, signals, and scores.

All validators operate on wide DataFrames with shape (date, symbol).
Engine (PR 4c) calls these validators before portfolio construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_WEIGHT_SUM_TOLERANCE = 1e-6


def validate_weights(weights: pd.DataFrame, universe: list[str]) -> None:
    """Validate a strategy weight matrix.

    Rules
    -----
    * Index is a sorted DatetimeIndex
    * Columns are a subset of ``universe`` (extra symbols → ValueError)
    * Values are finite numeric (no NaN)
    * Per-row weights sum to ~1.0 (long-only) or ~0.0 (long-short)

    Raises ValueError on the first violation.
    """
    # ── index ──────────────────────────────────────────────────────────
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise ValueError(
            f"weights.index must be DatetimeIndex, got {type(weights.index).__name__}"
        )
    if not weights.index.is_monotonic_increasing:
        raise ValueError("weights.index must be sorted (monotonic increasing)")

    # ── columns ────────────────────────────────────────────────────────
    universe_set = set(universe)
    extra = sorted(set(weights.columns) - universe_set)
    if extra:
        raise ValueError(
            f"weights columns contain symbols not in universe: {extra}. "
            f"Universe: {sorted(universe_set)}"
        )

    # ── values ─────────────────────────────────────────────────────────
    if not weights.empty:
        if not np.issubdtype(weights.values.dtype, np.floating):
            raise ValueError("weights must contain numeric (float) values")
        if weights.isnull().any().any():
            raise ValueError("weights must not contain NaN")

    # ── row sums ───────────────────────────────────────────────────────
    if weights.empty:
        return

    row_sums = weights.sum(axis=1)
    for idx, s in row_sums.items():
        if not np.isfinite(s):
            raise ValueError(f"Row sum at {idx} is not finite: {s}")

        # Long-only (sum == 1)?
        if abs(s - 1.0) <= _WEIGHT_SUM_TOLERANCE:
            continue  # ok

        # Long-short (sum == 0)?
        if abs(s) <= _WEIGHT_SUM_TOLERANCE:
            continue  # ok

        raise ValueError(
            f"Row sum at {idx} is {s:.8f}. "
            f"Weights must sum to ~1.0 (long-only) or ~0.0 (long-short) "
            f"within tolerance {_WEIGHT_SUM_TOLERANCE}"
        )


def validate_signals(signals: pd.DataFrame, universe: list[str]) -> None:
    """Validate a strategy signal matrix.

    Rules
    -----
    * Index is a sorted DatetimeIndex
    * Columns are a subset of ``universe``
    * Values ∈ {-1, 0, 1} only (no NaN)

    Raises ValueError on the first violation.
    """
    # ── index ──────────────────────────────────────────────────────────
    if not isinstance(signals.index, pd.DatetimeIndex):
        raise ValueError(
            f"signals.index must be DatetimeIndex, got {type(signals.index).__name__}"
        )
    if not signals.index.is_monotonic_increasing:
        raise ValueError("signals.index must be sorted (monotonic increasing)")

    # ── columns ────────────────────────────────────────────────────────
    universe_set = set(universe)
    extra = sorted(set(signals.columns) - universe_set)
    if extra:
        raise ValueError(
            f"signals columns contain symbols not in universe: {extra}"
        )

    # ── values ─────────────────────────────────────────────────────────
    if signals.empty:
        return

    vals = signals.values
    if vals.dtype == np.bool_:
        vals = vals.astype(np.int8)
    if not np.issubdtype(vals.dtype, np.integer) and not np.issubdtype(
        vals.dtype, np.floating
    ):
        raise ValueError(
            f"signals must contain numeric values, got {vals.dtype}"
        )

    valid = (vals == -1) | (vals == 0) | (vals == 1)
    if not valid.all():
        invalid_count = (~valid).sum()
        # Find first offending position for a helpful error
        bad_mask = ~valid
        bad_rows, bad_cols = np.where(bad_mask)
        first_bad_row = signals.index[bad_rows[0]]
        first_bad_col = signals.columns[bad_cols[0]]
        first_bad_val = vals[bad_rows[0], bad_cols[0]]
        raise ValueError(
            f"signals must contain only {{-1, 0, 1}}. "
            f"Found {invalid_count} value(s) outside domain. "
            f"First: [{first_bad_row}, {first_bad_col}] = {first_bad_val}"
        )


def validate_scores(scores: pd.DataFrame, universe: list[str]) -> None:
    """Validate a strategy score matrix.

    Rules
    -----
    * Index is a sorted DatetimeIndex
    * Columns are a subset of ``universe``
    * Values are finite numeric (range unconstrained — engine z-scores downstream)

    Raises ValueError on the first violation.
    """
    # ── index ──────────────────────────────────────────────────────────
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise ValueError(
            f"scores.index must be DatetimeIndex, got {type(scores.index).__name__}"
        )
    if not scores.index.is_monotonic_increasing:
        raise ValueError("scores.index must be sorted (monotonic increasing)")

    # ── columns ────────────────────────────────────────────────────────
    universe_set = set(universe)
    extra = sorted(set(scores.columns) - universe_set)
    if extra:
        raise ValueError(
            f"scores columns contain symbols not in universe: {extra}"
        )

    # ── values ─────────────────────────────────────────────────────────
    if scores.empty:
        return

    vals = scores.values
    if not np.issubdtype(vals.dtype, np.floating) and not np.issubdtype(
        vals.dtype, np.integer
    ):
        raise ValueError(
            f"scores must contain numeric values, got {vals.dtype}"
        )

    # Reject NaN
    if pd.isna(scores).any().any():
        nan_count = pd.isna(scores).sum().sum()
        raise ValueError(f"scores must not contain NaN. Found {nan_count} NaN(s).")

    # Reject non-finite (inf)
    if not np.isfinite(vals).all():
        raise ValueError("scores must contain only finite values (no inf)")
