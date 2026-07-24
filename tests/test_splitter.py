"""Tests for walk-forward CV splitter (backtests/splitter.py)."""
import pandas as pd
import pytest
import numpy as np
from backtests.splitter import WalkForwardCV


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(n_rows: int, with_datetime: bool = True) -> pd.DataFrame:
    """Return a DataFrame with *n_rows* rows and dummy columns."""
    if with_datetime:
        index = pd.date_range("2020-01-01", periods=n_rows, freq="B", tz="UTC")
    else:
        index = pd.RangeIndex(n_rows)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "open": rng.normal(100, 10, n_rows),
            "high": rng.normal(102, 10, n_rows),
            "low": rng.normal(98, 10, n_rows),
            "close": rng.normal(101, 10, n_rows),
            "volume": rng.integers(1000, 10000, n_rows),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Test 1: default construction yields exactly 3 folds
# ---------------------------------------------------------------------------

def test_default_construction_yields_three_folds():
    cv = WalkForwardCV()
    df = _make_synthetic_df(1260)
    folds = cv.split(df)
    assert len(folds) == 3


# ---------------------------------------------------------------------------
# Test 2: holdout identical across folds
# ---------------------------------------------------------------------------

def test_holdout_identical_across_folds():
    cv = WalkForwardCV()
    df = _make_synthetic_df(1260)
    folds = cv.split(df)
    for k in range(1, len(folds)):
        pd.testing.assert_frame_equal(folds[0][2], folds[k][2])


# ---------------------------------------------------------------------------
# Test 3: val slices are non-overlapping
# ---------------------------------------------------------------------------

def test_val_slices_non_overlapping():
    cv = WalkForwardCV()
    df = _make_synthetic_df(1260)
    folds = cv.split(df)
    seen_indices: set = set()
    for _, val_df, _ in folds:
        val_positions = {df.index.get_loc(idx) for idx in val_df.index}
        assert seen_indices.isdisjoint(val_positions), (
            f"Val overlap detected at positions: {val_positions & seen_indices}"
        )
        seen_indices |= val_positions


# ---------------------------------------------------------------------------
# Test 4: train expands monotonically
# ---------------------------------------------------------------------------

def test_train_expands_monotonically():
    cv = WalkForwardCV()
    df = _make_synthetic_df(1260)
    folds = cv.split(df)
    prev_len = len(folds[0][0])
    for k in range(1, len(folds)):
        cur_len = len(folds[k][0])
        assert cur_len > prev_len, (
            f"Fold {k} train ({cur_len}) not larger than fold {k-1} ({prev_len})"
        )
        prev_len = cur_len


# ---------------------------------------------------------------------------
# Test 5: embargo gap enforced
# ---------------------------------------------------------------------------

def test_embargo_gap_enforced():
    cv = WalkForwardCV(embargo_days=21)
    df = _make_synthetic_df(1260)
    folds = cv.split(df)
    for train_df, val_df, _ in folds:
        train_last_iloc = df.index.get_loc(train_df.index[-1])
        val_first_iloc = df.index.get_loc(val_df.index[0])
        gap = val_first_iloc - train_last_iloc
        assert gap >= 21, (
            f"Embargo gap too small: {gap} rows (need >= 21). "
            f"train ends at iloc {train_last_iloc}, val starts at iloc {val_first_iloc}"
        )


# ---------------------------------------------------------------------------
# Test 6: no leak into holdout
# ---------------------------------------------------------------------------

def test_no_leak_into_holdout():
    cv = WalkForwardCV()
    df = _make_synthetic_df(1260)
    folds = cv.split(df)

    # Holdout must start at R + embargo_days
    T = len(df)
    R = int(T * (cv.train_frac + cv.val_frac))
    expected_holdout_start = R + cv.embargo_days
    first_holdout_iloc = df.index.get_loc(folds[0][2].index[0])
    assert first_holdout_iloc == expected_holdout_start, (
        f"Holdout starts at iloc {first_holdout_iloc}, "
        f"expected {expected_holdout_start} (R={R} + embargo={cv.embargo_days})"
    )

    holdout_indices = set(folds[0][2].index)
    for train_df, val_df, _ in folds:
        train_in_holdout = set(train_df.index) & holdout_indices
        val_in_holdout = set(val_df.index) & holdout_indices
        assert not train_in_holdout, f"Train leak into holdout: {train_in_holdout}"
        assert not val_in_holdout, f"Val leak into holdout: {val_in_holdout}"


# ---------------------------------------------------------------------------
# Test 7: raises ValueError when data too small
# ---------------------------------------------------------------------------

def test_raises_when_too_small():
    cv = WalkForwardCV(n_folds=3, embargo_days=21)
    df = _make_synthetic_df(30)
    with pytest.raises(ValueError, match="No rows left for holdout"):
        cv.split(df)


# ---------------------------------------------------------------------------
# Test 8: fold_dates populated for DatetimeIndex
# ---------------------------------------------------------------------------

def test_fold_dates_populated_for_datetime_index():
    cv = WalkForwardCV()
    df = _make_synthetic_df(1260, with_datetime=True)
    cv.split(df)
    dates = cv.fold_dates
    assert len(dates) == 3
    for train_end, val_start, val_end in dates:
        assert isinstance(train_end, pd.Timestamp)
        assert isinstance(val_start, pd.Timestamp)
        assert isinstance(val_end, pd.Timestamp)


# ---------------------------------------------------------------------------
# Test 9: 1260-row synthetic matches the plan's numeric example
# ---------------------------------------------------------------------------

def test_1260_row_synthetic_matches_plan():
    """
    Verify the concrete numeric layout from the docstring / plan document.

    T=1260, defaults (n_folds=3, embargo=21, train_frac=0.6, val_frac=0.2):

        R = int(1260 * 0.8) = 1008
        holdout = df.iloc[1029:1260] (231 rows)
        val_region = [604:1008] split into [134, 135, 135]

        Fold 0: train[0:583],  val[604:738]
        Fold 1: train[0:717],  val[738:873]
        Fold 2: train[0:852],  val[873:1008]
    """
    cv = WalkForwardCV()
    df = _make_synthetic_df(1260)
    folds = cv.split(df)

    # Holdout
    assert len(folds[0][2]) == 231

    # Fold 0
    assert len(folds[0][0]) == 583
    assert len(folds[0][1]) == 134

    # Fold 1
    assert len(folds[1][0]) == 717
    assert len(folds[1][1]) == 135

    # Fold 2
    assert len(folds[2][0]) == 852
    assert len(folds[2][1]) == 135

    # Holdout starts at R + embargo_days = 1008 + 21 = 1029
    expected_holdout = df.iloc[1029:]
    pd.testing.assert_frame_equal(folds[0][2], expected_holdout)


# ---------------------------------------------------------------------------
# Test 10: embargo gap between last val slice and holdout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_datetime", [True, False])
def test_last_val_to_holdout_embargo_gap(with_datetime):
    """The gap between the last fold's val_end and holdout start must be >= embargo_days."""
    cv = WalkForwardCV(embargo_days=21, n_folds=3)
    df = _make_synthetic_df(1260, with_datetime=with_datetime)
    folds = cv.split(df)

    # Last fold's val slice ends at val_slices[-1][1]
    val_end_iloc = df.index.get_loc(folds[-1][1].index[-1]) + 1  # exclusive end
    holdout_start_iloc = df.index.get_loc(folds[0][2].index[0])
    gap = holdout_start_iloc - val_end_iloc

    assert gap >= cv.embargo_days, (
        f"Gap between last val end (iloc {val_end_iloc}) and holdout start "
        f"(iloc {holdout_start_iloc}) is {gap}, need >= {cv.embargo_days}"
    )


# ---------------------------------------------------------------------------
# Test 11: ValueError when embargo gap leaves no holdout rows
# ---------------------------------------------------------------------------

def test_holdout_embargo_leaves_no_rows():
    """R + embargo_days >= T should raise ValueError with a clear message."""
    cv = WalkForwardCV(n_folds=1, embargo_days=5)
    # T=15 → R=int(15*0.8)=12, R+5=17 >= 15 → no holdout rows
    df = _make_synthetic_df(15)
    with pytest.raises(ValueError, match="No rows left for holdout after embargo gap"):
        cv.split(df)
