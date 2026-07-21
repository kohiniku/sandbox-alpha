"""
Walk-forward cross-validation splitter with embargo gap.

Supersedes the single-shot split_walkforward() in metrics.py for gate-v2 use.
Provides expanding-window walk-forward CV splits with an explicit trade-day
embargo between train and validation periods.
"""

import pandas as pd


class WalkForwardCV:
    """Expanding-window walk-forward cross-validation splitter with embargo.

    Splits a DataFrame chronologically into *K* folds of (train, val, holdout)
    tuples.  The holdout is the same final segment across all folds and is never
    touched by CV.  Inside the train+val region, each fold expands the training
    window while using a distinct, non-overlapping validation slice.  An embargo
    gap of *embargo_days* trading rows is enforced between the end of train and
    the start of val in every fold.

    Parameters
    ----------
    n_folds : int
        Number of CV folds (default 3).
    embargo_days : int
        Rows to leave as a gap between train end and val start (default 21).
    train_frac : float
        Fraction of total rows used for the expanding train region (default 0.6).
    val_frac : float
        Fraction of total rows partitioned into the validation slices (default 0.2).
        The remaining ``1 - train_frac - val_frac`` becomes the holdout.

    Concrete layout for defaults on a 1260-row DataFrame::

        T = 1260   R = int(T * 0.8) = 1008
        holdout = df.iloc[1008:1260]   (252 rows, identical across folds)

        Val region [604:1008] (404 rows), split into 3 slices (134 + 135 + 135):

        Fold 0:  train[0:583]   gap[583:604]   val[604:738]   (134 val)
        Fold 1:  train[0:717]   gap[717:738]   val[738:873]   (135 val)
        Fold 2:  train[0:852]   gap[852:873]   val[873:1008]  (135 val)
    """

    def __init__(
        self,
        n_folds: int = 3,
        embargo_days: int = 21,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
    ):
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")
        if not (0 < train_frac < 1):
            raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
        if not (0 < val_frac < 1):
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
        if train_frac + val_frac > 1:
            raise ValueError(
                f"train_frac + val_frac must be <= 1, "
                f"got {train_frac} + {val_frac} = {train_frac + val_frac}"
            )

        self.n_folds = n_folds
        self.embargo_days = embargo_days
        self.train_frac = train_frac
        self.val_frac = val_frac
        self._fold_dates: list[tuple] = []

    @property
    def fold_dates(self) -> list[tuple]:
        """Return (train_end, val_start, val_end) for each fold.

        Values are index labels (timestamps for DatetimeIndex, integer
        positions otherwise).  Populated after ``split()`` is called.
        """
        return self._fold_dates

    def split(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Split *df* into K folds of (train, val, holdout).

        Returns
        -------
        list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
            Each element is ``(train_df, val_df, holdout_df)``.  The holdout
            is identical across all folds.
        """
        T = len(df)
        holdout_cut = int(T * (self.train_frac + self.val_frac))
        if holdout_cut >= T:
            holdout_cut = T
        R = holdout_cut  # CV region size

        holdout_df = df.iloc[holdout_cut:]

        # Val slices live inside [val_region_start:R]
        val_region_start = int(R * self.train_frac)
        val_region_size = R - val_region_start

        if val_region_size < self.n_folds:
            raise ValueError(
                f"Data too small for {self.n_folds} folds: "
                f"val region has {val_region_size} rows but needs at least "
                f"{self.n_folds}.  Total rows={T}, CV region={R}, "
                f"val_region=[{val_region_start}:{R}]."
            )

        # Minimum rows needed per fold: at least 1 val row + embargo rows for
        # the first fold (later folds need train + embargo + val).
        min_val_per_fold = 1
        min_train_start = self.embargo_days + 1
        if (
            val_region_size < self.n_folds * min_val_per_fold
            or val_region_start < min_train_start
        ):
            raise ValueError(
                f"Data too small for {self.n_folds} folds with embargo={self.embargo_days}: "
                f"total rows={T}, CV region={R}, "
                f"val_region=[{val_region_start}:{R}] size={val_region_size}. "
                f"Need at least {self.n_folds} val rows and "
                f"train_start >= {min_train_start} rows."
            )

        # Partition the val region into n_folds equal-ish slices
        base_size = val_region_size // self.n_folds
        remainder = val_region_size % self.n_folds
        val_slices: list[tuple[int, int]] = []
        cursor = val_region_start
        for k in range(self.n_folds):
            size = base_size + (1 if k >= self.n_folds - remainder else 0)
            val_slices.append((cursor, cursor + size))
            cursor += size

        folds: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
        self._fold_dates = []
        is_datetime = isinstance(df.index, pd.DatetimeIndex)

        for val_start, val_end in val_slices:
            train_end = val_start - self.embargo_days
            if train_end <= 0:
                raise ValueError(
                    f"Embargo gap too large for fold: "
                    f"val_start={val_start}, embargo={self.embargo_days} "
                    f"would require train_end={train_end} <= 0."
                )

            train_df = df.iloc[:train_end]
            val_df = df.iloc[val_start:val_end]

            folds.append((train_df, val_df, holdout_df))

            if is_datetime:
                self._fold_dates.append((
                    df.index[train_end - 1],
                    df.index[val_start],
                    df.index[val_end - 1],
                ))
            else:
                self._fold_dates.append((
                    train_end - 1,
                    val_start,
                    val_end - 1,
                ))

        return folds
