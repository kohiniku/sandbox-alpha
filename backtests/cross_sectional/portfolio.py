"""Portfolio construction methods — stateless, pure weight math.

All methods are @staticmethod on PortfolioBuilder.  Every method operates
on one row at a time (cross-section) and returns per-row weights that sum
to 1 (long-only) or 0 (long-short).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioBuilder:
    """Stateless portfolio construction — no instance state, pure functions."""

    @staticmethod
    def top_k_weights(
        scores: pd.DataFrame, k: int, long_only: bool = True
    ) -> pd.DataFrame:
        """Per row: pick top-k symbols by score → equal weight 1/k long.

        If long_only=False, additionally short bottom-k at -1/k.

        Parameters
        ----------
        scores : DataFrame (date × symbol) of numeric scores
        k : int — how many symbols in each leg
        long_only : bool — if False, long top-k + short bottom-k (net 0)

        Returns
        -------
        DataFrame of weights, same shape as scores.  Rows sum to 1
        (long_only) or 0 (long-short).  NaN scores → weight 0.
        """
        result = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

        if long_only:
            for idx in scores.index:
                row = scores.loc[idx]
                valid = row.dropna()
                if len(valid) == 0:
                    continue
                top = valid.nlargest(min(k, len(valid)))
                result.loc[idx, top.index] = 1.0 / len(top)
        else:
            for idx in scores.index:
                row = scores.loc[idx]
                valid = row.dropna()
                if len(valid) < 2 * k:
                    # Need at least 2k symbols for meaningful LS
                    if len(valid) >= k:
                        # At least long side possible
                        top = valid.nlargest(min(k, len(valid)))
                        result.loc[idx, top.index] = 1.0 / len(top)
                    continue
                top = valid.nlargest(k)
                bottom = valid.nsmallest(k)
                result.loc[idx, top.index] = 1.0 / k
                result.loc[idx, bottom.index] = -1.0 / k

        return result

    @staticmethod
    def quintile_ls_weights(
        scores: pd.DataFrame, quintiles: int = 5
    ) -> pd.DataFrame:
        """Per row: top quintile = equal-weight long (sum=1),
        bottom quintile = equal-weight short (sum=-1), net 0.

        Parameters
        ----------
        scores : DataFrame (date × symbol)
        quintiles : int — number of quantile bins (default 5)

        Returns
        -------
        DataFrame of weights.  NaN scores → weight 0.
        """
        result = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        inv_q = 1.0 / quintiles

        for idx in scores.index:
            row = scores.loc[idx]
            valid = row.dropna()
            if len(valid) < quintiles:
                continue
            ranked = valid.rank(pct=True)
            # Top quintile: score pct >= (1 - 1/quintiles)
            top_mask = ranked > (1.0 - inv_q)
            # Bottom quintile: score pct <= 1/quintiles
            bottom_mask = ranked <= inv_q

            n_top = top_mask.sum()
            n_bottom = bottom_mask.sum()

            if n_top > 0:
                result.loc[idx, top_mask[top_mask].index] = 1.0 / n_top
            if n_bottom > 0:
                result.loc[idx, bottom_mask[bottom_mask].index] = -1.0 / n_bottom

        return result

    @staticmethod
    def zscore_continuous_weights(
        scores: pd.DataFrame,
        threshold: float = 0.0,
        long_only: bool = True,
    ) -> pd.DataFrame:
        """Per row: z-score the scores, clip values below `threshold` to zero,
        normalize survivors to sum to 1 (long-only) or normalize positive/negative
        sides separately to sum ±1 (long-short).

        Parameters
        ----------
        scores : DataFrame (date × symbol)
        threshold : float — minimum z-score to keep (below → 0)
        long_only : bool — if True, all non-zero weights are positive summing to 1.
                     If False, convert z-scores to signed weights (±1 per leg).

        Returns
        -------
        DataFrame of weights.
        """
        result = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

        for idx in scores.index:
            row = scores.loc[idx]
            valid = row.dropna()
            if len(valid) < 2:
                continue
            z = (valid - valid.mean()) / max(valid.std(ddof=0), 1e-10)
            # Clip below threshold
            clipped = z.where(z >= threshold, 0.0)

            if long_only:
                pos_sum = clipped[clipped > 0].sum()
                if pos_sum > 0:
                    result.loc[idx, clipped.index] = (
                        clipped.where(clipped > 0, 0.0) / pos_sum
                    )
            else:
                pos = clipped.where(clipped > 0, 0.0)
                neg = clipped.where(clipped < 0, 0.0)
                pos_sum = pos.sum()
                neg_sum = abs(neg.sum())
                if pos_sum > 0:
                    result.loc[idx, pos.index] = pos / pos_sum
                if neg_sum > 0:
                    result.loc[idx, neg.index] = neg / neg_sum

        return result

    @staticmethod
    def signals_to_equal_weights(signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals in {-1, 0, 1} to equal-weight-normalized portfolio weights.

        Per row:
        - Long positions (signal=1): weight = 1/n_active_long
        - Short positions (signal=-1): weight = -1/n_active_short
        - Flat (signal=0): weight = 0
        If all signals are zero, all weights are zero (flat).

        This is the same logic as manifest_runner._signals_to_weights,
        exposed as a standalone method in the canonical engine location.
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for idx in signals.index:
            row = signals.loc[idx]
            n_long = (row == 1).sum()
            n_short = (row == -1).sum()
            if n_long > 0:
                weights.loc[idx, row == 1] = 1.0 / n_long
            if n_short > 0:
                weights.loc[idx, row == -1] = -1.0 / n_short

        return weights

    @staticmethod
    def apply_rebalance_calendar(
        weights: pd.DataFrame, cadence: str
    ) -> pd.DataFrame:
        """Apply rebalance cadence: forward-fill weights between rebalance dates.

        cadence values:
          - "daily"   → identity (return input unchanged)
          - "weekly"  → rebalance every Monday (weekday==0), ffill between
          - "monthly" → rebalance on the first trading day of each month, ffill

        Never looks ahead: weight at time t is decided from data available at t.
        """
        if cadence == "daily":
            return weights.copy()

        if cadence == "weekly":
            mask = weights.index.weekday == 0  # Monday
        elif cadence == "monthly":
            mask = weights.index.to_series().diff().dt.days > 0
            # First day is always a rebalance
            first_loc = weights.index[0]
            if first_loc not in weights.index[mask]:
                mask = weights.index.isin([first_loc]) | mask
            else:
                pass
            # Better approach: rebalance on the first trading day of each month
            month_start = weights.index.to_series().dt.month
            prev_month = month_start.shift(1)
            mask = month_start != prev_month
            mask.iloc[0] = True  # first trading day always rebalances
        else:
            raise ValueError(
                f"Unknown cadence '{cadence}'. Must be daily, weekly, or monthly."
            )

        reb_weights = weights.copy()
        reb_weights[~mask] = np.nan
        reb_weights = reb_weights.ffill().fillna(0.0)
        return reb_weights
