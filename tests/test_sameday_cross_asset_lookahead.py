"""Regression tests for issue #65: same-day cross-asset lookahead detection.

The existing causality check truncates data at val_end and compares val metrics.
This catches cross-day lookahead but NOT same-day cross-asset lookahead, where a
strategy uses today's own multi-asset return vector to decide which asset's
same-day return to credit itself (e.g. GMM regime switching over SPY/QQQ/TLT).

These tests verify that the new per-day masking check catches that pattern.
"""

import base64
import json
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest_runner import run_manifest
from manifest import StrategyManifest


# ---------------------------------------------------------------------------
# Helpers (mirrors test_manifest_runner.py conventions)
# ---------------------------------------------------------------------------

def _make_ohlcv_csv(data_dir: str, symbol: str, dates, close_prices, seed=42):
    rng = np.random.RandomState(seed)
    n = len(dates)
    df = pd.DataFrame({
        "Date": dates,
        "Open": close_prices * (1 + rng.uniform(-0.005, 0.005, n)),
        "High": close_prices * (1 + rng.uniform(0.001, 0.02, n)),
        "Low": close_prices * (1 - rng.uniform(0.001, 0.02, n)),
        "Close": close_prices,
        "Volume": rng.randint(100_000, 10_000_000, n),
    })
    df.to_csv(os.path.join(data_dir, f"{symbol}.csv"), index=False)


def _make_manifest(name="test_strat", code="", universe=None,
                   start="2023-01-01", end="2023-12-31",
                   execution_mode="expert"):
    if universe is None:
        universe = ["SPY", "QQQ", "TLT"]
    code_b64 = base64.b64encode(code.encode()).decode()
    payload = {
        "name": name,
        "code_b64": code_b64,
        "data_sources": [
            {"type": "ohlcv", "universe": universe, "start": start, "end": end}
        ],
        "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
        "evaluator": {
            "type": "portfolio",
            "metrics": ["sharpe", "max_drawdown_pct"],
            "benchmark": None,
        },
        "execution_mode": execution_mode,
    }
    return StrategyManifest.from_dict(payload)


def _setup_data(tmp_path, symbols, n_days=200):
    dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")
    rng = np.random.RandomState(42)
    for i, sym in enumerate(symbols):
        prices = 100.0 + np.cumsum(rng.randn(n_days) * 0.5) + i * 10
        prices = np.maximum(prices, 1.0)
        _make_ohlcv_csv(str(tmp_path), sym, dates, prices, seed=i + 42)


# ---------------------------------------------------------------------------
# Strategy code templates
# ---------------------------------------------------------------------------

# Same-day cross-asset lookahead: picks the best asset TODAY using today's
# returns, then credits itself that asset's same-day return.  This is the
# exact pattern from issue #65 (GMM regime switching simplified to argmax).
SAMEDAY_LOOKAHEAD_CODE = textwrap.dedent("""\
    import numpy as np
    import pandas as pd

    def run(data, train_end, val_end, benchmark, config):
        symbols = sorted(k for k in data if not k.startswith("_"))
        closes = pd.DataFrame({s: data[s]["Close"] for s in symbols})
        rets = closes.pct_change().fillna(0)

        # Same-day lookahead: classify regime using TODAY's cross-asset
        # return vector, then pick the best asset for that "regime".
        best_idx = rets.values.argmax(axis=1)

        # Credit ourselves the chosen asset's SAME-DAY return (no shift).
        port_rets = pd.Series(
            np.diag(rets.values[:, best_idx]),
            index=closes.index,
        )

        val_r = port_rets[(port_rets.index > train_end) & (port_rets.index <= val_end)]
        ho_r  = port_rets[port_rets.index > val_end]

        def _sharpe(r):
            if len(r) < 2 or r.std(ddof=1) == 0:
                return 0.0
            return float(r.mean() / r.std(ddof=1) * np.sqrt(252))

        return {
            "val_sharpe": _sharpe(val_r),
            "val_max_drawdown_pct": 5.0,
            "val_total_return_pct": float((1 + val_r).prod() - 1) * 100,
            "holdout_sharpe": _sharpe(ho_r),
            "holdout_max_drawdown_pct": 3.0,
            "holdout_total_return_pct": float((1 + ho_r).prod() - 1) * 100,
            "returns": port_rets,
        }
""")

# Legitimate strategy: uses a trailing 5-day momentum signal computed from
# data up to day t-1 to decide the position for day t.  The runner's
# weights-to-returns shift means the signal for day t earns day t+1's return.
# Even without the shift, the signal for day t uses only past data.
LEGITIMATE_CODE = textwrap.dedent("""\
    import numpy as np
    import pandas as pd

    def run(data, train_end, val_end, benchmark, config):
        symbols = sorted(k for k in data if not k.startswith("_"))
        closes = pd.DataFrame({s: data[s]["Close"] for s in symbols})
        rets = closes.pct_change().fillna(0)

        # Trailing 5-day momentum signal — uses only past data.
        # Signal for day t is based on returns from t-5 to t-1.
        mom = rets.rolling(5).sum().shift(1)   # shift(1): signal[t] uses data up to t-1
        best_idx = mom.values.argmax(axis=1)

        # Build weights: long the asset with the best trailing momentum.
        weights = pd.DataFrame(0.0, index=closes.index, columns=symbols)
        for i, s in enumerate(symbols):
            mask = best_idx == i
            weights.loc[closes.index[mask], s] = 1.0

        # Portfolio return: weight[t] earns return[t] (runner will shift again,
        # but even without shift the signal is lagged so no same-day leakage).
        port_rets = (weights * rets).sum(axis=1)

        val_r = port_rets[(port_rets.index > train_end) & (port_rets.index <= val_end)]
        ho_r  = port_rets[port_rets.index > val_end]

        def _sharpe(r):
            if len(r) < 2 or r.std(ddof=1) == 0:
                return 0.0
            return float(r.mean() / r.std(ddof=1) * np.sqrt(252))

        return {
            "val_sharpe": _sharpe(val_r),
            "val_max_drawdown_pct": 5.0,
            "val_total_return_pct": float((1 + val_r).prod() - 1) * 100,
            "holdout_sharpe": _sharpe(ho_r),
            "holdout_max_drawdown_pct": 3.0,
            "holdout_total_return_pct": float((1 + ho_r).prod() - 1) * 100,
            "returns": port_rets,
            "weights": weights,
        }
""")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSameDayCrossAssetLookahead:
    """Issue #65: causality check must catch same-day cross-asset leakage."""

    def test_sameday_lookahead_rejected(self, tmp_path):
        """A strategy that picks assets using today's cross-asset returns
        and credits itself the same-day return must be rejected."""
        symbols = ["SPY", "QQQ", "TLT"]
        _setup_data(tmp_path, symbols, n_days=200)

        manifest = _make_manifest(
            code=SAMEDAY_LOOKAHEAD_CODE, universe=symbols,
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error", (
            f"Expected same-day lookahead to be rejected, got: {result}"
        )
        assert "same-day" in result["error"].lower() or "cross-asset" in result["error"].lower(), (
            f"Error message should mention same-day/cross-asset: {result['error']}"
        )

    def test_legitimate_strategy_passes(self, tmp_path):
        """A strategy using trailing momentum (lagged signal) must NOT be
        flagged by the same-day check."""
        symbols = ["SPY", "QQQ", "TLT"]
        _setup_data(tmp_path, symbols, n_days=200)

        manifest = _make_manifest(
            code=LEGITIMATE_CODE, universe=symbols,
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok", (
            f"Legitimate strategy should pass, got error: {result}"
        )

    def test_sameday_check_skipped_for_single_asset(self, tmp_path):
        """With only one asset there is no cross-asset dimension, so the
        same-day cross-asset check should be a no-op."""
        symbols = ["SPY"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                sym = [k for k in data if not k.startswith("_")][0]
                rets = data[sym]["Close"].pct_change().fillna(0)
                val_r = rets[(rets.index > train_end) & (rets.index <= val_end)]
                ho_r  = rets[rets.index > val_end]
                def _s(r):
                    if len(r) < 2 or r.std(ddof=1) == 0:
                        return 0.0
                    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))
                return {
                    "val_sharpe": _s(val_r),
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": float((1 + val_r).prod() - 1) * 100,
                    "holdout_sharpe": _s(ho_r),
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": float((1 + ho_r).prod() - 1) * 100,
                }
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))
        assert result["status"] == "ok", (
            f"Single-asset strategy should pass: {result}"
        )
