"""Regression test: expert-mode run() returning non-serializable extras must
produce valid JSON output (no TypeError: Object of type Series is not JSON
serializable).

Reproduces the bug hit by backlog ids 17c4a1ea (meanrev_nvda_hmm_regime) and
e65e4fda (meanrev_tech5_pairs) on 2026-07-25.

Two bugs fixed:
1. _SERIALIZABLE_EXCLUDE only covers {returns, weights, positions} — any other
   pandas object returned by run() (e.g. "signals", "regimes", "scores") flows
   into out["expert_extras"] and crashes json.dumps(out).
2. The independent-verification block builds returns_df from the FULL
   multi-column per-asset DataFrame and passes it to evaluate() with
   weights=None, computing equal-weight-average metrics instead of using the
   user's actual portfolio_ret_series.  This produces semantically wrong
   verification metrics for single-signal/pairs strategies.
"""

import base64
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest_runner import run_manifest
from manifest import StrategyManifest


# ---------------------------------------------------------------------------
# Helpers (follow conventions from test_manifest_runner.py)
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


def _setup_data(tmp_path, symbols, n_days=400):
    """Create CSVs with enough rows for val + holdout segments (>=20 each)."""
    dates = pd.bdate_range("2021-01-04", periods=n_days, freq="B")
    rng = np.random.RandomState(42)
    for i, sym in enumerate(symbols):
        prices = 100.0 + np.cumsum(rng.randn(n_days) * 0.5) + i * 10
        prices = np.maximum(prices, 1.0)
        _make_ohlcv_csv(str(tmp_path), sym, dates, prices, seed=i + 42)
    return dates


def _make_expert_manifest(code: str, name: str = "test_expert_returns",
                          universe=None, metrics=None):
    if universe is None:
        universe = ["NVDA"]
    if metrics is None:
        metrics = ["sharpe", "max_drawdown_pct", "total_return_pct"]
    code_b64 = base64.b64encode(textwrap.dedent(code).encode()).decode()
    payload = {
        "name": name,
        "code_b64": code_b64,
        "data_sources": [{
            "type": "ohlcv",
            "universe": universe,
            "start": "2021-01-01",
            "end": "2022-12-31",
        }],
        "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
        "evaluator": {
            "type": "portfolio",
            "metrics": metrics,
            "benchmark": None,
        },
        "execution_mode": "expert",
    }
    return StrategyManifest.from_dict(payload)


# ---------------------------------------------------------------------------
# run() bodies
# ---------------------------------------------------------------------------

# Minimal momentum strategy that returns "returns" Series + extra "signals"
# Series.  The "signals" key is NOT in _SERIALIZABLE_EXCLUDE, so it flows
# into expert_extras and crashes json.dumps on current master.
_EXPERT_RUN_WITH_SIGNALS = """\
def run(data, train_end, val_end, benchmark, config):
    import pandas as pd, numpy as np
    df = data['NVDA']
    close = df['Close']
    ret = close.pct_change().dropna()
    sig = (ret.rolling(20).mean() > 0).astype(float) * 2 - 1
    port_ret = sig.shift(1) * ret
    port_ret = port_ret.dropna()
    dates = port_ret.index
    val_mask = (dates >= train_end) & (dates <= val_end)
    holdout_mask = dates > val_end
    val_rets = port_ret[val_mask]
    holdout_rets = port_ret[holdout_mask]
    def _sharpe(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))
    def _mdd(r):
        if len(r) == 0:
            return 0.0
        eq = (1 + r).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax()
        return float(dd.min() * 100)
    return {
        "val_sharpe": _sharpe(val_rets),
        "val_max_drawdown_pct": _mdd(val_rets),
        "val_total_return_pct": float((1 + val_rets).prod() - 1) * 100 if len(val_rets) > 0 else 0.0,
        "holdout_sharpe": _sharpe(holdout_rets),
        "holdout_max_drawdown_pct": _mdd(holdout_rets),
        "holdout_total_return_pct": float((1 + holdout_rets).prod() - 1) * 100 if len(holdout_rets) > 0 else 0.0,
        "returns": port_ret,
        "signals": sig,
    }
"""

# Same shape as the failing production manifests (meanrev_nvda_hmm_regime):
# returns only "returns" as extra key (which IS in _SERIALIZABLE_EXCLUDE).
# This tests the semantic fix: independent verification must use the user's
# portfolio_ret_series, not the multi-column equal-weight average.
_EXPERT_RUN_RETURNS_ONLY = """\
def run(data, train_end, val_end, benchmark, config):
    import pandas as pd, numpy as np
    df = data['NVDA']
    close = df['Close']
    ret = close.pct_change().dropna()
    sig = (ret.rolling(20).mean() > 0).astype(float) * 2 - 1
    port_ret = sig.shift(1) * ret
    port_ret = port_ret.dropna()
    dates = port_ret.index
    val_mask = (dates >= train_end) & (dates <= val_end)
    holdout_mask = dates > val_end
    val_rets = port_ret[val_mask]
    holdout_rets = port_ret[holdout_mask]
    def _sharpe(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))
    def _mdd(r):
        if len(r) == 0:
            return 0.0
        eq = (1 + r).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax()
        return float(dd.min() * 100)
    return {
        "val_sharpe": _sharpe(val_rets),
        "val_max_drawdown_pct": _mdd(val_rets),
        "val_total_return_pct": float((1 + val_rets).prod() - 1) * 100 if len(val_rets) > 0 else 0.0,
        "holdout_sharpe": _sharpe(holdout_rets),
        "holdout_max_drawdown_pct": _mdd(holdout_rets),
        "holdout_total_return_pct": float((1 + holdout_rets).prod() - 1) * 100 if len(holdout_rets) > 0 else 0.0,
        "returns": port_ret,
    }
"""


class TestExpertReturnsSeriesSerialization:
    """Expert-mode run() returning non-serializable extras must produce valid JSON."""

    def test_extra_series_key_produces_valid_json(self, tmp_path):
        """run() returning an extra 'signals' Series (not in _SERIALIZABLE_EXCLUDE)
        must not crash json.dumps.  This is the exact TypeError from production."""
        _setup_data(tmp_path, ["NVDA"], n_days=400)
        manifest = _make_expert_manifest(_EXPERT_RUN_WITH_SIGNALS)
        result_str = run_manifest(manifest, str(tmp_path))

        # Must be valid JSON (this is where the TypeError manifested)
        result = json.loads(result_str)
        assert result.get("status") == "ok", f"Expected ok, got: {result}"
        assert result.get("execution_mode") == "expert"

        # All metrics must be JSON-serializable scalars
        metrics = result.get("metrics", {})
        for k, v in metrics.items():
            assert isinstance(v, (int, float)), (
                f"metrics[{k!r}] is {type(v).__name__}, expected scalar"
            )

    def test_returns_only_produces_valid_json(self, tmp_path):
        """run() returning only 'returns' as extra key (the production shape)."""
        _setup_data(tmp_path, ["NVDA"], n_days=400)
        manifest = _make_expert_manifest(_EXPERT_RUN_RETURNS_ONLY)
        result_str = run_manifest(manifest, str(tmp_path))

        result = json.loads(result_str)
        assert result.get("status") == "ok", f"Expected ok, got: {result}"

        metrics = result.get("metrics", {})
        for k, v in metrics.items():
            assert isinstance(v, (int, float)), (
                f"metrics[{k!r}] is {type(v).__name__}, expected scalar"
            )

        # "returns" must NOT leak into the output
        assert "returns" not in result.get("expert_extras", {})

    def test_multi_symbol_returns_series(self, tmp_path):
        """Multi-symbol universe (like meanrev_tech5_pairs)."""
        symbols = ["AAA", "BBB", "CCC"]
        _setup_data(tmp_path, symbols, n_days=400)

        code = """\
def run(data, train_end, val_end, benchmark, config):
    import pandas as pd, numpy as np
    closes = pd.DataFrame({s: df["Close"] for s, df in data.items()})
    rets = closes.pct_change().dropna()
    port_ret = rets.mean(axis=1)
    dates = port_ret.index
    val_mask = (dates >= train_end) & (dates <= val_end)
    holdout_mask = dates > val_end
    val_rets = port_ret[val_mask]
    holdout_rets = port_ret[holdout_mask]
    def _sharpe(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))
    def _mdd(r):
        if len(r) == 0:
            return 0.0
        eq = (1 + r).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax()
        return float(dd.min() * 100)
    return {
        "val_sharpe": _sharpe(val_rets),
        "val_max_drawdown_pct": _mdd(val_rets),
        "val_total_return_pct": float((1 + val_rets).prod() - 1) * 100 if len(val_rets) > 0 else 0.0,
        "holdout_sharpe": _sharpe(holdout_rets),
        "holdout_max_drawdown_pct": _mdd(holdout_rets),
        "holdout_total_return_pct": float((1 + holdout_rets).prod() - 1) * 100 if len(holdout_rets) > 0 else 0.0,
        "returns": port_ret,
    }
"""
        manifest = _make_expert_manifest(code, name="test_multi_sym",
                                         universe=symbols)
        result_str = run_manifest(manifest, str(tmp_path))
        result = json.loads(result_str)
        assert result.get("status") == "ok", f"Expected ok, got: {result}"

        metrics = result.get("metrics", {})
        for k, v in metrics.items():
            assert isinstance(v, (int, float)), (
                f"metrics[{k!r}] is {type(v).__name__}, expected scalar"
            )


class TestIndependentVerificationUsesPortfolioReturns:
    """The independent verification must recompute metrics from the user's
    portfolio_ret_series, NOT from the multi-column equal-weight average."""

    def test_verification_uses_portfolio_series_not_equal_weight(self, tmp_path):
        """When run() returns a 'returns' Series that differs from the
        equal-weight average of all assets, the recomputed metrics must
        match the portfolio series, not the equal-weight average."""
        symbols = ["AAA", "BBB"]
        _setup_data(tmp_path, symbols, n_days=400)

        # Strategy that goes 100% long AAA only (not equal-weight)
        code = """\
def run(data, train_end, val_end, benchmark, config):
    import pandas as pd, numpy as np
    close_a = data["AAA"]["Close"]
    close_b = data["BBB"]["Close"]
    ret_a = close_a.pct_change().dropna()
    ret_b = close_b.pct_change().dropna()
    # Portfolio = 100% AAA (NOT equal-weight of AAA+BBB)
    port_ret = ret_a
    dates = port_ret.index
    val_mask = (dates >= train_end) & (dates <= val_end)
    holdout_mask = dates > val_end
    val_rets = port_ret[val_mask]
    holdout_rets = port_ret[holdout_mask]
    def _sharpe(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))
    def _mdd(r):
        if len(r) == 0:
            return 0.0
        eq = (1 + r).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax()
        return float(dd.min() * 100)
    return {
        "val_sharpe": _sharpe(val_rets),
        "val_max_drawdown_pct": _mdd(val_rets),
        "val_total_return_pct": float((1 + val_rets).prod() - 1) * 100 if len(val_rets) > 0 else 0.0,
        "holdout_sharpe": _sharpe(holdout_rets),
        "holdout_max_drawdown_pct": _mdd(holdout_rets),
        "holdout_total_return_pct": float((1 + holdout_rets).prod() - 1) * 100 if len(holdout_rets) > 0 else 0.0,
        "returns": port_ret,
    }
"""
        manifest = _make_expert_manifest(code, name="test_single_asset",
                                         universe=symbols)
        result_str = run_manifest(manifest, str(tmp_path))
        result = json.loads(result_str)
        assert result.get("status") == "ok", f"Expected ok, got: {result}"

        # The recomputed val_sharpe should be close to the self-reported one
        # (both computed from AAA-only returns), NOT from equal-weight AAA+BBB.
        metrics = result.get("metrics", {})
        sr = result.get("self_reported_metrics", {})

        if result.get("verification") == "independent_recompute":
            # Recomputed sharpe should be close to self-reported.
            # A small difference (~15%) is expected due to the train_end
            # boundary: self-reported uses >= train_end, evaluator uses > train_end.
            # The key assertion is that the diff is NOT ~87% (which would mean
            # the verification used the equal-weight average of all assets).
            if "val_sharpe" in metrics and "val_sharpe" in sr:
                rv = metrics["val_sharpe"]
                sv = sr["val_sharpe"]
                if sv != 0:
                    rel_diff = abs(rv - sv) / abs(sv)
                    assert rel_diff < 0.25, (
                        f"Recomputed val_sharpe={rv:.4f} differs too much from "
                        f"self-reported={sv:.4f} (rel_diff={rel_diff:.2%}). "
                        f"Verification likely used equal-weight average instead "
                        f"of the portfolio return series."
                    )
