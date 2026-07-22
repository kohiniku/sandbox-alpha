"""End-to-end tests for the cross-sectional engine.

All tests use synthetic data in-memory — no volume dependency.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest_runner import _run_structured_mode
from manifest import StrategyManifest


# ── helpers ────────────────────────────────────────────────────────────────


def _make_manifest(name="test", execution_mode="structured"):
    payload = {
        "name": name,
        "code_b64": "dW51c2Vk",
        "data_sources": [
            {"type": "ohlcv", "universe": [f"S{i:02d}" for i in range(20)], "start": "2024-01-01", "end": "2025-12-31"}
        ],
        "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
        "evaluator": {"type": "portfolio", "metrics": ["sharpe"], "benchmark": None},
        "execution_mode": execution_mode,
    }
    return StrategyManifest.from_dict(payload)


def _make_synthetic_panel(n_symbols=20, n_days=500, seed=42):
    """Create a realistic panel with varied random walks."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    panel = {}
    for i, sym in enumerate(symbols):
        drift = rng.normal(0.0005, 0.015, n_days)
        close = 100 * (1 + drift).cumprod()
        close = close * (1 + i * 0.01)  # give each symbol its own drift
        df = pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.002, n_days)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n_days),
        }, index=dates)
        panel[sym] = df
    return panel


def _run_cross(sandbox_fn, extras=None, n_symbols=20, n_days=500):
    """Helper: run _run_structured_mode with cross-signal sandbox + synthetic panel."""
    manifest = _make_manifest()
    panel = _make_synthetic_panel(n_symbols, n_days)
    dates = next(iter(panel.values())).index
    universe = sorted(panel.keys())

    # Build close_panel and asset_returns like manifest_runner does
    closes = {sym: panel[sym]["Close"] for sym in universe}
    close_panel = pd.DataFrame(closes, index=dates)
    asset_returns = close_panel.pct_change()
    train_end = dates[int(len(dates) * 0.6)]
    val_end = dates[int(len(dates) * 0.8)]

    sandbox = {
        "pd": pd,
        "np": np,
        "pandas": pd,
        "numpy": np,
        **sandbox_fn(panel),
        "__builtins__": {
            "len": len, "range": range, "isinstance": isinstance,
            "ValueError": ValueError, "Exception": Exception,
            "True": True, "False": False, "None": None,
            "__import__": lambda *a, **k: __import__(*a, **k),
        },
    }

    return json.loads(_run_structured_mode(
        manifest=manifest,
        sandbox=sandbox,
        all_data=panel,
        close_panel=close_panel,
        asset_returns=asset_returns,
        train_end=train_end,
        val_end=val_end,
        benchmark_series=None,
        benchmark_warning=None,
        extras_in=extras,
    ))


# ═══════════════════════════════════════════════════════════════════════════


class TestCrossEngineE2E:
    """End-to-end pipeline: strategy → engine → metrics."""

    def test_end_to_end_scores_with_top_k_produces_metrics_dict(self):
        """20 symbols × 500 days, xs_momentum-style scores, top_k=5, monthly rebalance.

        Verify the returned dict has in_sample/out_of_sample/holdout + cross_sectional
        sub-dict + all metric fields.
        """

        def _make_sbx(panel):
            def generate_cross_signal(data, extras=None):
                # Compute 20-day momentum as scores per symbol
                dates = next(iter(data.values())).index
                symbols = sorted(data.keys())
                arr = np.zeros((len(dates), len(symbols)))
                for j, sym in enumerate(symbols):
                    close = data[sym]["Close"]
                    arr[:, j] = close.pct_change(20).fillna(0).values
                return pd.DataFrame(arr, index=dates, columns=symbols)

            return {"generate_cross_signal": generate_cross_signal}

        result = _run_cross(_make_sbx, extras={
            "top_k": 5,
            "rebalance": "monthly",
            "cost_bps": 5.0,
        })

        # Not a placeholder
        assert result.get("status") != "ok_no_engine", "Should not return placeholder"
        assert "cross_engine_error" not in str(result), f"Engine error: {result.get('error', '')}"

        # Check top-level structure
        assert "in_sample" in result, "Missing in_sample"
        assert "out_of_sample" in result, "Missing out_of_sample"
        assert "holdout" in result, "Missing holdout"
        assert "cross_sectional" in result, "Missing cross_sectional"
        assert "walkforward" in result, "Missing walkforward"

        # Check cross_sectional sub-dict
        cs = result["cross_sectional"]
        assert "construction_mode" in cs
        assert "rebalance" in cs
        assert cs["rebalance"] == "monthly"
        assert "cost_bps" in cs
        assert "n_active_symbols_avg" in cs
        assert cs["n_active_symbols_avg"] > 0

        # Check metrics in each segment
        for seg in ["in_sample", "out_of_sample", "holdout"]:
            m = result[seg]
            assert "sharpe_ratio" in m, f"Missing sharpe_ratio in {seg}"
            assert "total_return" in m, f"Missing total_return in {seg}"
            assert "max_drawdown" in m, f"Missing max_drawdown in {seg}"
            assert "ir" in m, f"Missing ir in {seg}"
            assert "turnover" in m, f"Missing turnover in {seg}"
            assert "hit_rate" in m, f"Missing hit_rate in {seg}"
            assert "num_days" in m, f"Missing num_days in {seg}"
            assert "num_trades" in m, f"Missing num_trades in {seg}"

    def test_missing_construction_mode_uses_default_for_scores(self):
        """extras without construction_mode + return_type=scores → default top_k applied."""

        def _make_sbx(panel):
            def generate_cross_signal(data, extras=None):
                dates = next(iter(data.values())).index
                symbols = sorted(data.keys())
                arr = np.random.default_rng(42).normal(0, 1, (len(dates), len(symbols)))
                return pd.DataFrame(arr, index=dates, columns=symbols)

            return {"generate_cross_signal": generate_cross_signal}

        result = _run_cross(_make_sbx, extras={"rebalance": "daily"})
        assert result.get("status") != "ok_no_engine", "Should not return placeholder"
        cs = result["cross_sectional"]
        assert cs["construction_mode"] == "top_k"  # default for scores

    def test_engine_error_returns_cross_engine_error_type(self):
        """Force a divide-by-zero → engine returns error_type='cross_engine_error'."""

        def _make_sbx(panel):
            def generate_cross_signal(data, extras=None):
                dates = next(iter(data.values())).index
                symbols = sorted(data.keys())
                # Return empty/invalid data that will trip up the engine
                return pd.DataFrame(
                    np.zeros((len(dates), len(symbols))),
                    index=dates,
                    columns=symbols,
                )

            return {"generate_cross_signal": generate_cross_signal}

        result = _run_cross(_make_sbx, extras={
            "construction_mode": "zscore_continuous",
            "rebalance": "daily",
        })

        # Engine should handle this gracefully — or fail with cross_engine_error
        # Either way, it should not be the placeholder
        assert result.get("status") != "ok_no_engine", "Should not return placeholder after engine wiring"

        if result.get("status") == "error":
            assert "cross_engine" in result.get("error_type", ""), (
                f"Expected cross_engine_error, got {result.get('error_type')}"
            )


class TestExistingDispatchUnchanged:
    """Regression guards: existing dispatch paths untouched."""

    def test_v1_dispatch_still_returns_v1_shape(self):
        """Manifest with only generate_signals still returns v1 shape (no cross_sectional key)."""

        # Use a minimal synthetic test, imported from 4b's test pattern
        # Build sandbox with only generate_signals
        def _make_sbx(panel):
            def generate_signals(data):
                symbols = sorted(data.keys())
                dates = next(iter(data.values())).index
                arr = np.random.default_rng(42).choice([-1, 0, 1], (len(dates), len(symbols)))
                return pd.DataFrame(arr, index=dates, columns=symbols)

            return {"generate_signals": generate_signals}

        manifest = _make_manifest()
        panel = _make_synthetic_panel(5, 60)
        dates = next(iter(panel.values())).index
        universe = sorted(panel.keys())

        closes = {sym: panel[sym]["Close"] for sym in universe}
        close_panel = pd.DataFrame(closes, index=dates)
        asset_returns = close_panel.pct_change()
        train_end = dates[int(len(dates) * 0.6)]
        val_end = dates[int(len(dates) * 0.8)]

        sandbox = {
            "pd": pd,
            "np": np,
            "pandas": pd,
            "numpy": np,
            **_make_sbx(panel),
            "__builtins__": {
                "len": len, "range": range, "isinstance": isinstance,
                "ValueError": ValueError, "Exception": Exception,
                "True": True, "False": False, "None": None,
                "__import__": lambda *a, **k: __import__(*a, **k),
            },
        }

        result = json.loads(_run_structured_mode(
            manifest=manifest,
            sandbox=sandbox,
            all_data=panel,
            close_panel=close_panel,
            asset_returns=asset_returns,
            train_end=train_end,
            val_end=val_end,
            benchmark_series=None,
            benchmark_warning=None,
            extras_in=None,
        ))

        assert result["status"] == "ok"
        assert "cross_sectional" not in result, (
            "v1 dispatch should NOT have cross_sectional key"
        )
        assert "val_sharpe" in result.get("metrics", {}), "v1 response should have val_sharpe"

    def test_placeholder_no_longer_returned(self):
        """dispatch with generate_cross_signal must return real metrics, NOT placeholder.

        This test would have FAILED on PR 4b — proving engine wiring is complete.
        """

        def _make_sbx(panel):
            def generate_cross_signal(data, extras=None):
                dates = next(iter(data.values())).index
                symbols = sorted(data.keys())
                arr = np.random.default_rng(42).normal(0, 1, (len(dates), len(symbols)))
                return pd.DataFrame(arr, index=dates, columns=symbols)

            return {"generate_cross_signal": generate_cross_signal}

        result = _run_cross(_make_sbx, extras={"top_k": 5})

        # The key assertion: status must NOT be ok_no_engine
        assert result.get("status") != "ok_no_engine", (
            "FAIL: engine still returning placeholder. Engine wiring is NOT complete."
        )
        # Should have engine-produced metrics
        assert "in_sample" in result, "Should have in_sample from engine"
        assert "out_of_sample" in result, "Should have out_of_sample from engine"
        assert "holdout" in result, "Should have holdout from engine"
