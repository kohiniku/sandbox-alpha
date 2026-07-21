"""Tests for generate_cross_signal dispatch in manifest_runner.py (PR 4b).

These tests exercise the cross-sectional dispatch path in _run_structured_mode:
entrypoint recognition, validator routing, contract violation handling, and
placeholder response shape. The engine wiring (portfolio metrics, cost model)
lands in PR 4c — the placeholder response is intentional.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest_runner import _run_structured_mode
from manifest import StrategyManifest


# ── helpers ────────────────────────────────────────────────────────────────


def _make_manifest(name="test", execution_mode="structured"):
    """Minimal manifest shell — only the fields read by _run_structured_mode."""
    payload = {
        "name": name,
        "code_b64": "dW51c2Vk",  # "unused" in b64
        "data_sources": [{"type": "ohlcv", "universe": ["A", "B"], "start": "2024-01-01", "end": "2024-12-31"}],
        "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
        "evaluator": {"type": "portfolio", "metrics": ["sharpe"], "benchmark": None},
        "execution_mode": execution_mode,
    }
    return StrategyManifest.from_dict(payload)


def _make_dates(n=20):
    return pd.bdate_range("2024-01-02", periods=n)


def _make_all_data(symbols=("A", "B"), n_days=20):
    """Create minimal dict-of-DataFrames like load_ohlcv returns."""
    dates = _make_dates(n_days)
    rng = np.random.default_rng(42)
    data = {}
    for sym in symbols:
        data[sym] = pd.DataFrame({
            "Close": 100 * (1 + rng.normal(0, 0.01, n_days)).cumprod(),
        }, index=dates)
    return data


def _make_close_panel(all_data, symbols=("A", "B")):
    """Build close_panel like manifest_runner does from all_data."""
    dates = _make_dates(len(next(iter(all_data.values()))))
    closes = {sym: all_data[sym]["Close"] for sym in symbols}
    return pd.DataFrame(closes, index=dates)


# ═══════════════════════════════════════════════════════════════════════════
# Direct dispatch tests via _run_structured_mode
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossDispatch:
    """Tests for generate_cross_signal dispatch in _run_structured_mode."""

    # ── infrastructure ─────────────────────────────────────────────────

    def _run_cross(self, sandbox_fn, extras=None, symbols=("A", "B")):
        """Helper: call _run_structured_mode with a cross-signal sandbox."""
        manifest = _make_manifest()
        all_data = _make_all_data(symbols)
        dates = _make_dates(len(all_data[symbols[0]]))
        close_panel = _make_close_panel(all_data, symbols)
        asset_returns = close_panel.pct_change()
        train_end = dates[int(len(dates) * 0.6)]
        val_end = dates[int(len(dates) * 0.8)]

        sandbox = {
            "pd": pd,
            "np": np,
            "pandas": pd,
            "numpy": np,
            **sandbox_fn(all_data),
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
            all_data=all_data,
            close_panel=close_panel,
            asset_returns=asset_returns,
            train_end=train_end,
            val_end=val_end,
            benchmark_series=None,
            benchmark_warning=None,
            extras_in=extras,
        ))

    # ── test: dispatch recognition ─────────────────────────────────────

    def test_dispatch_recognizes_generate_cross_signal(self):
        """generate_cross_signal is recognized and invoked exactly once."""

        call_count = [0]

        def _make_sandbox(data):
            def generate_cross_signal(data2, extras=None):
                call_count[0] += 1
                dates = next(iter(data2.values())).index
                return pd.DataFrame(
                    {"A": np.random.randn(len(dates)),
                     "B": np.random.randn(len(dates))},
                    index=dates,
                )
            return {"generate_cross_signal": generate_cross_signal}

        result = self._run_cross(_make_sandbox)
        assert call_count[0] == 1, f"Expected 1 call, got {call_count[0]}"
        assert result["entrypoint"] == "generate_cross_signal"
        assert result["return_type"] == "scores"  # default

    # ── test: validator routing by return_type ─────────────────────────

    def test_dispatch_calls_correct_validator_for_return_type(self):
        """extras["cross_return_type"] routes to the matching validator."""
        # scores (default) — should pass
        def _make_scores_sbx(data):
            def generate_cross_signal(data2, extras=None):
                dates = next(iter(data2.values())).index
                return pd.DataFrame(
                    np.random.randn(len(dates), 2),
                    index=dates, columns=["A", "B"],
                )
            return {"generate_cross_signal": generate_cross_signal}

        result = self._run_cross(_make_scores_sbx)
        assert result["status"] == "ok_no_engine"
        assert result["return_type"] == "scores"

        # weights — must pass validate_weights
        def _make_weights_sbx(data):
            def generate_cross_signal(data2, extras=None):
                dates = next(iter(data2.values())).index
                return pd.DataFrame(
                    [[0.6, 0.4]] * len(dates),
                    index=dates, columns=["A", "B"],
                )
            return {"generate_cross_signal": generate_cross_signal}

        result = self._run_cross(_make_weights_sbx, extras={"cross_return_type": "weights"})
        assert result["status"] == "ok_no_engine"
        assert result["return_type"] == "weights"

        # signals — must pass validate_signals
        def _make_signals_sbx(data):
            def generate_cross_signal(data2, extras=None):
                dates = next(iter(data2.values())).index
                return pd.DataFrame(
                    [[1, -1]] * len(dates),
                    index=dates, columns=["A", "B"],
                )
            return {"generate_cross_signal": generate_cross_signal}

        result = self._run_cross(_make_signals_sbx, extras={"cross_return_type": "signals"})
        assert result["status"] == "ok_no_engine"
        assert result["return_type"] == "signals"

        # explicit "scores" with extras
        result = self._run_cross(_make_scores_sbx, extras={"cross_return_type": "scores"})
        assert result["status"] == "ok_no_engine"
        assert result["return_type"] == "scores"

    # ── test: contract violation → error ───────────────────────────────

    def test_dispatch_returns_error_on_contract_violation(self):
        """Strategy returns weights that don't sum to 1 → error JSON."""

        def _make_bad_sbx(data):
            def generate_cross_signal(data2, extras=None):
                dates = next(iter(data2.values())).index
                # Weights sum to 0.5 — violates long-only contract
                return pd.DataFrame(
                    [[0.3, 0.2]] * len(dates),
                    index=dates, columns=["A", "B"],
                )
            return {"generate_cross_signal": generate_cross_signal}

        result = self._run_cross(_make_bad_sbx, extras={"cross_return_type": "weights"})
        assert result["status"] == "error"
        assert "cross_contract" in result["error_type"]

    # ── test: valid output → placeholder ───────────────────────────────

    def test_dispatch_returns_placeholder_on_valid_output(self):
        """Engine wiring (PR 4c) replaces this placeholder with real metrics.

        This test verifies that the placeholder shape is correct so PR 4c
        knows exactly what structure to replace.
        """
        def _make_sbx(data):
            def generate_cross_signal(data2, extras=None):
                dates = next(iter(data2.values())).index
                return pd.DataFrame(
                    np.random.randn(len(dates), 2),
                    index=dates, columns=["A", "B"],
                )
            return {"generate_cross_signal": generate_cross_signal}

        result = self._run_cross(_make_sbx)
        assert result["status"] == "ok_no_engine"
        assert result["entrypoint"] == "generate_cross_signal"
        assert result["return_type"] == "scores"
        assert isinstance(result["shape"], list) and len(result["shape"]) == 2
        assert result["shape"][0] > 0
        assert result["shape"][1] == 2
        assert isinstance(result["index_range"], list) and len(result["index_range"]) == 2
        assert isinstance(result["sum_per_row_sample"], list)

    # ── test: non-DataFrame return → error ─────────────────────────────

    def test_dispatch_rejects_non_dataframe_return(self):
        def _make_sbx(data):
            def generate_cross_signal(data2, extras=None):
                return {"A": [1, 2, 3]}  # dict, not DataFrame
            return {"generate_cross_signal": generate_cross_signal}

        result = self._run_cross(_make_sbx)
        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "DataFrame" in result["error"]

    # ── test: unknown return_type → error ──────────────────────────────

    def test_dispatch_rejects_unknown_return_type(self):
        def _make_sbx(data):
            def generate_cross_signal(data2, extras=None):
                dates = next(iter(data2.values())).index
                return pd.DataFrame(
                    np.random.randn(len(dates), 2),
                    index=dates, columns=["A", "B"],
                )
            return {"generate_cross_signal": generate_cross_signal}

        result = self._run_cross(_make_sbx, extras={"cross_return_type": "bogus"})
        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "bogus" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════
# Regression: existing dispatch unchanged
# ═══════════════════════════════════════════════════════════════════════════


class TestExistingDispatchUnchanged:
    """generate_signals and generate_weights dispatch is bit-identical to main."""

    def test_existing_generate_signals_dispatch_unchanged(self, tmp_path):
        """A manifest with only generate_signals still works as on main."""
        import base64 as b64

        # Write synthetic CSVs
        symbols = ["A", "B"]
        dates = pd.bdate_range("2024-01-02", periods=60)
        for i, sym in enumerate(symbols):
            rng = np.random.RandomState(i + 42)
            prices = 100 + np.cumsum(rng.randn(60) * 0.5)
            prices = np.maximum(prices, 1.0)
            df = pd.DataFrame({
                "Date": dates,
                "Open": prices * (1 + rng.uniform(-0.005, 0.005, 60)),
                "High": prices * (1 + rng.uniform(0.001, 0.02, 60)),
                "Low": prices * (1 - rng.uniform(0.001, 0.02, 60)),
                "Close": prices,
                "Volume": rng.randint(100_000, 10_000_000, 60),
            })
            df.to_csv(os.path.join(str(tmp_path), f"{sym}.csv"), index=False)

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_signals(data):
                signals = {}
                for sym, df in data.items():
                    ma = df["Close"].rolling(5).mean()
                    sig = (df["Close"] > ma).astype(int) - (df["Close"] < ma).astype(int)
                    signals[sym] = sig
                return pd.DataFrame(signals)
        """)

        code_b64 = b64.b64encode(code.encode()).decode()
        payload = {
            "name": "regression_test",
            "code_b64": code_b64,
            "data_sources": [{"type": "ohlcv", "universe": symbols,
                              "start": "2024-01-01", "end": "2024-12-31"}],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe", "max_drawdown_pct"], "benchmark": None},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(payload)

        from manifest_runner import run_manifest
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["execution_mode"] == "structured"
        assert result["universe_size"] == 2
        assert result["n_days"] > 0
        assert "val_sharpe" in result["metrics"]
        assert "holdout_sharpe" in result["metrics"]
        assert result["config"]["weighting"] == "equal_active_signals"
