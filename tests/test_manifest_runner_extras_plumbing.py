"""Regression test for evaluator.extras → _run_structured_mode plumbing.

Before this hotfix, `run_manifest` invoked `_run_structured_mode` without
passing `manifest.evaluator.extras` as `extras_in`. Consequence: any
LLM-provided config (cross_return_type, construction_mode, top_k,
cost_bps, long_only, quintiles, zscore_threshold) NEVER reached
`generate_cross_signal` or the cross_sectional engine. Everything ran on
hardcoded defaults from `backtests/cross_sectional/engine.py`.
"""
import base64
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest_runner import run_manifest
from manifest import StrategyManifest


def _make_ohlcv_csv(data_dir, symbol, dates, close_prices):
    df = pd.DataFrame({
        "Date": dates,
        "Open": close_prices,
        "High": close_prices * 1.01,
        "Low": close_prices * 0.99,
        "Close": close_prices,
        "Volume": 1_000_000,
    })
    df.to_csv(os.path.join(data_dir, f"{symbol}.csv"), index=False)


def _setup_data(tmp_path, symbols, n_days=400):
    dates = pd.bdate_range("2020-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(42)
    for i, sym in enumerate(symbols):
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n_days)) + i * 10
        prices = np.maximum(prices, 1.0)
        _make_ohlcv_csv(str(tmp_path), sym, dates, prices)


def _make_cross_manifest(evaluator_extras, symbols):
    code = textwrap.dedent("""
        def generate_cross_signal(data, extras=None):
            import pandas as pd
            closes = pd.DataFrame({s: df["Close"] for s, df in data.items()})
            return closes.pct_change(30).fillna(0.0)
    """)
    payload = {
        "name": "extras_probe",
        "code_b64": base64.b64encode(code.encode()).decode(),
        "data_sources": [{
            "type": "ohlcv",
            "universe": symbols,
            "start": "2020-01-01",
            "end": "2021-06-30",
        }],
        "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
        "evaluator": {
            "type": "portfolio",
            "metrics": ["sharpe"],
            "benchmark": None,
            "extras": evaluator_extras,
        },
        "execution_mode": "structured",
    }
    return StrategyManifest.from_dict(payload)


class TestEvaluatorExtrasReachesCrossEngine:
    """evaluator.extras must be threaded into cross_sectional engine config."""

    def test_cost_bps_override_reaches_engine(self, tmp_path):
        symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        _setup_data(tmp_path, symbols)
        manifest = _make_cross_manifest(
            {"cross_return_type": "scores", "construction_mode": "top_k",
             "top_k": 3, "cost_bps": 12.34},
            symbols,
        )

        result = json.loads(run_manifest(manifest, str(tmp_path)))
        assert "error" not in result, f"unexpected error: {result}"
        # Pre-hotfix would report cost_bps=5.0 (engine default); post-hotfix
        # echoes the manifest-supplied value.
        assert result["cross_sectional"]["cost_bps"] == 12.34

    def test_top_k_override_reaches_engine(self, tmp_path):
        symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        _setup_data(tmp_path, symbols)
        manifest = _make_cross_manifest(
            {"cross_return_type": "scores", "construction_mode": "top_k",
             "top_k": 2},
            symbols,
        )

        result = json.loads(run_manifest(manifest, str(tmp_path)))
        assert "error" not in result, f"unexpected error: {result}"
        # top_k=2 on 5-symbol universe → avg active symbols ~2. Engine
        # default is 50 (clipped to universe size = 5), so pre-hotfix would
        # show ~5 actives instead.
        assert result["cross_sectional"]["n_active_symbols_avg"] <= 2.5

    def test_construction_mode_override_reaches_engine(self, tmp_path):
        symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
        _setup_data(tmp_path, symbols)
        manifest = _make_cross_manifest(
            {"cross_return_type": "scores", "construction_mode": "quintile_ls",
             "quintiles": 5},
            symbols,
        )

        result = json.loads(run_manifest(manifest, str(tmp_path)))
        assert "error" not in result, f"unexpected error: {result}"
        assert result["cross_sectional"]["construction_mode"] == "quintile_ls"
