"""Regression tests for issue #81: deterministic data-coverage failures.

Two structural gaps covered:

1. An empty-panel alignment failure (no common dates across the universe)
   was classified as ``infra`` and retried 3x by the loop, even though it is
   deterministic — the manifest's universe/window is not backed by data.
   Regression: it must be classified ``code`` (no retry) and carry a
   per-symbol coverage report naming the culprits.

2. ``MissingDataError`` (symbol CSV absent from the cache) was also
   classified ``infra`` and retried. It is equally deterministic — retrying
   never materialises the file. Regression: ``code`` with the symbol named.

The diagnostics are what let the ideation loop / a human learn which
symbols or windows are viable, instead of a bare "no common dates" string.
"""

import base64
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest import StrategyManifest
from manifest_runner import run_manifest


# ---------------------------------------------------------------------------
# Helpers (follow conventions from tests/test_manifest_runner.py)
# ---------------------------------------------------------------------------

def _make_ohlcv_csv(data_dir, symbol, dates, close_prices, seed=42):
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


def _make_manifest(universe, start="2023-01-01", end="2023-12-31"):
    payload = {
        "name": "coverage_test",
        "code_b64": base64.b64encode(b"").decode(),
        "data_sources": [
            {"type": "ohlcv", "universe": universe, "start": start, "end": end}
        ],
        "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
        "evaluator": {"type": "portfolio", "metrics": ["sharpe"], "benchmark": None},
        "execution_mode": "structured",
    }
    return StrategyManifest.from_dict(payload)


def _setup_well_behaved(tmp_path, symbols, start="2023-01-02", periods=200):
    """CSVs for symbols fully inside the 2023 window."""
    dates = pd.bdate_range(start, periods=periods, freq="B")
    rng = np.random.RandomState(42)
    for i, sym in enumerate(symbols):
        prices = 100.0 + np.cumsum(rng.randn(periods) * 0.5) + i * 10
        prices = np.maximum(prices, 1.0)
        _make_ohlcv_csv(str(tmp_path), sym, dates, prices, seed=i + 42)


# ---------------------------------------------------------------------------
# Regression: empty-panel alignment failure (the VIX-style gap)
# ---------------------------------------------------------------------------

class TestEmptyPanelCoverageDiagnostics:
    def test_empty_panel_is_code_error_with_coverage_report(self, tmp_path):
        """Universe containing a symbol with NO data in the requested window
        must produce a 'code' (no-retry) error naming the culprit, with a
        per-symbol coverage report — not a retryable 'infra' error."""
        _setup_well_behaved(tmp_path, ["SPY", "TLT"], periods=200)
        # VIX CSV exists but its data lives entirely OUTSIDE the 2023 window.
        out_of_window = pd.bdate_range("2024-01-02", periods=60, freq="B")
        _make_ohlcv_csv(str(tmp_path), "VIX", out_of_window,
                        50.0 + np.arange(60), seed=99)

        manifest = _make_manifest(["SPY", "TLT", "VIX"])
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        # Deterministic data-coverage failure: not retryable infra.
        assert result["error_type"] == "code", (
            f"empty-panel coverage failure must be 'code' (no retry), got "
            f"{result['error_type']!r}: {result['error']}"
        )
        # Message must point at the culprit symbol, not a bare string.
        assert "VIX" in result["error"]
        # Per-symbol coverage diagnostics must be present.
        coverage = result.get("coverage")
        assert coverage is not None, "empty-panel error must carry coverage diagnostics"
        assert coverage["VIX"]["rows"] == 0
        assert coverage["VIX"]["empty"] is True
        assert coverage["SPY"]["rows"] > 0
        assert coverage["TLT"]["rows"] > 0

    def test_all_symbols_out_of_window_still_code_error(self, tmp_path):
        """Even when every symbol is out of range (empty panel regardless),
        the failure stays a deterministic 'code' error with diagnostics."""
        for i, sym in enumerate(["SPY", "TLT"]):
            dates = pd.bdate_range("2024-06-01", periods=30, freq="B")
            _make_ohlcv_csv(str(tmp_path), sym, dates,
                            100.0 + i, seed=i)

        manifest = _make_manifest(["SPY", "TLT"])
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        coverage = result.get("coverage", {})
        assert all(c["rows"] == 0 for c in coverage.values())


# ---------------------------------------------------------------------------
# Regression: MissingDataError (symbol CSV absent from cache)
# ---------------------------------------------------------------------------

class TestMissingSymbolDataError:
    def test_missing_csv_is_code_error_naming_symbol(self, tmp_path):
        """A symbol with no CSV at all in the cache is deterministic — the
        runner cannot retry it into existence. Must be 'code', not 'infra'."""
        _setup_well_behaved(tmp_path, ["SPY"], periods=200)

        manifest = _make_manifest(["SPY", "VIX"])
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code", (
            f"missing-symbol failure must be 'code' (no retry), got "
            f"{result['error_type']!r}: {result['error']}"
        )
        assert "VIX" in result["error"]
