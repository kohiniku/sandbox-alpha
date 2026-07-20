"""Tests for manifest_runner.py (v2 full execution pipeline)."""

import base64
import json
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest_runner import run_manifest, _signals_to_weights, _dict_signals_to_wide
from manifest import StrategyManifest


# ---------------------------------------------------------------------------
# Fixtures: build CSV data in tmp_path
# ---------------------------------------------------------------------------

def _make_ohlcv_csv(data_dir: str, symbol: str, dates, close_prices, seed=42):
    """Write a minimal OHLCV CSV for a symbol."""
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


def _make_manifest(
    name="test_strat",
    code="",
    universe=None,
    start="2023-01-01",
    end="2023-12-31",
    metrics=None,
    benchmark=None,
):
    """Build a StrategyManifest with code_b64 encoded."""
    if universe is None:
        universe = ["AAPL", "MSFT", "GOOG"]
    if metrics is None:
        metrics = ["sharpe", "max_drawdown"]
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
            "metrics": metrics,
            "benchmark": benchmark,
        },
    }
    return StrategyManifest.from_dict(payload)


def _setup_data(tmp_path, symbols, n_days=60):
    """Create CSVs for symbols with enough rows for metrics (>=20)."""
    dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")
    rng = np.random.RandomState(42)
    for i, sym in enumerate(symbols):
        prices = 100.0 + np.cumsum(rng.randn(n_days) * 0.5) + i * 10
        prices = np.maximum(prices, 1.0)  # keep positive
        _make_ohlcv_csv(str(tmp_path), sym, dates, prices, seed=i + 42)


# ---------------------------------------------------------------------------
# Test: happy path with generate_signals (wide DataFrame)
# ---------------------------------------------------------------------------

class TestHappyPathSignals:
    def test_generate_signals_wide_df(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=60)

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_signals(data):
                # Simple momentum: long if close > mean, short if close < mean
                signals = {}
                for sym, df in data.items():
                    ma = df["Close"].rolling(5).mean()
                    sig = (df["Close"] > ma).astype(int) - (df["Close"] < ma).astype(int)
                    signals[sym] = sig
                return pd.DataFrame(signals)
        """)

        manifest = _make_manifest(code=code, universe=symbols, metrics=["sharpe", "max_drawdown"])
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["manifest_name"] == "test_strat"
        assert result["universe_size"] == 3
        assert result["n_days"] > 0
        assert "sharpe" in result["metrics"]
        assert "max_drawdown" in result["metrics"]
        assert result["config"]["weighting"] == "equal_active_signals"
        assert result["config"]["benchmark"] is None


# ---------------------------------------------------------------------------
# Test: generate_weights path
# ---------------------------------------------------------------------------

class TestGenerateWeights:
    def test_generate_weights_direct(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=60)

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_weights(data):
                n = len(data)
                # Equal weight across all symbols
                first_df = next(iter(data.values()))
                w = pd.DataFrame(
                    1.0 / n,
                    index=first_df.index,
                    columns=list(data.keys()),
                )
                return w

            def generate_signals(data):
                # This should be ignored since generate_weights is present
                raise RuntimeError("should not be called")
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["config"]["weighting"] == "generate_weights"


# ---------------------------------------------------------------------------
# Test: user code raises -> error_type='code'
# ---------------------------------------------------------------------------

class TestUserCodeError:
    def test_runtime_error_in_user_code(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=60)

        code = textwrap.dedent("""\
            def generate_signals(data):
                raise ValueError("intentional bug")
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "intentional bug" in result["error"]
        assert "traceback" in result

    def test_no_entrypoint_defined(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=60)

        code = textwrap.dedent("""\
            # No generate_signals or generate_weights defined
            x = 42
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "generate_signals" in result["error"] or "generate_weights" in result["error"]


# ---------------------------------------------------------------------------
# Test: missing symbol data -> error_type='infra'
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_missing_csv(self, tmp_path):
        # Only create AAPL, not MSFT or GOOG
        _setup_data(tmp_path, ["AAPL"], n_days=60)

        code = textwrap.dedent("""\
            import pandas as pd
            def generate_signals(data):
                return pd.DataFrame(1, index=next(iter(data.values())).index,
                                    columns=list(data.keys()))
        """)

        manifest = _make_manifest(code=code, universe=["AAPL", "MSFT", "GOOG"])
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "infra"
        assert "MSFT" in result["error"] or "GOOG" in result["error"]


# ---------------------------------------------------------------------------
# Test: missing benchmark -> IR skipped with warning
# ---------------------------------------------------------------------------

class TestMissingBenchmark:
    def test_benchmark_not_in_universe(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=60)

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

        manifest = _make_manifest(
            code=code,
            universe=symbols,
            metrics=["sharpe", "ir", "max_drawdown"],
            benchmark="SPY",  # not in universe
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert "warning" in result
        assert "SPY" in result["warning"]
        assert "sharpe" in result["metrics"]
        assert "max_drawdown" in result["metrics"]
        # IR should be NaN since benchmark is None
        assert np.isnan(result["metrics"].get("ir", float("nan")))


# ---------------------------------------------------------------------------
# Test: signals dict form -> converted to wide DataFrame
# ---------------------------------------------------------------------------

class TestSignalsDictForm:
    def test_dict_signals_converted(self, tmp_path):
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=60)

        code = textwrap.dedent("""\
            import pandas as pd

            def generate_signals(data):
                # Return dict {symbol: Series}
                result = {}
                for sym, df in data.items():
                    result[sym] = pd.Series(1, index=df.index, name=sym)
                return result
        """)

        manifest = _make_manifest(code=code, universe=symbols, metrics=["sharpe"])
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["universe_size"] == 2


# ---------------------------------------------------------------------------
# Test: _signals_to_weights helper
# ---------------------------------------------------------------------------

class TestSignalsToWeights:
    def test_equal_weight_normalization(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        signals = pd.DataFrame(
            [[1, 1, 0], [1, -1, 0], [0, 0, 0]],
            index=idx,
            columns=["A", "B", "C"],
        )
        weights = _signals_to_weights(signals)

        # Row 0: 2 longs -> each 0.5
        assert weights.iloc[0]["A"] == pytest.approx(0.5)
        assert weights.iloc[0]["B"] == pytest.approx(0.5)
        assert weights.iloc[0]["C"] == 0.0

        # Row 1: 1 long (A=1), 1 short (B=-1)
        assert weights.iloc[1]["A"] == pytest.approx(1.0)
        assert weights.iloc[1]["B"] == pytest.approx(-1.0)
        assert weights.iloc[1]["C"] == 0.0

        # Row 2: all zero -> all zero
        assert (weights.iloc[2] == 0.0).all()


# ---------------------------------------------------------------------------
# Test: import restriction in user code
# ---------------------------------------------------------------------------

class TestImportRestriction:
    def test_os_import_blocked(self, tmp_path):
        symbols = ["AAPL"]
        _setup_data(tmp_path, symbols, n_days=60)

        code = textwrap.dedent("""\
            import os
            def generate_signals(data):
                return None
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "not allowed" in result["error"].lower() or "import" in result["error"].lower()
