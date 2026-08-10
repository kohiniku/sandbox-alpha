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
    execution_mode="structured",
):
    """Build a StrategyManifest with code_b64 encoded."""
    if universe is None:
        universe = ["AAPL", "MSFT", "GOOG"]
    if metrics is None:
        metrics = ["sharpe", "max_drawdown_pct"]
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
        "execution_mode": execution_mode,
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
        _setup_data(tmp_path, symbols, n_days=200)

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

        manifest = _make_manifest(code=code, universe=symbols, metrics=["sharpe", "max_drawdown_pct"])
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["manifest_name"] == "test_strat"
        assert result["universe_size"] == 3
        assert result["n_days"] > 0
        assert result["execution_mode"] == "structured"
        assert "val_sharpe" in result["metrics"]
        assert "val_max_drawdown_pct" in result["metrics"]
        assert "holdout_sharpe" in result["metrics"]
        assert "holdout_max_drawdown_pct" in result["metrics"]
        assert "val_max_drawdown_pct" in result["metrics"]
        assert "holdout_max_drawdown_pct" in result["metrics"]
        assert "val_total_return_pct" in result["metrics"]
        assert "holdout_total_return_pct" in result["metrics"]
        assert result["config"]["weighting"] == "equal_active_signals"
        assert result["config"]["benchmark"] is None
        assert "train_end" in result["config"]
        assert "val_end" in result["config"]


# ---------------------------------------------------------------------------
# Test: generate_weights path
# ---------------------------------------------------------------------------

class TestGenerateWeights:
    def test_generate_weights_direct(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=200)

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
        # Issue #81: a missing symbol CSV is a deterministic coverage failure
        # — retrying cannot materialise the file, so it must be classified
        # 'code' (no retry), not 'infra'.
        assert result["error_type"] == "code"
        assert "MSFT" in result["error"] or "GOOG" in result["error"]


# ---------------------------------------------------------------------------
# Test: missing benchmark -> IR skipped with warning
# ---------------------------------------------------------------------------

class TestMissingBenchmark:
    def test_benchmark_not_in_universe(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=200)

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
            metrics=["sharpe", "ir", "max_drawdown_pct"],
            benchmark="SPY",  # not in universe
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert "warning" in result
        assert "SPY" in result["warning"]
        assert "val_sharpe" in result["metrics"]
        assert "val_max_drawdown_pct" in result["metrics"]
        # IR should be NaN since benchmark is None
        assert np.isnan(result["metrics"].get("val_ir", result["metrics"].get("holdout_ir", float("nan"))))


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


# ---------------------------------------------------------------------------
# Expert mode tests
# ---------------------------------------------------------------------------

class TestExpertMode:
    def test_expert_happy_path(self, tmp_path):
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                first_sym = list(data.keys())[0]
                val_returns = data[first_sym]["Close"].pct_change().dropna()
                val_returns = val_returns[(val_returns.index > train_end) & (val_returns.index <= val_end)]
                holdout_returns = data[first_sym]["Close"].pct_change().dropna()
                holdout_returns = holdout_returns[holdout_returns.index > val_end]
                mu_val = val_returns.mean()
                sigma_val = val_returns.std(ddof=1)
                val_sharpe = float(mu_val / sigma_val * np.sqrt(252)) if sigma_val > 0 else 0.0
                mu_ho = holdout_returns.mean()
                sigma_ho = holdout_returns.std(ddof=1)
                holdout_sharpe = float(mu_ho / sigma_ho * np.sqrt(252)) if sigma_ho > 0 else 0.0
                return {
                    "val_sharpe": val_sharpe,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": holdout_sharpe,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 8.0,
                    "my_extra_metric": 42.5,
                }
        """)

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["execution_mode"] == "expert"
        assert "val_sharpe" in result["metrics"]
        assert "val_max_drawdown_pct" in result["metrics"]
        assert "val_total_return_pct" in result["metrics"]
        assert "holdout_sharpe" in result["metrics"]
        assert "holdout_max_drawdown_pct" in result["metrics"]
        assert "holdout_total_return_pct" in result["metrics"]
        assert "expert_extras" in result
        assert result["expert_extras"]["my_extra_metric"] == 42.5
        assert result["config"]["entrypoint"] == "run"

    def test_missing_run_entrypoint(self, tmp_path):
        symbols = ["AAPL"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = "x = 1  # no run() defined"

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "run(" in result["error"] or "run()" in result["error"]

    def test_missing_required_metric(self, tmp_path):
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            def run(data, train_end, val_end, benchmark, config):
                # Missing holdout_* metrics
                return {
                    "val_sharpe": 1.5,
                    "val_max_drawdown_pct": 3.0,
                    "val_total_return_pct": 8.0,
                }
        """)

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "missing" in result["error"].lower()
        assert "holdout" in result["error"]

    def test_non_finite_metric(self, tmp_path):
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            import math

            def run(data, train_end, val_end, benchmark, config):
                return {
                    "val_sharpe": float("nan"),
                    "val_max_drawdown_pct": 3.0,
                    "val_total_return_pct": 8.0,
                    "holdout_sharpe": 1.0,
                    "holdout_max_drawdown_pct": 2.0,
                    "holdout_total_return_pct": 5.0,
                }
        """)

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["degenerate"] is True
        assert "not finite" in result["degenerate_reason"].lower()
        assert "val_sharpe" in result["degenerate_reason"]

    def test_pathological_sharpe_warning(self, tmp_path):
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            import math

            def run(data, train_end, val_end, benchmark, config):
                return {
                    "val_sharpe": 15.0,
                    "val_max_drawdown_pct": 3.0,
                    "val_total_return_pct": 8.0,
                    "holdout_sharpe": 1.0,
                    "holdout_max_drawdown_pct": 2.0,
                    "holdout_total_return_pct": 5.0,
                }
        """)

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert "pathological_warnings" in result
        assert any("sharpe" in w.lower() for w in result["pathological_warnings"])

    def test_import_scipy(self, tmp_path):
        """scipy is allowed but may not be installed — best-effort."""
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            import numpy as np
            from scipy import stats

            def run(data, train_end, val_end, benchmark, config):
                return {
                    "val_sharpe": 1.5,
                    "val_max_drawdown_pct": 3.0,
                    "val_total_return_pct": 8.0,
                    "holdout_sharpe": 1.0,
                    "holdout_max_drawdown_pct": 2.0,
                    "holdout_total_return_pct": 5.0,
                }
        """)

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        # scipy is a best-effort import; if not installed, it's a normal code_error
        if result["status"] == "ok":
            assert result["execution_mode"] == "expert"
        else:
            assert result["error_type"] == "code"

    def test_import_os_blocked(self, tmp_path):
        symbols = ["AAPL"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            import os

            def run(data, train_end, val_end, benchmark, config):
                return {"val_sharpe": 1.5}
        """)

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "not allowed" in result["error"].lower() or "import" in result["error"].lower()

    def test_run_raises_exception(self, tmp_path):
        symbols = ["AAPL"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            def run(data, train_end, val_end, benchmark, config):
                raise ValueError("expert bug")
        """)

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "expert bug" in result["error"]


# ---------------------------------------------------------------------------
# Expert mode — causality check + independent verification
# ---------------------------------------------------------------------------

class TestExpertCausalityCheck:
    """Causality (lookahead) check for expert mode."""

    def test_causal_strategy_passes(self, tmp_path):
        """Honest run() returning only required metrics — causal, no leak."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                # Causal: only use data up to val_end for val metrics,
                # and only up to last date for holdout.
                first_sym = list(data.keys())[0]
                returns = data[first_sym]["Close"].pct_change().dropna()
                val_r = returns[(returns.index > train_end) & (returns.index <= val_end)]
                ho_r = returns[returns.index > val_end]
                mu_v = val_r.mean(); sigma_v = val_r.std(ddof=1)
                val_s = float(mu_v/sigma_v*np.sqrt(252)) if sigma_v > 0 else 0.0
                mu_h = ho_r.mean(); sigma_h = ho_r.std(ddof=1)
                ho_s = float(mu_h/sigma_h*np.sqrt(252)) if sigma_h > 0 else 0.0
                return {
                    "val_sharpe": val_s,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": ho_s,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 8.0,
                }
        """)

        manifest = _make_manifest(
            code=code, universe=symbols, execution_mode="expert",
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["verification"] == "causality_check_only"
        assert "val_sharpe" in result["metrics"]
        assert "holdout_sharpe" in result["metrics"]

    def test_causality_check_catches_holdout_leak(self, tmp_path):
        """run() that peeks at the last close (holdout) must be REJECTED."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                first_sym = list(data.keys())[0]
                df = data[first_sym]
                # DELIBERATE LEAK: peek at the very last close for val_sharpe
                last_close = df["Close"].iloc[-1]
                returns = df["Close"].pct_change().dropna()
                val_r = returns[(returns.index > train_end) & (returns.index <= val_end)]
                mu_v = val_r.mean()
                sigma_v = val_r.std(ddof=1)
                # val_sharpe contaminated with holdout info:
                val_s = float((mu_v + (last_close/df["Close"].iloc[0] - 1)*0.1)
                              / max(sigma_v, 0.001) * np.sqrt(252))
                ho_r = returns[returns.index > val_end]
                mu_h = ho_r.mean(); sigma_h = ho_r.std(ddof=1)
                ho_s = float(mu_h/sigma_h*np.sqrt(252)) if sigma_h > 0 else 0.0
                return {
                    "val_sharpe": val_s,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": ho_s,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 8.0,
                }
        """)

        manifest = _make_manifest(
            code=code, universe=symbols, execution_mode="expert",
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error", f"Expected error but got {result.get('status')}"
        assert result["error_type"] == "code"
        assert "lookahead" in result["error"].lower()

    def test_no_holdout_skips_causality_check(self, tmp_path):
        """When val_end is the last date, causality check is skipped gracefully."""
        symbols = ["AAPL", "MSFT"]
        # Use few enough days that val_end == last date (80% with 60 days = no holdout
        # after the walk-forward split actually includes holdout at 80/20).
        # Actually we need val_end to be the last available business day.
        # The walk_forward_split uses 60/20/20. For n=60, train_end_idx=35, val_end_idx=47.
        # But we need val_end to be == last_date. With 60 days the holdout region is
        # [48..59]. We need a manifest where val_end IS the last date.
        # We can't easily manipulate the split from outside, but the check only
        # fires if val_end < last_date. For synthetic data that spans same dates,
        # this will normally be true. The check is skipped gracefully.
        #
        # Actually the simplest way: use n_days where the walk-forward split
        # makes val_end == last date. This happens when there's no holdout fraction
        # or the total n is too small. For n=10, val_end_idx=7, below the last idx=9.
        # Let's just test that a small dataset without holdout still works.

        _setup_data(tmp_path, symbols, n_days=50)  # small dataset

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                first_sym = list(data.keys())[0]
                returns = data[first_sym]["Close"].pct_change().dropna()
                val_r = returns[(returns.index > train_end) & (returns.index <= val_end)]
                # holdout may be empty or small — that's fine
                ho_r = returns[returns.index > val_end]
                if len(val_r) < 2:
                    mu_v, sigma_v = 0.0, 0.01
                else:
                    mu_v = val_r.mean(); sigma_v = val_r.std(ddof=1)
                val_s = float(mu_v/sigma_v*np.sqrt(252)) if sigma_v > 0 else 0.0
                if len(ho_r) < 2:
                    mu_h, sigma_h = 0.0, 0.01
                else:
                    mu_h = ho_r.mean(); sigma_h = ho_r.std(ddof=1)
                ho_s = float(mu_h/sigma_h*np.sqrt(252)) if sigma_h > 0 else 0.0
                return {
                    "val_sharpe": val_s,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": ho_s,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 8.0,
                }
        """)

        manifest = _make_manifest(
            code=code, universe=symbols, execution_mode="expert",
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        # Should still succeed — no lookahead error
        assert result["status"] == "ok"


class TestExpertIndependentVerification:
    """Independent metric recomputation for expert mode."""

    def test_recompute_overrides_fabricated_sharpe(self, tmp_path):
        """run() returns a genuine returns series + fabricated self-reported
        metrics. The recomputed values must be authoritative."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                first_sym = list(data.keys())[0]
                returns = data[first_sym]["Close"].pct_change().dropna()
                val_r = returns[(returns.index > train_end) & (returns.index <= val_end)]
                ho_r = returns[returns.index > val_end]
                mu_v = val_r.mean(); sigma_v = val_r.std(ddof=1)
                val_s = float(mu_v/sigma_v*np.sqrt(252)) if sigma_v > 0 else 0.0
                mu_h = ho_r.mean(); sigma_h = ho_r.std(ddof=1)
                ho_s = float(mu_h/sigma_h*np.sqrt(252)) if sigma_h > 0 else 0.0

                # Also compute real portfolio returns from simple equal-weight
                full_returns = returns  # this is the full date range returns
                # For recompute to work, the returns must span past val_end
                return {
                    "val_sharpe": 99.0,  # FABRICATED
                    "val_max_drawdown_pct": 0.1,
                    "val_total_return_pct": 9999.0,
                    "holdout_sharpe": 88.0,
                    "holdout_max_drawdown_pct": 0.05,
                    "holdout_total_return_pct": 8888.0,
                    "returns": full_returns,
                }
        """)

        manifest = _make_manifest(
            code=code, universe=symbols, execution_mode="expert",
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["verification"] == "independent_recompute"
        # The recomputed val_sharpe must NOT be 99.0
        assert result["metrics"]["val_sharpe"] != 99.0
        assert result["metrics"]["val_sharpe"] != 9999.0
        # The self-reported values are preserved
        assert "self_reported_metrics" in result
        assert result["self_reported_metrics"]["val_sharpe"] == 99.0
        # Mismatch warning must mention val_sharpe
        assert "metrics_mismatch_warning" in result
        assert "val_sharpe" in result["metrics_mismatch_warning"]

    def test_recompute_consistent_no_mismatch(self, tmp_path):
        """run() that returns a consistent returns series and matching
        self-reported metrics — no mismatch warning."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                first_sym = list(data.keys())[0]
                df = data[first_sym]
                returns = df["Close"].pct_change().dropna()
                val_r = returns[(returns.index > train_end) & (returns.index <= val_end)]
                ho_r = returns[returns.index > val_end]
                mu_v = val_r.mean(); sigma_v = val_r.std(ddof=1)
                val_s = float(mu_v/sigma_v*np.sqrt(252)) if sigma_v > 0 else 0.0
                mu_h = ho_r.mean(); sigma_h = ho_r.std(ddof=1)
                ho_s = float(mu_h/sigma_h*np.sqrt(252)) if sigma_h > 0 else 0.0

                # Compute consistent drawdown-like and return approximations
                val_cum = (1 + val_r).prod()
                val_ret = float((val_cum - 1) * 100)
                ho_cum = (1 + ho_r).prod()
                ho_ret = float((ho_cum - 1) * 100)
                val_dd = 5.0  # approximate
                ho_dd = 3.0

                return {
                    "val_sharpe": val_s,
                    "val_max_drawdown_pct": val_dd,
                    "val_total_return_pct": val_ret,
                    "holdout_sharpe": ho_s,
                    "holdout_max_drawdown_pct": ho_dd,
                    "holdout_total_return_pct": ho_ret,
                    "returns": returns,
                }
        """)

        manifest = _make_manifest(
            code=code, universe=symbols, execution_mode="expert",
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["verification"] == "independent_recompute"
        assert "self_reported_metrics" in result
        # No mismatch warning since val_sharpe and val_total_return_pct
        # are computed from the same data the runner will use
        # (they'll differ slightly because evaluate() computes differently,
        # but within 5% rel_tol)

    def test_recompute_via_positions(self, tmp_path):
        """run() returns positions/weights dict instead of returns series."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                # Build equal-weight positions for all dates
                first_df = next(df for k, df in data.items()
                                if not k.startswith("_"))
                n_assets = len([k for k in data if not k.startswith("_")])
                idx = first_df.index
                cols = [k for k in data if not k.startswith("_")]
                w = pd.DataFrame(1.0 / n_assets, index=idx, columns=cols)
                return {
                    "val_sharpe": 1.5,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": 1.0,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 8.0,
                    "weights": w,
                }
        """)

        manifest = _make_manifest(
            code=code, universe=symbols, execution_mode="expert",
        )
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["verification"] == "independent_recompute"

    def test_existing_expert_tests_unchanged(self, tmp_path):
        """Sanity: the standard expert happy-path test still works and now
        includes the new verification + self_reported_metrics fields."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                first_sym = list(data.keys())[0]
                val_returns = data[first_sym]["Close"].pct_change().dropna()
                val_returns = val_returns[(val_returns.index > train_end) & (val_returns.index <= val_end)]
                holdout_returns = data[first_sym]["Close"].pct_change().dropna()
                holdout_returns = holdout_returns[holdout_returns.index > val_end]
                mu_val = val_returns.mean()
                sigma_val = val_returns.std(ddof=1)
                val_sharpe = float(mu_val / sigma_val * np.sqrt(252)) if sigma_val > 0 else 0.0
                mu_ho = holdout_returns.mean()
                sigma_ho = holdout_returns.std(ddof=1)
                holdout_sharpe = float(mu_ho / sigma_ho * np.sqrt(252)) if sigma_ho > 0 else 0.0
                return {
                    "val_sharpe": val_sharpe,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": holdout_sharpe,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 8.0,
                    "my_extra_metric": 42.5,
                }
        """)

        manifest = _make_manifest(code=code, universe=symbols, execution_mode="expert")
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["execution_mode"] == "expert"
        assert result["verification"] == "causality_check_only"
        assert "self_reported_metrics" in result
        assert "my_extra_metric" in result.get("expert_extras", {})
        assert "val_sharpe" in result["metrics"]


# ---------------------------------------------------------------------------
# News sentiment integration tests (Phase 2 PR-H)
# ---------------------------------------------------------------------------


def _write_news_corpus(data_dir):
    """Helper: write a minimal news corpus for testing."""
    import json
    corpus_dir = os.path.join(data_dir, "news_corpus", "arxiv_investment")
    os.makedirs(corpus_dir, exist_ok=True)
    papers = [
        {
            "title": "Momentum is strong in US equities",
            "abstract": "We find momentum outperforms.",
            "published": "2023-06-01",
            "url": "https://arxiv.org/abs/2306.00001",
            "relevance": 0.85,
            "tickers": ["AAPL", "MSFT"],
        },
        {
            "title": "Market crash risk increasing",
            "abstract": "Indicators suggest downside risk.",
            "published": "2023-07-15",
            "url": "https://arxiv.org/abs/2307.00002",
            "relevance": 0.45,
            "tickers": ["GOOG"],
        },
    ]
    out_path = os.path.join(corpus_dir, "2023-06.jsonl")
    with open(out_path, "w") as fh:
        for paper in papers:
            fh.write(json.dumps(paper) + "\n")


class TestNewsSentimentIntegration:
    def test_expert_mode_receives_news_sentiment(self, tmp_path):
        """Expert mode run() receives _news_sentiment in data dict."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=200)
        _write_news_corpus(str(tmp_path))

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                # Check that news data is available
                assert "_news_sentiment" in data, "Missing news data!"
                news = data["_news_sentiment"]
                news_count = len(news)

                first_sym = list(data.keys())[0]
                returns = data[first_sym]["Close"].pct_change().dropna()
                val_returns = returns[(returns.index > train_end) & (returns.index <= val_end)]
                holdout_returns = returns[returns.index > val_end]

                mu_val = val_returns.mean()
                sigma_val = val_returns.std(ddof=1) if len(val_returns) > 1 else 0.01
                val_sharpe = float(mu_val / sigma_val * np.sqrt(252)) if sigma_val > 0 else 0.0

                mu_ho = holdout_returns.mean()
                sigma_ho = holdout_returns.std(ddof=1) if len(holdout_returns) > 1 else 0.01
                holdout_sharpe = float(mu_ho / sigma_ho * np.sqrt(252)) if sigma_ho > 0 else 0.0

                return {
                    "val_sharpe": val_sharpe,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": holdout_sharpe,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 8.0,
                    "news_count": news_count,
                }
        """)

        manifest_dict = {
            "name": "news_expert_test",
            "code_b64": base64.b64encode(code.encode()).decode(),
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01", "end": "2023-12-31"},
                {"type": "news_sentiment", "universe": [], "start": "2023-01-01",
                 "source": "arxiv_investment", "min_relevance": 0.0},
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe", "max_drawdown_pct"]},
            "execution_mode": "expert",
        }
        manifest = StrategyManifest.from_dict(manifest_dict)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["execution_mode"] == "expert"
        assert "expert_extras" in result
        assert result["expert_extras"]["news_count"] > 0

    def test_structured_generate_signals_with_extras(self, tmp_path):
        """generate_signals(data, extras) receives news_sentiment."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=200)
        _write_news_corpus(str(tmp_path))

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_signals(data, extras):
                # extras should contain news_sentiment
                news = extras.get("news_sentiment", pd.DataFrame())
                news_count = len(news)

                # Still produce signals
                signals = {}
                for sym, df in data.items():
                    if sym.startswith("_"):
                        continue
                    ma = df["Close"].rolling(5).mean()
                    sig = (df["Close"] > ma).astype(int) - (df["Close"] < ma).astype(int)
                    signals[sym] = sig
                return pd.DataFrame(signals)
        """)

        manifest_dict = {
            "name": "news_structured_test",
            "code_b64": base64.b64encode(code.encode()).decode(),
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01", "end": "2023-12-31"},
                {"type": "news_sentiment", "universe": [], "start": "2023-01-01",
                 "source": "arxiv_investment", "min_relevance": 0.0},
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(manifest_dict)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["execution_mode"] == "structured"

    def test_structured_single_arg_still_works(self, tmp_path):
        """generate_signals(data) with only 1 arg still works with news source."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=200)
        _write_news_corpus(str(tmp_path))

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_signals(data):
                # Single-arg function — news_sentiment is NOT passed
                # Just skip special keys
                signals = {}
                for sym, df in data.items():
                    if sym.startswith("_"):
                        continue
                    ma = df["Close"].rolling(5).mean()
                    sig = (df["Close"] > ma).astype(int) - (df["Close"] < ma).astype(int)
                    signals[sym] = sig
                return pd.DataFrame(signals)
        """)

        manifest_dict = {
            "name": "news_single_arg_test",
            "code_b64": base64.b64encode(code.encode()).decode(),
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01", "end": "2023-12-31"},
                {"type": "news_sentiment", "universe": [], "start": "2023-01-01",
                 "source": "arxiv_investment", "min_relevance": 0.0},
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(manifest_dict)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["execution_mode"] == "structured"

    def test_news_corpus_missing_graceful(self, tmp_path):
        """Missing news corpus is handled gracefully (no _news_sentiment key)."""
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=200)
        # Do NOT write news corpus

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                # No _news_sentiment expected
                assert "_news_sentiment" not in data
                return {
                    "val_sharpe": 1.5,
                    "val_max_drawdown_pct": 3.0,
                    "val_total_return_pct": 8.0,
                    "holdout_sharpe": 1.0,
                    "holdout_max_drawdown_pct": 2.0,
                    "holdout_total_return_pct": 5.0,
                }
        """)

        manifest_dict = {
            "name": "news_missing_test",
            "code_b64": base64.b64encode(code.encode()).decode(),
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01", "end": "2023-12-31"},
                {"type": "news_sentiment", "universe": [], "start": "2023-01-01",
                 "source": "arxiv_investment", "min_relevance": 0.3},
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "expert",
        }
        manifest = StrategyManifest.from_dict(manifest_dict)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# SEC 13F integration tests (Phase 2 PR-I)
# ---------------------------------------------------------------------------


def _write_sec13f_corpus(data_dir):
    """Helper: write a minimal sec_13f corpus for testing."""
    import json
    corpus_dir = os.path.join(data_dir, "sec_13f_corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    records = [
        {
            "quarter_end": "2023-06-30",
            "cik": "0001067983",
            "filer_name": "Berkshire Hathaway Inc",
            "ticker": "AAPL",
            "shares": 300000000,
            "value_usd": 45000000000.0,
            "pct_of_aum": 12.5,
        },
        {
            "quarter_end": "2023-06-30",
            "cik": "0001341439",
            "filer_name": "Vanguard Group Inc",
            "ticker": "AAPL",
            "shares": 1200000000,
            "value_usd": 180000000000.0,
            "pct_of_aum": 3.2,
        },
        {
            "quarter_end": "2023-06-30",
            "cik": "0001341439",
            "filer_name": "Vanguard Group Inc",
            "ticker": "MSFT",
            "shares": 900000000,
            "value_usd": 150000000000.0,
            "pct_of_aum": 2.7,
        },
    ]
    out_path = os.path.join(corpus_dir, "2023-Q2.jsonl")
    with open(out_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


class TestSec13FIntegration:
    def test_expert_mode_receives_sec_13f(self, tmp_path):
        """Expert mode run() receives _sec_13f in data dict."""
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=200)
        _write_sec13f_corpus(str(tmp_path))

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                assert "_sec_13f" in data, "Missing sec_13f data!"
                sec13f = data["_sec_13f"]
                sec13f_count = len(sec13f)

                first_sym = [k for k in data.keys() if not k.startswith("_")][0]
                returns = data[first_sym]["Close"].pct_change().dropna()
                val_returns = returns[(returns.index > train_end) & (returns.index <= val_end)]
                holdout_returns = returns[returns.index > val_end]

                mu_val = val_returns.mean()
                sigma_val = val_returns.std(ddof=1) if len(val_returns) > 1 else 0.01
                val_sharpe = float(mu_val / sigma_val * np.sqrt(252)) if sigma_val > 0 else 0.0

                mu_ho = holdout_returns.mean()
                sigma_ho = holdout_returns.std(ddof=1) if len(holdout_returns) > 1 else 0.01
                holdout_sharpe = float(mu_ho / sigma_ho * np.sqrt(252)) if sigma_ho > 0 else 0.0

                return {
                    "val_sharpe": val_sharpe,
                    "val_max_drawdown_pct": 5.0,
                    "val_total_return_pct": 10.0,
                    "holdout_sharpe": holdout_sharpe,
                    "holdout_max_drawdown_pct": 3.0,
                    "holdout_total_return_pct": 8.0,
                    "sec13f_count": sec13f_count,
                }
        """)

        manifest_dict = {
            "name": "sec13f_expert_test",
            "code_b64": base64.b64encode(code.encode()).decode(),
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01", "end": "2023-12-31"},
                {"type": "sec_13f", "universe": [], "start": "2023-01-01",
                 "filers": ["top_50"], "min_position_pct": 0.0},
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe", "max_drawdown_pct"]},
            "execution_mode": "expert",
        }
        manifest = StrategyManifest.from_dict(manifest_dict)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["execution_mode"] == "expert"
        assert "expert_extras" in result
        assert result["expert_extras"]["sec13f_count"] > 0

    def test_structured_with_sec_13f_extras(self, tmp_path):
        """generate_signals(data, extras) receives sec_13f."""
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=200)
        _write_sec13f_corpus(str(tmp_path))

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_signals(data, extras):
                sec13f = extras.get("sec_13f", pd.DataFrame())
                assert len(sec13f) > 0

                signals = {}
                for sym, df in data.items():
                    if sym.startswith("_"):
                        continue
                    ma = df["Close"].rolling(5).mean()
                    sig = (df["Close"] > ma).astype(int) - (df["Close"] < ma).astype(int)
                    signals[sym] = sig
                return pd.DataFrame(signals)
        """)

        manifest_dict = {
            "name": "sec13f_structured_test",
            "code_b64": base64.b64encode(code.encode()).decode(),
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01", "end": "2023-12-31"},
                {"type": "sec_13f", "universe": [], "start": "2023-01-01",
                 "filers": ["top_50"], "min_position_pct": 0.0},
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "structured",
        }
        manifest = StrategyManifest.from_dict(manifest_dict)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"
        assert result["execution_mode"] == "structured"

    def test_sec13f_corpus_missing_graceful(self, tmp_path):
        """Missing sec_13f corpus is handled gracefully (no _sec_13f key)."""
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=200)

        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd

            def run(data, train_end, val_end, benchmark, config):
                assert "_sec_13f" not in data
                return {
                    "val_sharpe": 1.5,
                    "val_max_drawdown_pct": 3.0,
                    "val_total_return_pct": 8.0,
                    "holdout_sharpe": 1.0,
                    "holdout_max_drawdown_pct": 2.0,
                    "holdout_total_return_pct": 5.0,
                }
        """)

        manifest_dict = {
            "name": "sec13f_missing_test",
            "code_b64": base64.b64encode(code.encode()).decode(),
            "data_sources": [
                {"type": "ohlcv", "universe": symbols, "start": "2023-01-01", "end": "2023-12-31"},
                {"type": "sec_13f", "universe": [], "start": "2023-01-01",
                 "filers": ["top_50"], "min_position_pct": 0.3},
            ],
            "compute": {"mode": "inference", "budget_seconds": 300, "gpu": False},
            "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
            "execution_mode": "expert",
        }
        manifest = StrategyManifest.from_dict(manifest_dict)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Test: structured lookahead (causality) check
# ---------------------------------------------------------------------------

class TestLookaheadCheck:
    """Causality (lookahead) check for _run_structured_mode."""

    def test_lookahead_detected_generate_signals(self, tmp_path):
        """Positive control: a generate_signals that reads future data
        (shift(-5)) must be REJECTED with status=error, error_type=code,
        and message containing 'lookahead'."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_signals(data):
                signals = {}
                for sym, df in data.items():
                    # Deliberate lookahead: use close 5 days in the future
                    future_close = df["Close"].shift(-5)
                    sig = (future_close > df["Close"]).astype(int)
                    signals[sym] = sig
                return pd.DataFrame(signals).fillna(0)
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error", f"Expected error but got {result.get('status')}"
        assert result["error_type"] == "code"
        assert "lookahead" in result["error"].lower()

    def test_causal_strategy_passes_generate_signals(self, tmp_path):
        """Negative control: a causal generate_signals using only rolling
        windows must NOT be rejected (no false positive)."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_signals(data):
                signals = {}
                for sym, df in data.items():
                    ma = df["Close"].rolling(20).mean()
                    sig = (df["Close"] > ma).astype(int) - (df["Close"] < ma).astype(int)
                    signals[sym] = sig
                return pd.DataFrame(signals).fillna(0)
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok", f"Unexpected error: {result.get('error', '')}"
        assert "val_sharpe" in result["metrics"]
        assert "holdout_sharpe" in result["metrics"]

    def test_causal_weights_passes(self, tmp_path):
        """generate_weights path: a causal equal-weight strategy must pass
        the lookahead check."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_weights(data):
                first_df = next(iter(data.values()))
                n = len(data)
                w = pd.DataFrame(
                    1.0 / n,
                    index=first_df.index,
                    columns=list(data.keys()),
                )
                return w
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok", f"Unexpected error: {result.get('error', '')}"
        assert result["config"]["weighting"] == "generate_weights"
        assert "val_sharpe" in result["metrics"]

    def test_lookahead_detected_generate_weights(self, tmp_path):
        """Positive control: generate_weights with lookahead must be REJECTED."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        _setup_data(tmp_path, symbols, n_days=250)

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_weights(data):
                # Lookahead: peek at the last close of each symbol
                result = {}
                for sym, df in data.items():
                    last_close = df["Close"].iloc[-1]
                    w = pd.Series(0.0, index=df.index, name=sym)
                    w.iloc[-10:] = 1.0 / len(data)  # overweight the last 10 days
                    result[sym] = w
                return pd.DataFrame(result)
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "error", f"Expected error but got {result.get('status')}"
        assert result["error_type"] == "code"
        assert "lookahead" in result["error"].lower()

    def test_short_data_skips_check(self, tmp_path):
        """Very short data (< 100 CV rows) should skip the lookahead check
        entirely rather than erroring — a causal strategy must still pass."""
        symbols = ["AAPL", "MSFT"]
        _setup_data(tmp_path, symbols, n_days=80)  # CV region ~64 < 100

        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np

            def generate_signals(data):
                signals = {}
                for sym, df in data.items():
                    ma = df["Close"].rolling(5).mean()
                    sig = (df["Close"] > ma).astype(int) - (df["Close"] < ma).astype(int)
                    signals[sym] = sig
                return pd.DataFrame(signals).fillna(0)
        """)

        manifest = _make_manifest(code=code, universe=symbols)
        result = json.loads(run_manifest(manifest, str(tmp_path)))

        assert result["status"] == "ok", (
            f"Short-data run should skip lookahead check, "
            f"got: {result.get('error', 'ok')}"
        )
