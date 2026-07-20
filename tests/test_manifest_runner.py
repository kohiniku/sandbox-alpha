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
        assert result["error_type"] == "infra"
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

        assert result["status"] == "error"
        assert result["error_type"] == "code"
        assert "not finite" in result["error"].lower()

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
