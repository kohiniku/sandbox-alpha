#!/usr/bin/env python3
"""
Tests for manifest.py — Strategy Manifest Schema.

Covers:
  - Round-trip serialization (to_dict / from_dict)
  - Each validation rule: positive and negative
  - Discriminated union: unknown type raises, new subclass via registry
  - No pandas/numpy imports in manifest.py
"""

import ast
import base64
import pytest

from manifest import (
    StrategyManifest,
    DataSource,
    OhlcvSource,
    ModelArtifact,
    ComputeSpec,
    EvaluatorSpec,
    ManifestValidationError,
    VALID_METRICS,
    MAX_CODE_BYTES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(code: str) -> str:
    return base64.b64encode(code.encode("utf-8")).decode("ascii")


VALID_CODE = "def generate_signals(df):\n    return df['Close'] * 0\n"
VALID_CODE_B64 = _b64(VALID_CODE)


def _minimal_dict(**overrides):
    """Return a minimal valid manifest dict."""
    d = {
        "name": "test_strategy",
        "code_b64": VALID_CODE_B64,
        "data_sources": [
            {"type": "ohlcv", "universe": ["AAPL"], "start": "2020-01-01"}
        ],
        "model_artifacts": [],
        "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
        "evaluator": {
            "type": "portfolio",
            "metrics": ["sharpe"],
        },
        "execution_mode": "structured",
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_minimal_roundtrip(self):
        """Valid minimal manifest round-trips through to_dict/from_dict."""
        d = _minimal_dict()
        m = StrategyManifest.from_dict(d)
        out = m.to_dict()
        assert out == d

    def test_full_roundtrip_with_optional_fields(self):
        """Manifest with all optional fields round-trips correctly."""
        d = {
            "name": "full_strategy",
            "code_b64": VALID_CODE_B64,
            "data_sources": [
                {"type": "ohlcv", "universe": ["AAPL", "MSFT"], "start": "2020-01-01", "end": "2023-12-31"}
            ],
            "model_artifacts": [{"name": "timesfm-base", "revision": "v1.0"}],
            "compute": {"mode": "training", "budget_seconds": 3600, "gpu": True},
            "evaluator": {
                "type": "portfolio",
                "metrics": ["sharpe", "ir", "turnover", "cvar_95", "factor_exposure"],
                "benchmark": "SPY",
                "extras": {"custom_flag": True},
            },
            "execution_mode": "structured",
        }
        m = StrategyManifest.from_dict(d)
        out = m.to_dict()
        assert out == d

    def test_roundtrip_omits_none_optional(self):
        """Optional fields with None are omitted from to_dict."""
        d = _minimal_dict()
        m = StrategyManifest.from_dict(d)
        out = m.to_dict()
        # data_source[0] should NOT have "end"
        assert "end" not in out["data_sources"][0]
        # model_artifacts should be empty list
        assert out["model_artifacts"] == []
        # evaluator should NOT have benchmark or extras
        assert "benchmark" not in out["evaluator"]
        assert "extras" not in out["evaluator"]


# ---------------------------------------------------------------------------
# name validation
# ---------------------------------------------------------------------------

class TestNameValidation:
    def test_valid_name(self):
        d = _minimal_dict(name="my_strat_01")
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_empty_name(self):
        d = _minimal_dict(name="")
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("name" in v for v in violations)

    def test_none_name(self):
        d = _minimal_dict()
        d["name"] = None
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("name" in v for v in violations)


# ---------------------------------------------------------------------------
# code_b64 validation
# ---------------------------------------------------------------------------

class TestCodeB64Validation:
    def test_valid_code_b64(self):
        d = _minimal_dict()
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_invalid_base64(self):
        d = _minimal_dict(code_b64="not-valid-base64!!!")
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("base64" in v for v in violations)

    def test_empty_code_b64(self):
        d = _minimal_dict(code_b64="")
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("code_b64" in v for v in violations)

    def test_code_too_large(self):
        # Create code that exceeds MAX_CODE_BYTES when decoded
        big_code = "x = 1\n" * (MAX_CODE_BYTES // 5 + 100)
        d = _minimal_dict(code_b64=_b64(big_code))
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("exceeds max" in v for v in violations)

    def test_code_at_limit(self):
        """Code exactly at the limit is valid."""
        code = "x" * MAX_CODE_BYTES
        d = _minimal_dict(code_b64=_b64(code))
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert not any("exceeds max" in v for v in violations)


# ---------------------------------------------------------------------------
# DataSource / OhlcvSource validation
# ---------------------------------------------------------------------------

class TestDataSourceValidation:
    def test_valid_ohlcv(self):
        d = _minimal_dict()
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_multiple_symbols(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "ohlcv", "universe": ["AAPL", "MSFT", "GOOGL"], "start": "2020-01-01"}
        ]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_invalid_symbol(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "ohlcv", "universe": ["invalid symbol!"], "start": "2020-01-01"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("symbol regex" in v for v in violations)

    def test_empty_universe(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "ohlcv", "universe": [], "start": "2020-01-01"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("universe" in v for v in violations)

    def test_bad_start_date(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "ohlcv", "universe": ["AAPL"], "start": "2020/01/01"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("start" in v and "YYYY-MM-DD" in v for v in violations)

    def test_bad_end_date(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "ohlcv", "universe": ["AAPL"], "start": "2020-01-01", "end": "not-a-date"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("end" in v and "YYYY-MM-DD" in v for v in violations)

    def test_valid_end_date(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "ohlcv", "universe": ["AAPL"], "start": "2020-01-01", "end": "2023-12-31"}
        ]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_symbol_with_dash(self):
        """BTC-USD style symbols are valid."""
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "ohlcv", "universe": ["BTC-USD"], "start": "2020-01-01"}
        ]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_symbol_with_dot(self):
        """BRK.B style symbols are valid."""
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "ohlcv", "universe": ["BRK.B"], "start": "2020-01-01"}
        ]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

class TestDiscriminatedUnion:
    def test_unknown_type_raises(self):
        """Unknown data_source.type raises ManifestValidationError during from_dict."""
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "unknown_future_type", "foo": "bar"}
        ]
        with pytest.raises(ManifestValidationError, match="unknown data_source.type"):
            StrategyManifest.from_dict(d)

    def test_missing_type_raises(self):
        """data_source without 'type' raises."""
        d = _minimal_dict()
        d["data_sources"] = [{"universe": ["AAPL"]}]
        with pytest.raises(ManifestValidationError, match="missing 'type'"):
            StrategyManifest.from_dict(d)

    def test_register_new_subclass(self):
        """Adding a new subclass through the registry works."""
        @DataSource.register("test_news_type")
        class TestNewsSource(DataSource):
            def __init__(self, corpus_url="", start=""):
                super().__init__(source_type="test_news_type")
                self.corpus_url = corpus_url
                self.start = start

            @classmethod
            def _from_dict(cls, d):
                return cls(corpus_url=d.get("corpus_url", ""), start=d.get("start", ""))

            def to_dict(self):
                return {"type": self.type, "corpus_url": self.corpus_url, "start": self.start}

        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "test_news_type", "corpus_url": "https://example.com/corpus", "start": "2020-01-01"}
        ]
        m = StrategyManifest.from_dict(d)
        assert len(m.data_sources) == 1
        assert isinstance(m.data_sources[0], TestNewsSource)
        out = m.to_dict()
        assert out["data_sources"][0]["type"] == "test_news_type"

        # Cleanup registry
        del DataSource._registry["test_news_type"]


# ---------------------------------------------------------------------------
# ComputeSpec validation
# ---------------------------------------------------------------------------

class TestComputeValidation:
    def test_valid_inference(self):
        d = _minimal_dict()
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_valid_training(self):
        d = _minimal_dict()
        d["compute"] = {"mode": "training", "budget_seconds": 3600, "gpu": True}
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_invalid_mode(self):
        d = _minimal_dict()
        d["compute"] = {"mode": "fine_tuning", "budget_seconds": 60, "gpu": False}
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("compute.mode" in v for v in violations)

    def test_negative_budget(self):
        d = _minimal_dict()
        d["compute"] = {"mode": "inference", "budget_seconds": -1, "gpu": False}
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("budget_seconds" in v for v in violations)

    def test_non_int_budget(self):
        d = _minimal_dict()
        d["compute"] = {"mode": "inference", "budget_seconds": 60.5, "gpu": False}
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("budget_seconds" in v for v in violations)


# ---------------------------------------------------------------------------
# EvaluatorSpec validation
# ---------------------------------------------------------------------------

class TestEvaluatorValidation:
    def test_valid_portfolio(self):
        d = _minimal_dict()
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_valid_single_asset(self):
        d = _minimal_dict()
        d["evaluator"] = {"type": "single_asset", "metrics": ["sharpe", "max_drawdown_pct"]}
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_valid_custom(self):
        d = _minimal_dict()
        d["evaluator"] = {"type": "custom", "metrics": ["total_return_pct"], "extras": {"objective": "recursive_utility"}}
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_invalid_type(self):
        d = _minimal_dict()
        d["evaluator"] = {"type": "bogus", "metrics": ["sharpe"]}
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("evaluator.type" in v for v in violations)

    def test_invalid_metric(self):
        d = _minimal_dict()
        d["evaluator"] = {"type": "portfolio", "metrics": ["sharpe", "made_up_metric"]}
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("made_up_metric" in v for v in violations)

    def test_all_valid_metrics(self):
        d = _minimal_dict()
        d["evaluator"] = {"type": "portfolio", "metrics": sorted(VALID_METRICS)}
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_empty_metrics_list(self):
        """Empty metrics list is valid (evaluator implementation decides)."""
        d = _minimal_dict()
        d["evaluator"] = {"type": "portfolio", "metrics": []}
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_invalid_benchmark_symbol(self):
        d = _minimal_dict()
        d["evaluator"] = {"type": "portfolio", "metrics": ["sharpe"], "benchmark": "bad symbol!"}
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("benchmark" in v and "symbol regex" in v for v in violations)

    def test_valid_benchmark(self):
        d = _minimal_dict()
        d["evaluator"] = {"type": "portfolio", "metrics": ["sharpe"], "benchmark": "SPY"}
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []


# ---------------------------------------------------------------------------
# execution_mode validation
# ---------------------------------------------------------------------------

class TestExecutionModeValidation:
    def test_default_is_structured(self):
        d = _minimal_dict()
        m = StrategyManifest.from_dict(d)
        assert m.execution_mode == "structured"
        assert m.validate() == []

    def test_valid_expert(self):
        d = _minimal_dict(execution_mode="expert")
        m = StrategyManifest.from_dict(d)
        assert m.execution_mode == "expert"
        assert m.validate() == []

    def test_invalid_mode(self):
        d = _minimal_dict(execution_mode="experimental")
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("execution_mode" in v for v in violations)

    def test_roundtrip_preserves_mode(self):
        d = _minimal_dict(execution_mode="expert")
        m = StrategyManifest.from_dict(d)
        out = m.to_dict()
        assert out["execution_mode"] == "expert"


# ---------------------------------------------------------------------------
# ModelArtifact validation
# ---------------------------------------------------------------------------

class TestModelArtifactValidation:
    def test_valid_artifact(self):
        d = _minimal_dict()
        d["model_artifacts"] = [{"name": "timesfm-base", "revision": "v1.0"}]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_artifact_no_revision(self):
        d = _minimal_dict()
        d["model_artifacts"] = [{"name": "timesfm-base"}]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_artifact_empty_name(self):
        d = _minimal_dict()
        d["model_artifacts"] = [{"name": ""}]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("model_artifacts[0].name" in v for v in violations)


# ---------------------------------------------------------------------------
# Multiple violations collected
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    def test_all_violations_collected(self):
        """validate() does NOT stop at first violation."""
        d = _minimal_dict(name="", code_b64="!!!")
        d["compute"] = {"mode": "bogus", "budget_seconds": -1, "gpu": False}
        d["evaluator"] = {"type": "bogus", "metrics": ["nope"]}
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        # Should have multiple violations: name, code_b64, compute.mode, compute.budget_seconds, evaluator.type, evaluator.metrics
        assert len(violations) >= 5


# ---------------------------------------------------------------------------
# Contract: no pandas/numpy in manifest.py
# ---------------------------------------------------------------------------

class TestImportContract:
    def test_no_pandas_numpy_imports(self):
        """manifest.py must not import pandas or numpy."""
        import os
        manifest_path = os.path.join(os.path.dirname(__file__), "..", "manifest.py")
        with open(manifest_path) as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("pandas", "numpy"), \
                        f"manifest.py imports {alias.name} — contract violation"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    assert root not in ("pandas", "numpy"), \
                        f"manifest.py imports from {node.module} — contract violation"


# ---------------------------------------------------------------------------
# DataSource from_dict error cases
# ---------------------------------------------------------------------------

class TestDataSourceFromDictErrors:
    def test_non_dict_input(self):
        with pytest.raises(ManifestValidationError, match="must be a dict"):
            DataSource.from_dict("not a dict")

    def test_missing_type_key(self):
        with pytest.raises(ManifestValidationError, match="missing 'type'"):
            DataSource.from_dict({"universe": ["AAPL"]})


# ---------------------------------------------------------------------------
# NewsSentimentSource validation (Phase 2 PR-H)
# ---------------------------------------------------------------------------

class TestNewsSentimentSource:
    def test_valid_news_sentiment(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["AAPL"], "start": "2023-01-01",
             "source": "arxiv_investment", "min_relevance": 0.5}
        ]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_valid_empty_universe(self):
        """Empty universe = market-wide is valid."""
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": [], "start": "2023-01-01",
             "source": "arxiv_investment"}
        ]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_invalid_source_type(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["AAPL"], "start": "2023-01-01",
             "source": "invalid_source"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("source" in v and "invalid_source" in v for v in violations)

    def test_invalid_symbol_in_universe(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["bad symbol!"], "start": "2023-01-01",
             "source": "arxiv_investment"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("symbol regex" in v for v in violations)

    def test_bad_start_date(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["AAPL"], "start": "2023/01/01",
             "source": "arxiv_investment"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("start" in v and "YYYY-MM-DD" in v for v in violations)

    def test_bad_end_date(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["AAPL"], "start": "2023-01-01",
             "end": "not-a-date", "source": "arxiv_investment"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("end" in v and "YYYY-MM-DD" in v for v in violations)

    def test_min_relevance_out_of_range(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["AAPL"], "start": "2023-01-01",
             "source": "arxiv_investment", "min_relevance": 1.5}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("min_relevance" in v and "1.5" in v for v in violations)

    def test_min_relevance_negative(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["AAPL"], "start": "2023-01-01",
             "source": "arxiv_investment", "min_relevance": -0.1}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("min_relevance" in v for v in violations)

    def test_default_min_relevance(self):
        """Default min_relevance is 0.3."""
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["AAPL"], "start": "2023-01-01",
             "source": "arxiv_investment"}
        ]
        m = StrategyManifest.from_dict(d)
        ds = m.data_sources[0]
        assert ds.min_relevance == 0.3

    def test_roundtrip_news_sentiment_source(self):
        """Round-trip includes all fields."""
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "news_sentiment", "universe": ["AAPL", "MSFT"],
             "start": "2023-01-01", "end": "2024-12-31",
             "source": "general_arxiv", "min_relevance": 0.7}
        ]
        m = StrategyManifest.from_dict(d)
        out = m.to_dict()
        ds_out = out["data_sources"][0]
        assert ds_out["type"] == "news_sentiment"
        assert ds_out["source"] == "general_arxiv"
        assert ds_out["min_relevance"] == 0.7
        assert "end" in ds_out

    def test_registry_dispatch(self):
        """NewsSentimentSource is correctly dispatched by DataSource.from_dict."""
        d = {"type": "news_sentiment", "universe": ["AAPL"], "start": "2023-01-01",
             "source": "arxiv_investment"}
        from manifest import NewsSentimentSource
        ds = DataSource.from_dict(d)
        assert isinstance(ds, NewsSentimentSource)
        assert ds.source == "arxiv_investment"


# ---------------------------------------------------------------------------
# MacroSource validation (Phase 2 PR-K)
# ---------------------------------------------------------------------------

class TestMacroSource:
    def test_valid_macro_source(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": ["DGS10", "UNRATE"],
             "start": "2020-01-01", "frequency": "monthly"}
        ]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_valid_with_end_and_quarterly(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": ["DFF", "CPIAUCSL"],
             "start": "2020-01-01", "end": "2024-12-31",
             "frequency": "quarterly"}
        ]
        m = StrategyManifest.from_dict(d)
        assert m.validate() == []

    def test_valid_all_frequencies(self):
        for freq in ["daily", "weekly", "monthly", "quarterly"]:
            d = _minimal_dict()
            d["data_sources"] = [
                {"type": "macro", "series": ["DGS10"],
                 "start": "2020-01-01", "frequency": freq}
            ]
            m = StrategyManifest.from_dict(d)
            assert m.validate() == []

    def test_empty_series(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": [], "start": "2020-01-01",
             "frequency": "monthly"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("series" in v for v in violations)

    def test_invalid_frequency(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": ["DGS10"],
             "start": "2020-01-01", "frequency": "yearly"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("frequency" in v and "yearly" in v for v in violations)

    def test_bad_start_date(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": ["DGS10"],
             "start": "2020/01/01", "frequency": "monthly"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("start" in v and "YYYY-MM-DD" in v for v in violations)

    def test_bad_end_date(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": ["DGS10"],
             "start": "2020-01-01", "end": "not-a-date",
             "frequency": "monthly"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("end" in v and "YYYY-MM-DD" in v for v in violations)

    def test_empty_series_id(self):
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": ["", "UNRATE"],
             "start": "2020-01-01", "frequency": "monthly"}
        ]
        m = StrategyManifest.from_dict(d)
        violations = m.validate()
        assert any("series[0]" in v for v in violations)

    def test_default_frequency(self):
        """Default frequency is monthly."""
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": ["DGS10"],
             "start": "2020-01-01"}
        ]
        m = StrategyManifest.from_dict(d)
        ds = m.data_sources[0]
        assert ds.frequency == "monthly"
        assert m.validate() == []

    def test_roundtrip_macro_source(self):
        """Round-trip preserves all fields including end."""
        d = _minimal_dict()
        d["data_sources"] = [
            {"type": "macro", "series": ["DGS10", "DGS2", "UNRATE"],
             "start": "2020-01-01", "end": "2024-12-31",
             "frequency": "weekly"}
        ]
        m = StrategyManifest.from_dict(d)
        out = m.to_dict()
        ds_out = out["data_sources"][0]
        assert ds_out["type"] == "macro"
        assert ds_out["series"] == ["DGS10", "DGS2", "UNRATE"]
        assert ds_out["frequency"] == "weekly"
        assert "end" in ds_out

    def test_registry_dispatch(self):
        """MacroSource is correctly dispatched by DataSource.from_dict."""
        d = {"type": "macro", "series": ["DGS10", "UNRATE"],
             "start": "2020-01-01", "frequency": "monthly"}
        from manifest import MacroSource
        ds = DataSource.from_dict(d)
        assert isinstance(ds, MacroSource)
        assert ds.series == ["DGS10", "UNRATE"]
        assert ds.frequency == "monthly"
