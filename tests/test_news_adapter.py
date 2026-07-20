"""Tests for data_adapters/news_sentiment.py (Phase 2 PR-H)."""

import json
import os
import textwrap

import pandas as pd
import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_adapters.news_sentiment import (
    load_news_sentiment,
    _score_sentiment,
    _compute_relevance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_corpus(tmp_path, source="arxiv_investment", papers=None):
    """Write a minimal JSONL corpus for testing."""
    if papers is None:
        papers = [
            {
                "title": "Momentum is strong in US equities",
                "abstract": "We find momentum outperforms.",
                "published": "2025-01-15",
                "url": "https://arxiv.org/abs/2501.00001",
                "relevance": 0.85,
                "tickers": ["AAPL", "MSFT"],
            },
            {
                "title": "Market crash risk increasing",
                "abstract": "Indicators suggest downside risk.",
                "published": "2025-02-10",
                "url": "https://arxiv.org/abs/2502.00002",
                "relevance": 0.45,
                "tickers": ["GOOG"],
            },
            {
                "title": "Neutral analysis of trading patterns",  # no strong sentiment keywords
                "abstract": "We analyze daily trading data.",
                "published": "2025-03-20",
                "url": "https://arxiv.org/abs/2503.00003",
                "relevance": 0.25,
                "tickers": [],
            },
            {
                "title": "Innovation breakthrough in weak markets",
                "abstract": "A recovery signal.",
                "published": "2025-04-01",
                "url": "https://arxiv.org/abs/2504.00004",
                "relevance": 0.10,
                "tickers": ["TSLA"],
            },
        ]

    corpus_dir = os.path.join(tmp_path, "news_corpus", source)
    os.makedirs(corpus_dir, exist_ok=True)
    out_path = os.path.join(corpus_dir, "2025-01.jsonl")
    with open(out_path, "w") as fh:
        for paper in papers:
            fh.write(json.dumps(paper) + "\n")


# ---------------------------------------------------------------------------
# _score_sentiment
# ---------------------------------------------------------------------------

class TestScoreSentiment:
    def test_bullish_title(self):
        """Titles with bullish keywords score positive."""
        score = _score_sentiment("Strong growth and momentum outperform the market")
        assert score > 0.0

    def test_bearish_title(self):
        """Titles with bearish keywords score negative."""
        score = _score_sentiment("Market crash and recession risk loom large")
        assert score < 0.0

    def test_neutral_title(self):
        """Titles without sentiment keywords score zero."""
        score = _score_sentiment("An empirical analysis of daily trading patterns")
        assert score == 0.0

    def test_mixed_terms_partial_cancel(self):
        """Mixed bullish/bearish terms partially cancel."""
        score = _score_sentiment("Outperformance and recession both observed")
        # Both terms present, should be near zero
        assert abs(score) < 0.6

    def test_score_in_range(self):
        """All scores are within [-1, 1]."""
        titles = [
            "Super bullish rally gain surge strong outperform",
            "Bearish crisis crash default bankruptcy distress",
            "Normal market analysis paper",
        ]
        for title in titles:
            score = _score_sentiment(title)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for: {title}"


# ---------------------------------------------------------------------------
# _compute_relevance
# ---------------------------------------------------------------------------

class TestComputeRelevance:
    def test_finance_content_scores_high(self):
        score = _compute_relevance(
            "A novel momentum trading strategy",
            "We use portfolio construction and factor timing to generate alpha.",
        )
        assert score > 0.3

    def test_non_finance_content_scores_low(self):
        score = _compute_relevance(
            "Quantum computing advances",
            "We demonstrate a new qubit architecture.",
        )
        assert score < 0.3

    def test_relevance_in_range(self):
        score = _compute_relevance("Title", "Abstract")
        assert 0.0 <= score <= 0.95


# ---------------------------------------------------------------------------
# load_news_sentiment
# ---------------------------------------------------------------------------

class TestLoadNewsSentiment:
    def test_basic_loading(self, tmp_path):
        """Basic load with no universe filter."""
        _write_corpus(str(tmp_path))
        df = load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.0,
            data_dir=str(tmp_path),
        )
        assert len(df) > 0
        expected_cols = ["date", "ticker", "headline", "source_url", "relevance_score", "sentiment_score"]
        assert list(df.columns) == expected_cols

    def test_min_relevance_filter(self, tmp_path):
        """min_relevance filters out low-relevance papers."""
        _write_corpus(str(tmp_path))
        df_all = load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.0,
            data_dir=str(tmp_path),
        )
        df_filtered = load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.5,
            data_dir=str(tmp_path),
        )
        # With min_relevance=0.5, only papers with relevance >= 0.5 pass
        assert len(df_filtered) < len(df_all)

    def test_universe_filter(self, tmp_path):
        """Universe filter restricts to matching tickers."""
        _write_corpus(str(tmp_path))
        df = load_news_sentiment(
            universe=["AAPL"],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.0,
            data_dir=str(tmp_path),
        )
        assert len(df) > 0
        # Only AAPL rows
        assert set(df["ticker"].unique()) == {"AAPL"}

    def test_date_range_filter(self, tmp_path):
        """Date range filter works correctly."""
        _write_corpus(str(tmp_path))
        df = load_news_sentiment(
            universe=[],
            start="2025-03-01",  # only >= March
            end="2025-03-31",
            source="arxiv_investment",
            min_relevance=0.0,
            data_dir=str(tmp_path),
        )
        assert len(df) > 0
        # All dates should be in March 2025
        for d in df["date"]:
            assert "2025-03" in str(d)

    def test_empty_corpus_returns_empty_df(self, tmp_path):
        """Missing corpus dir returns empty DataFrame gracefully."""
        df = load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.3,
            data_dir=str(tmp_path),  # no corpus dir exists
        )
        assert df.empty
        expected_cols = ["date", "ticker", "headline", "source_url", "relevance_score", "sentiment_score"]
        assert list(df.columns) == expected_cols

    def test_empty_corpus_no_jsonl_files(self, tmp_path):
        """Empty corpus dir returns empty DataFrame."""
        corpus_dir = os.path.join(str(tmp_path), "news_corpus", "arxiv_investment")
        os.makedirs(corpus_dir, exist_ok=True)
        df = load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.3,
            data_dir=str(tmp_path),
        )
        assert df.empty

    def test_date_index_compatible(self, tmp_path):
        """Date column is DatetimeIndex-compatible UTC naive."""
        _write_corpus(str(tmp_path))
        df = load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.0,
            data_dir=str(tmp_path),
        )
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_sentiment_column_in_range(self, tmp_path):
        """All sentiment scores in [-1, 1]."""
        _write_corpus(str(tmp_path))
        df = load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.0,
            data_dir=str(tmp_path),
        )
        assert ((df["sentiment_score"] >= -1.0) & (df["sentiment_score"] <= 1.0)).all()

    def test_general_arxiv_source(self, tmp_path):
        """general_arxiv source reads from different directory."""
        _write_corpus(str(tmp_path), source="general_arxiv")
        df = load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="general_arxiv",
            min_relevance=0.0,
            data_dir=str(tmp_path),
        )
        assert len(df) > 0

    def test_empty_corpus_logs_warning(self, tmp_path, caplog):
        """Missing corpus dir logs a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        load_news_sentiment(
            universe=[],
            start="2025-01-01",
            end=None,
            source="arxiv_investment",
            min_relevance=0.3,
            data_dir=str(tmp_path),
        )
        assert "News corpus directory not found" in caplog.text
