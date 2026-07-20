#!/usr/bin/env python3
"""
News sentiment data adapter for sandbox-alpha v2 (Phase 2 PR-H).

Reads a pre-fetched arxiv paper corpus from JSONL files on disk and returns
a pandas DataFrame with headline-level sentiment scores.

No network calls at runtime. Sentiment is scored by a simple keyword-based
lexicon (see _score_sentiment). A real ML model would replace this later.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword-based sentiment lexicon
# ---------------------------------------------------------------------------
# Design note (Phase 2 PR-H):
#   This is a deliberately simple keyword scorer. Each paper title (headline)
#   is tokenized into lowercase words. Bullish/bearish terms contribute
#   +1/-1 respectively. The final score is tanh(sum / (1 + n_terms)) so
#   longer titles with many keywords dampen towards sensible [-1, 1].
#
#   Future work: replace with a BERT-era finBERT or a LLM-based few-shot
#   scorer, once the pipeline matures to Phase 3.

_BULLISH_TERMS: set = {
    "outperform", "beat", "alpha", "profitable", "profitability",
    "momentum", "growth", "positive", "bullish", "upward", "rally",
    "gain", "surge", "strong", "outperformance", "excess",
    "premium", "long", "buy", "overweight", "upside",
    "improve", "improvement", "recovery", "recover", "rebound",
    "optimistic", "favorable", "boost", "accelerate",
    "breakthrough", "innovation", "innovative",
}

_BEARISH_TERMS: set = {
    "underperform", "loss", "risk", "crash", "decline", "negative",
    "bearish", "downward", "sell-off", "selloff", "downturn",
    "recession", "contraction", "volatile", "volatility",
    "bubble", "overvalued", "short", "underweight", "downside",
    "deteriorate", "deterioration", "drop", "plunge", "tumble",
    "pessimistic", "unfavorable", "weak", "weakness",
    "crisis", "default", "bankruptcy", "distress",
}

# Neutral-ish but finance-relevant stop words that shouldn't swing score
_NEUTRAL_TERMS: set = {
    "market", "stock", "price", "return", "returns", "trading",
    "investor", "investment", "portfolio", "asset", "equity",
    "data", "model", "paper", "evidence", "study", "analysis",
    "effect", "impact", "factor", "strategy", "strategies",
    "index", "fund", "rate", "rates", "term", "short-term",
    "long-term", "cross-section", "cross-sectional",
}


def _score_sentiment(text: str) -> float:
    """Score sentiment of a single headline/abstract title.

    Returns a float in [-1, 1].

    Algorithm
    ---------
    1. Lowercase and split on whitespace/punctuation.
    2. Count bullish and bearish keyword hits.
    3. ``raw = bullish - bearish``.
    4. ``tanh(raw / (1.0 + total_keyword_hits))`` — longer texts with
       many keywords dampen so a short title like "Market crash looms"
       scores strongly negative (~ -0.76), while a long rambling title
       with mixed terms scores near 0.
    """
    # Simple tokenization: lowercase, split on non-alphanumeric
    tokens: List[str] = []
    for word in text.lower().replace("-", " ").replace("_", " ").split():
        # Strip punctuation from each token
        token = "".join(ch for ch in word if ch.isalnum())
        if token:
            tokens.append(token)

    bullish = sum(1 for t in tokens if t in _BULLISH_TERMS)
    bearish = sum(1 for t in tokens if t in _BEARISH_TERMS)
    total = bullish + bearish

    if total == 0:
        return 0.0

    raw = bullish - bearish
    return float(np.tanh(raw / (1.0 + total)))


def _compute_relevance(title: str, abstract: str) -> float:
    """Compute a simple relevance score for a paper.

    Returns a float in [0.0, 1.0]. Scores higher when the title or abstract
    contains finance-related keywords (investment, trading, portfolio, stock,
    return, alpha, etc.). This is a placeholder — real relevance would use
    embeddings or a classifier.

    This function is used during ingest to pre-compute relevance scores.
    At load time, ``min_relevance`` filters on this pre-computed field.
    """
    finance_keywords = {
        "trading", "portfolio", "stock", "return", "alpha", "beta",
        "factor", "momentum", "volatility", "risk", "sharpe",
        "market", "equity", "asset", "pricing", "investment",
        "investor", "arbitrage", "hedge", "option", "futures",
        "quantitative", "quant", "strategy", "signal",
        "cross-section", "time-series", "predict", "prediction",
        "forecast", "anomaly", "premium",
    }
    combined = (title + " " + abstract).lower()
    hits = sum(1 for kw in finance_keywords if kw in combined)
    # Sigmoid-ish: more keywords → higher score, max ~0.95 at 10+ hits
    return min(float(hits) / (float(hits) + 5.0), 0.95)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_news_sentiment(
    universe: List[str],
    start: str,
    end: Optional[str],
    source: str,
    min_relevance: float,
    data_dir: str,
) -> pd.DataFrame:
    """Load news sentiment data from a pre-fetched JSONL corpus.

    Parameters
    ----------
    universe : list[str]
        Ticker symbols to filter on. Empty list = no ticker filter (all rows).
        Each paper may be tagged with a list of tickers; rows whose tickers
        intersect with ``universe`` are included. When ``universe`` is empty,
        all rows are included regardless of ticker tags.
    start : str
        Inclusive start date, ISO format "YYYY-MM-DD".
    end : str or None
        Inclusive end date "YYYY-MM-DD". None = through last row.
    source : str
        Corpus source key: ``"arxiv_investment"`` or ``"general_arxiv"``.
        Maps to ``/data/news_corpus/{source}/*.jsonl``.
    min_relevance : float
        Minimum relevance score (0.0–1.0). Rows with relevance below this
        threshold are silently dropped.
    data_dir : str
        Root data directory (usually ``/data``). The corpus lives at
        ``{data_dir}/news_corpus/{source}/*.jsonl``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, ticker, headline, source_url, relevance_score,
        sentiment_score]``.
        - ``date``: DatetimeIndex-compatible (UTC naive).
        - ``relevance_score``: float in [0, 1].
        - ``sentiment_score``: float in [-1, 1].
        Empty DataFrame if no corpus files are found (graceful degradation).
    """
    corpus_dir = os.path.join(data_dir, "news_corpus", source)
    if not os.path.isdir(corpus_dir):
        logger.warning(
            "News corpus directory not found: %s. "
            "Returning empty DataFrame (run scripts/ingest_arxiv_papers.py first).",
            corpus_dir,
        )
        return _empty_df()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else None
    universe_set: Optional[set] = set(universe) if universe else None

    rows: list = []
    jsonl_files = sorted(
        f for f in os.listdir(corpus_dir) if f.endswith(".jsonl")
    )

    if not jsonl_files:
        logger.warning(
            "No .jsonl files in %s. Returning empty DataFrame.", corpus_dir
        )
        return _empty_df()

    for filename in jsonl_files:
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line in %s", filepath)
                    continue

                # Filter by relevance
                rel = float(rec.get("relevance", 0.0))
                if rel < min_relevance:
                    continue

                # Filter by date
                pub = rec.get("published", "")
                if not pub:
                    continue
                try:
                    pub_ts = pd.Timestamp(pub)
                except (ValueError, TypeError):
                    continue
                if pub_ts < start_ts:
                    continue
                if end_ts is not None and pub_ts > end_ts:
                    continue

                # Filter by ticker
                tickers: List[str] = rec.get("tickers", [])
                if universe_set is not None and tickers:
                    # Only keep tickers that are in the universe
                    filtered = [t for t in tickers if t in universe_set]
                    if not filtered:
                        continue
                elif universe_set is not None and not tickers:
                    # Paper has no ticker tags but universe is specified — skip
                    continue

                # Score sentiment
                title = rec.get("title", "")
                sentiment = _score_sentiment(title)

                # Determine tickers to emit: if universe filter active,
                # only emit tickers in the intersection; otherwise emit all
                emit_tickers = (
                    [t for t in tickers if t in universe_set]
                    if universe_set and tickers
                    else tickers
                )

                # One row per ticker
                for t in (emit_tickers or ["__market__"]):
                    rows.append({
                        "date": pub_ts,
                        "ticker": t,
                        "headline": title,
                        "source_url": rec.get("url", ""),
                        "relevance_score": rel,
                        "sentiment_score": sentiment,
                    })

    if not rows:
        return _empty_df()

    df = pd.DataFrame(rows)

    # Ensure date is DatetimeIndex-compatible UTC naive
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    # Sort
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Reorder columns
    return df[
        ["date", "ticker", "headline", "source_url", "relevance_score", "sentiment_score"]
    ]


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard schema."""
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "headline",
            "source_url",
            "relevance_score",
            "sentiment_score",
        ]
    )
