"""Tests for scripts/ingest_arxiv_papers.py."""

import json
import os
import sys
import tempfile

import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_arxiv_papers import (
    _parse_markdown_file,
    _extract_tickers,
    ingest,
)


# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "sample_arxiv_output.md"
)


# ---------------------------------------------------------------------------
# _extract_tickers
# ---------------------------------------------------------------------------

class TestExtractTickers:
    def test_extract_from_text(self):
        text = "AAPL and MSFT show momentum, while GOOG underperforms"
        tickers = _extract_tickers(text)
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOG" in tickers

    def test_filter_known_tickers(self):
        text = "AAPL and XYZ show momentum"
        tickers = _extract_tickers(text, known_tickers={"AAPL", "MSFT"})
        assert tickers == ["AAPL"]  # XYZ not in known

    def test_dedup_order_preserved(self):
        text = "AAPL AAPL MSFT"
        tickers = _extract_tickers(text)
        assert tickers == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# _parse_markdown_file
# ---------------------------------------------------------------------------

class TestParseMarkdownFile:
    def test_parse_fixture(self):
        """Parse the sample fixture and extract expected papers."""
        papers = _parse_markdown_file(FIXTURE_PATH)
        assert len(papers) >= 3  # at least 3 papers

        # Check key fields
        titles = {p["title"] for p in papers}
        assert "Stock Return Prediction with Machine Learning" in titles
        assert "Factor Investing in Volatile Markets" in titles

        # Check URL extraction
        urls = {p["url"] for p in papers}
        assert "https://arxiv.org/abs/2501.00123" in urls
        assert "https://arxiv.org/abs/2502.00456" in urls

        # Check published dates
        dates = {p["published"] for p in papers}
        assert "2025-01-15" in dates
        assert "2025-03-10" in dates

    def test_papers_have_abstract(self):
        """Papers parsed from fixture have content."""
        papers = _parse_markdown_file(FIXTURE_PATH)
        for paper in papers:
            # At least one of title or abstract should be non-empty
            assert paper["title"] or paper["abstract"] or paper["url"]


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestIngestRoundtrip:
    def test_ingest_dry_run(self, tmp_path):
        """Dry run parses fixture and reports correct counts."""
        # Copy fixture to tmp_path
        import shutil
        fixture_dir = str(tmp_path / "input")
        os.makedirs(fixture_dir, exist_ok=True)
        shutil.copy(FIXTURE_PATH, fixture_dir)

        stats = ingest(
            input_dir=fixture_dir,
            output_dir=str(tmp_path / "output"),
            dry_run=True,
        )
        assert stats["files_processed"] == 1
        assert stats["papers_ingested"] >= 3

    def test_ingest_writes_jsonl(self, tmp_path):
        """Full ingest writes JSONL files."""
        import shutil
        fixture_dir = str(tmp_path / "input")
        os.makedirs(fixture_dir, exist_ok=True)
        shutil.copy(FIXTURE_PATH, fixture_dir)

        output_dir = str(tmp_path / "output")

        stats = ingest(
            input_dir=fixture_dir,
            output_dir=output_dir,
            dry_run=False,
        )
        assert stats["papers_ingested"] >= 3

        # Verify JSONL files exist
        from pathlib import Path
        jsonl_files = sorted(Path(output_dir).glob("*.jsonl"))
        assert len(jsonl_files) > 0

        # Verify content
        for jf in jsonl_files:
            with open(jf, "r") as fh:
                for line in fh:
                    rec = json.loads(line.strip())
                    assert "title" in rec
                    assert "url" in rec
                    assert "relevance" in rec
                    assert 0.0 <= rec["relevance"] <= 1.0

    def test_ingest_dedup_on_rerun(self, tmp_path):
        """Re-running ingest is idempotent (no duplicate papers by URL)."""
        import shutil
        fixture_dir = str(tmp_path / "input")
        os.makedirs(fixture_dir, exist_ok=True)
        shutil.copy(FIXTURE_PATH, fixture_dir)

        output_dir = str(tmp_path / "output")

        # First run
        stats1 = ingest(
            input_dir=fixture_dir,
            output_dir=output_dir,
            dry_run=False,
        )

        # Second run (same input)
        stats2 = ingest(
            input_dir=fixture_dir,
            output_dir=output_dir,
            dry_run=False,
        )

        # Total papers should be same (idempotent)
        assert stats2["papers_ingested"] == stats1["papers_ingested"]

        # Verify no duplicates by URL in output files
        from pathlib import Path
        from collections import Counter
        all_urls = []
        for jf in sorted(Path(output_dir).glob("*.jsonl")):
            with open(jf, "r") as fh:
                for line in fh:
                    rec = json.loads(line.strip())
                    all_urls.append(rec["url"])
        url_counts = Counter(all_urls)
        duplicates = {url: count for url, count in url_counts.items() if count > 1}
        assert len(duplicates) == 0, f"Found duplicate URLs: {duplicates}"

    def test_ingest_missing_dir(self, capsys):
        """Non-existent input dir exits with error."""
        with pytest.raises(SystemExit):
            ingest(
                input_dir="/nonexistent/path/to/nowhere",
                output_dir="/tmp",
            )
