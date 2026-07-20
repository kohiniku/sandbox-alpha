#!/usr/bin/env python3
"""
Ingest arxiv paper collector cron output into JSONL corpus.

Reads markdown output files from the Hermes arxiv paper collector cron job
(e.g. ~/hermes-secure/data/cron/output/90918ca2f5e6/*.md) and converts them
to per-month JSONL files under /data/news_corpus/arxiv_investment/.

Usage
-----
    python scripts/ingest_arxiv_papers.py --input-dir /path/to/cron/output \\
        --output-dir /data/news_corpus/arxiv_investment

Schema (per JSONL line)
------------------------
    {
        "title": "Paper title",
        "abstract": "Full abstract text",
        "published": "YYYY-MM-DD",
        "url": "https://arxiv.org/abs/...",
        "relevance": 0.0-1.0,
        "tickers": ["AAPL", "MSFT"]
    }

Idempotent: re-running deduplicates on ``url`` within each month's file.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_adapters.news_sentiment import _compute_relevance


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

_ARXIV_URL_RE = re.compile(
    r"https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)"
)
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Ticker patterns for extracting from text: uppercase letter followed by
# alphanumeric/dot/dash, 1-12 chars (matching SYMBOL_REGEX in manifest.py)
_TICKER_RE = re.compile(
    r"\b([A-Z][A-Z0-9.\-]{0,11})\b"
)


def _extract_tickers(text: str, known_tickers: set = None) -> list:
    """Extract ticker-like tokens from text. Only returns known_tickers if set."""
    candidates = _TICKER_RE.findall(text)
    if known_tickers:
        return list({c for c in candidates if c in known_tickers})
    # Otherwise just return unique matches that look like tickers
    return list(dict.fromkeys(candidates))  # dedup, preserve order


def _parse_markdown_file(filepath: str) -> list:
    """Parse a single arxiv cron markdown file into a list of paper dicts.

    Expected format (per paper block):
        ## Paper Title
        - **Published**: YYYY-MM-DD
        - **URL**: https://arxiv.org/abs/NNNN.NNNNN
        Abstract text follows...

    Returns list of dicts with keys: title, abstract, published, url.
    """
    papers = []
    with open(filepath, "r", encoding="utf-8") as fh:
        content = fh.read()

    # Split on "## " heading
    blocks = re.split(r"\n(?=## )", content)

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        # First line is the title (## Title)
        title = lines[0].lstrip("#").strip()
        if not title:
            continue

        url = ""
        published = ""
        abstract_lines = []
        in_abstract = False

        for line in lines[1:]:
            line_stripped = line.strip()
            # Check for URL
            url_match = _ARXIV_URL_RE.search(line_stripped)
            if url_match:
                url = url_match.group(0).replace("/pdf/", "/abs/")
                continue
            # Check for published date
            if "Published" in line_stripped or "published" in line_stripped:
                date_match = _DATE_RE.search(line_stripped)
                if date_match:
                    published = date_match.group(0)
                continue
            # Check for abstract heading
            if line_stripped.lower().startswith("abstract") or line_stripped == "**Abstract**":
                in_abstract = True
                # If there's text after "Abstract:", capture it
                rest = re.sub(r"(?i)^\*?\*?abstract:?\*?\*?\s*", "", line_stripped)
                if rest:
                    abstract_lines.append(rest)
                continue
            # Skip markdown formatting markers
            if line_stripped in ("---", "***"):
                continue
            if in_abstract or line_stripped:
                # Heuristic: if we're past the metadata, everything is abstract
                if not in_abstract and not published and not url:
                    in_abstract = True
                if in_abstract and line_stripped and not line_stripped.startswith("**"):
                    abstract_lines.append(line_stripped)

        abstract = " ".join(abstract_lines).strip()

        papers.append({
            "title": title,
            "abstract": abstract,
            "published": published,
            "url": url,
        })

    return papers


# ---------------------------------------------------------------------------
# Main ingest pipeline
# ---------------------------------------------------------------------------


def ingest(
    input_dir: str,
    output_dir: str,
    known_tickers: set = None,
    dry_run: bool = False,
) -> dict:
    """Scan input_dir for .md files, parse, compute relevance, write per-month JSONL.

    Returns stats dict.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(input_path.glob("*.md"))
    if not md_files:
        print(f"Warning: no .md files found in {input_dir}", file=sys.stderr)
        return {"files_processed": 0, "papers_ingested": 0, "dedup_skipped": 0}

    # Collect all papers from all markdown files
    all_papers = []
    for md_file in md_files:
        parsed = _parse_markdown_file(str(md_file))
        for paper in parsed:
            if paper["url"] and paper["title"]:
                # Compute relevance
                paper["relevance"] = round(
                    _compute_relevance(paper["title"], paper["abstract"]), 3
                )
                # Extract tickers
                paper["tickers"] = _extract_tickers(
                    paper["title"] + " " + paper["abstract"], known_tickers
                )
                all_papers.append(paper)

    # Group by month (YYYY-MM) and dedup by URL within each month
    by_month: dict[str, dict[str, dict]] = defaultdict(dict)

    for paper in all_papers:
        published = paper.get("published", "")
        if not published:
            month = "unknown"
        else:
            month = published[:7]  # YYYY-MM

        url = paper["url"]
        # Dedup: keep the one with more complete data
        if url in by_month[month]:
            existing = by_month[month][url]
            # Prefer entries with abstract and non-empty published date
            if paper["abstract"] and not existing["abstract"]:
                by_month[month][url] = paper
            elif paper["published"] and not existing["published"]:
                by_month[month][url] = paper
        else:
            by_month[month][url] = paper

    # Count dedup
    total_before = len(all_papers)
    total_after = sum(len(month_papers) for month_papers in by_month.values())
    dedup_skipped = total_before - total_after

    if dry_run:
        print(f"[DRY RUN] Would ingest {total_after} papers into {len(by_month)} month files")
        print(f"  Dedup skipped: {dedup_skipped}")
        return {
            "files_processed": len(md_files),
            "papers_ingested": total_after,
            "dedup_skipped": dedup_skipped,
        }

    # Write per-month JSONL
    os.makedirs(str(output_path), exist_ok=True)
    for month, papers in by_month.items():
        out_file = output_path / f"{month}.jsonl"
        # Read existing lines for cross-run dedup
        existing_urls = set()
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("url"):
                            existing_urls.add(rec["url"])
                    except json.JSONDecodeError:
                        pass

        written = 0
        with open(out_file, "w", encoding="utf-8") as fh:
            # Rewrite existing entries
            if out_file.exists():
                # We'll need to read all existing non-skipped entries first
                existing_entries = []
                with open(out_file, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("url") not in {p["url"] for p in papers.values() if p["url"] in existing_urls}:
                                existing_entries.append(rec)
                        except json.JSONDecodeError:
                            pass
                for entry in existing_entries:
                    print(json.dumps(entry, ensure_ascii=False), file=fh)

            # Now write new papers
            for paper in papers.values():
                if paper["url"] in existing_urls:
                    continue
                record = {
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "published": paper["published"],
                    "url": paper["url"],
                    "relevance": paper["relevance"],
                    "tickers": paper["tickers"],
                }
                print(json.dumps(record, ensure_ascii=False), file=fh)
                written += 1

    print(f"Ingested {total_after} papers ({written} new) into {len(by_month)} month files")
    print(f"  Files scanned: {len(md_files)}")
    print(f"  Dedup skipped: {dedup_skipped}")

    return {
        "files_processed": len(md_files),
        "papers_ingested": total_after,
        "dedup_skipped": dedup_skipped,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest arxiv paper collector cron output -> JSONL corpus"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing arxiv cron .md output files",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/news_corpus/arxiv_investment",
        help="Output directory for per-month .jsonl files",
    )
    parser.add_argument(
        "--tickers-file",
        default=None,
        help="Optional file with one ticker per line (for filtering/extraction)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and compute stats but do not write files",
    )
    args = parser.parse_args()

    known_tickers = None
    if args.tickers_file and os.path.isfile(args.tickers_file):
        with open(args.tickers_file, "r") as fh:
            known_tickers = {line.strip() for line in fh if line.strip()}

    stats = ingest(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        known_tickers=known_tickers,
        dry_run=args.dry_run,
    )
    print(f"\nDone. Stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
