#!/usr/bin/env python3
"""
Universe constituent provider for sandbox-alpha cross-sectional expansion.

PR 4a: Single-universe (Russell 1000), US-primary-exchange only.
Wikipedia is the constituent source; static CSV is the fallback.
Manifests are dated snapshots to support survivorship-aware backtesting.

Dependencies: pandas (already in requirements.txt).
"""

import csv
import datetime
import json
import logging
import os
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class UniverseProvider:
    """Load and refresh constituent manifests for a named universe.

    Parameters
    ----------
    name : str
        Universe name (e.g. ``"russell1000"``). Must match a Wikipedia
        constituent table page or a pre-existing manifest.
    manifest_dir : str
        Directory where ``{name}_{YYYY-MM-DD}.csv`` manifests are stored.
    """

    # Wikipedia URL for the Russell 1000 Components table.
    _WIKI_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"

    # US-primary exchanges (NYSE, NASDAQ) — used to filter ADRs.
    _US_PRIMARY_EXCHANGES = frozenset({"NMS", "NYQ", "NYSE", "NASDAQ", "Nasdaq", "New York Stock Exchange", "NYSE American", "NYSE Arca"})

    def __init__(self, name: str = "russell1000", manifest_dir: str = "data_adapters/universe_manifests"):
        self.name = name
        self.manifest_dir = os.path.abspath(manifest_dir)
        os.makedirs(self.manifest_dir, exist_ok=True)

    # -- Manifest loading -----------------------------------------------------

    def _list_manifests(self) -> List[str]:
        """Return sorted list of manifest CSV paths (oldest first)."""
        prefix = f"{self.name}_"
        files = sorted(
            os.path.join(self.manifest_dir, f)
            for f in os.listdir(self.manifest_dir)
            if f.startswith(prefix) and f.endswith(".csv")
        )
        return files

    def _manifest_for_date(self, as_of: str) -> Optional[str]:
        """Return the most recent manifest whose valid_from <= as_of.

        Returns None if no manifest covers the date.
        """
        as_of_ts = pd.Timestamp(as_of)
        candidates = []
        for path in self._list_manifests():
            filename = os.path.basename(path)
            date_str = filename[len(self.name) + 1 : -4]  # strip "{name}_" and ".csv"
            try:
                manifest_date = pd.Timestamp(date_str)
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                continue
            if manifest_date <= as_of_ts:
                candidates.append((manifest_date, path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def load_constituents(self, as_of: Optional[str] = None) -> List[dict]:
        """Return list of constituent dicts with keys:
        ``symbol``, ``name``, ``valid_from``, ``valid_until``.

        If ``as_of`` is None, the latest manifest on disk is used.
        If no manifest exists for ``as_of``, a new one is scraped and
        written to disk first.
        """
        if as_of is None:
            as_of = datetime.date.today().isoformat()

        path = self._manifest_for_date(as_of)
        if path is None:
            logger.info("No manifest covering %s; refreshing from source.", as_of)
            path = self.refresh_from_source()

        return self._read_manifest(path)

    def _read_manifest(self, path: str) -> List[dict]:
        """Read a manifest CSV and return list of dicts."""
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "symbol": row["symbol"].strip(),
                    "name": row.get("name", "").strip(),
                    "valid_from": row.get("valid_from", "").strip(),
                    "valid_until": row.get("valid_until", "").strip(),
                })
        return rows

    def get_symbols(self, as_of: Optional[str] = None) -> List[str]:
        """Return just the symbol strings for the given as_of date."""
        constituents = self.load_constituents(as_of=as_of)
        return [c["symbol"] for c in constituents]

    # -- Scraping -------------------------------------------------------------

    def refresh_from_source(self) -> str:
        """Scrape Wikipedia for current constituent list, filter, and write a
        dated manifest CSV.

        Returns the path to the newly written manifest.

        Raises ValueError if the Wikipedia page structure cannot be parsed.
        """
        today = datetime.date.today().isoformat()
        path = os.path.join(self.manifest_dir, f"{self.name}_{today}.csv")

        logger.info("Scraping %s for %s constituents...", self._WIKI_URL, self.name)
        try:
            import io
            import requests as _req
            headers = {"User-Agent": "sandbox-alpha/0.1 (research; contact@example.com)"}
            resp = _req.get(self._WIKI_URL, headers=headers, timeout=30)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
        except Exception as exc:
            raise ValueError(
                f"Failed to fetch or parse {self._WIKI_URL}: {exc}"
            ) from exc

        # Try to find the Components table — usually the first or second
        # table with a "Ticker" or "Symbol" column.
        df = None
        for t in tables:
            cols_lower = [str(c).strip().lower() for c in t.columns]
            if "ticker" in cols_lower or "symbol" in cols_lower:
                df = t
                break

        if df is None:
            # Fallback: try the largest table on the page.
            if not tables:
                raise ValueError(
                    f"No tables found on {self._WIKI_URL}. Wikipedia page structure "
                    "may have changed."
                )
            df = max(tables, key=lambda t: t.shape[0])
            logger.warning(
                "Could not identify a ticker/symbol column in Wikipedia tables; "
                "using largest table (shape=%s) as fallback. This may produce "
                "incorrect results.", df.shape
            )

        # Detect the ticker column and the name column
        df.columns = [str(c).strip() for c in df.columns]
        cols_lower = [str(c).lower() for c in df.columns]
        ticker_col = None
        name_col = None

        for i, c in enumerate(cols_lower):
            if c in ("ticker", "symbol"):
                ticker_col = df.columns[i]
            if c in ("company", "name", "security"):
                name_col = df.columns[i]

        if ticker_col is None:
            raise ValueError(
                f"Could not find a 'Ticker' or 'Symbol' column in the Wikipedia "
                f"table. Columns found: {list(df.columns)}"
            )

        # Clean & filter
        symbols_raw = df[ticker_col].dropna().astype(str).str.strip()
        # Remove any row that doesn't look like a valid US ticker
        symbols_clean = symbols_raw[symbols_raw.str.match(r"^[A-Z]{1,5}$")]
        if len(symbols_clean) == 0:
            # No strict matches — use all non-empty values as-is
            symbols_clean = symbols_raw[symbols_raw.str.len() > 0]

        names = (
            df[name_col].fillna("").astype(str).str.strip()
            if name_col is not None
            else pd.Series([""] * len(df), index=df.index)
        )

        # Filter to US-primary-exchange only.
        # Russell 1000 Wikipedia table does not typically include an exchange
        # column. We rely on the fact that the Russell 1000 Components table
        # lists US-primary listings only (ADRs are tracked separately).
        # If an exchange column exists, use it; otherwise trust the table.
        exchange_col = None
        for i, c in enumerate(cols_lower):
            if c in ("exchange", "primary exchange"):
                exchange_col = df.columns[i]
                break

        if exchange_col is not None:
            exchanges = df[exchange_col].fillna("").astype(str).str.strip()
            us_mask = exchanges.isin(self._US_PRIMARY_EXCHANGES)
            if us_mask.sum() > 0:
                symbols_clean = symbols_clean[us_mask]
                names = names.loc[symbols_clean.index]

        # Deduplicate by symbol (first occurrence wins)
        deduped = {}
        for i in range(len(symbols_clean)):
            sym = symbols_clean.iloc[i]
            if sym not in deduped:
                deduped[sym] = names.iloc[i] if i < len(names) else ""

        # Write manifest CSV
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "name", "valid_from", "valid_until"])
            for sym, nm in sorted(deduped.items()):
                writer.writerow([sym, nm, today, ""])

        logger.info("Wrote %d constituents to %s", len(deduped), path)
        return path


# ---------------------------------------------------------------------------
# Universe alias expansion (PR 4e)
# ---------------------------------------------------------------------------

def resolve_universe_alias(alias: str, as_of: str | None = None) -> list[str]:
    """Expand a universe alias into an explicit symbol list.

    Supported aliases:
      russell1000            — full R1K constituents (~1000 syms)
      russell1000_top500     — top 500 by index weight (fallback: alphabetical head)
      russell1000_top200
      russell1000_top100
      russell1000_top50

    Unknown alias raises ValueError.
    """
    provider = UniverseProvider("russell1000")

    if alias == "russell1000":
        return provider.get_symbols(as_of=as_of)

    if alias.startswith("russell1000_top"):
        try:
            n = int(alias.split("top")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid universe alias: {alias!r}")

        symbols = provider.get_symbols(as_of=as_of)

        # TODO: cap-weighted — when manifest CSV includes a weight column,
        # sort by it descending before slicing.  For now, use natural order
        # (alphabetical by Wikipedia scrape).
        return symbols[:n]

    raise ValueError(f"Unknown universe alias: {alias!r}")
