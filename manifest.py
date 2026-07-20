#!/usr/bin/env python3
"""
Strategy Manifest Schema for sandbox-alpha v2.

Pure Python dataclass hierarchy for declaring strategy manifests.
No third-party dependencies (stdlib only).

Phase 0 PR-A: Foundation schema for v2 redesign.
"""

import base64
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CODE_BYTES = 64 * 1024  # 64 KB decoded

SYMBOL_REGEX = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$")
DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

VALID_METRICS = frozenset({
    "sharpe", "ir", "turnover", "cvar_95", "max_drawdown_pct",
    "total_return_pct", "factor_exposure",
})

VALID_COMPUTE_MODES = frozenset({"inference", "training"})
VALID_EVALUATOR_TYPES = frozenset({"portfolio", "single_asset", "custom"})
VALID_EXECUTION_MODES = frozenset({"structured", "expert"})
VALID_NEWS_SOURCES = frozenset({"arxiv_investment", "general_arxiv"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ManifestValidationError(Exception):
    """Raised when manifest deserialization or validation fails."""
    pass


# ---------------------------------------------------------------------------
# DataSource discriminated union
# ---------------------------------------------------------------------------

class DataSource:
    """Base class for data sources. Discriminated by ``type`` field.

    Subclasses register via ``@DataSource.register("type_name")`` and
    implement ``_from_dict`` / ``to_dict``.
    """

    _registry: ClassVar[Dict[str, Any]] = {}

    def __init__(self, source_type: str = "") -> None:
        self.type = source_type

    # -- registration -------------------------------------------------------

    @classmethod
    def register(cls, source_type: str):
        """Decorator to register a DataSource subclass."""
        def decorator(subclass):
            cls._registry[source_type] = subclass
            return subclass
        return decorator

    # -- serialisation ------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Any) -> "DataSource":
        """Deserialize from dict, dispatching to registered subclass."""
        if not isinstance(d, dict):
            raise ManifestValidationError(
                f"data_source must be a dict, got {builtins_type_name(d)}"
            )
        source_type = d.get("type")
        if not source_type:
            raise ManifestValidationError("data_source missing 'type' field")
        subclass = cls._registry.get(source_type)
        if subclass is None:
            raise ManifestValidationError(
                f"unknown data_source.type '{source_type}'. "
                f"Known types: {sorted(cls._registry.keys())}"
            )
        return subclass._from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_dict()")

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "DataSource":
        raise NotImplementedError("Subclasses must implement _from_dict()")


def builtins_type_name(obj):
    """Return type name without shadowing built-in ``type``."""
    return obj.__class__.__name__


@DataSource.register("ohlcv")
class OhlcvSource(DataSource):
    """OHLCV data source for a universe of symbols."""

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        start: str = "",
        end: Optional[str] = None,
    ) -> None:
        super().__init__(source_type="ohlcv")
        self.universe: List[str] = universe if universe is not None else []
        self.start = start
        self.end = end

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "OhlcvSource":
        return cls(
            universe=d.get("universe", []),
            start=d.get("start", ""),
            end=d.get("end"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": self.type,
            "universe": self.universe,
            "start": self.start,
        }
        if self.end is not None:
            result["end"] = self.end
        return result


@DataSource.register("news_sentiment")
class NewsSentimentSource(DataSource):
    """News/sentiment alternative data source (Phase 2 PR-H).

    Reads from a pre-fetched arxiv paper corpus (no network at runtime).
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        start: str = "",
        end: Optional[str] = None,
        source: str = "arxiv_investment",
        min_relevance: float = 0.3,
    ) -> None:
        super().__init__(source_type="news_sentiment")
        self.universe: List[str] = universe if universe is not None else []
        self.start = start
        self.end = end
        self.source = source
        self.min_relevance = min_relevance

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "NewsSentimentSource":
        return cls(
            universe=d.get("universe", []),
            start=d.get("start", ""),
            end=d.get("end"),
            source=d.get("source", "arxiv_investment"),
            min_relevance=float(d.get("min_relevance", 0.3)),
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": self.type,
            "universe": self.universe,
            "start": self.start,
            "source": self.source,
            "min_relevance": self.min_relevance,
        }
        if self.end is not None:
            result["end"] = self.end
        return result


# ---------------------------------------------------------------------------
# Top 50 asset managers by AUM (CIK list for 'top_50' shortcut)
# ---------------------------------------------------------------------------
# Sourced from public 13F filings. Each entry is a CIK (10-digit string).
# These represent the 50 largest institutional investment managers by
# reported 13F assets under management as of Q4 2025.
# fmt: off
TOP_50_CIKS: tuple = (
    "0001067983",  # Berkshire Hathaway
    "0001341439",  # Vanguard Group
    "0000036405",  # BlackRock
    "0001056288",  # State Street
    "0000938528",  # Fidelity Management & Research
    "0001535520",  # Geode Capital Management
    "0001423053",  # Capital World Investors
    "0001166559",  # Price T Rowe Associates
    "0001350694",  # Nuveen Asset Management
    "0000727139",  # Invesco
    "0001802145",  # Morgan Stanley
    "0000038787",  # Goldman Sachs Group
    "0000769993",  # JP Morgan Chase
    "0000070858",  # Bank of America / Merrill Lynch
    "0000094716",  # Northern Trust
    "0001061165",  # Legal & General Group
    "0001037389",  # Charles Schwab Investment Management
    "0001085146",  # UBS Group
    "0001001085",  # Dimensional Fund Advisors
    "0001479543",  # D.E. Shaw
    "0001541617",  # Two Sigma Investments
    "0001035674",  # Renaissance Technologies
    "0001166928",  # AQR Capital Management
    "0001603466",  # Citadel Advisors
    "0001103804",  # Wellington Management Group
    "0001289419",  # Millennium Management
    "0001167483",  # Franklin Resources
    "0000315066",  # Ameriprise Financial
    "0000350797",  # Bank of New York Mellon
    "0000094011",  # Amundi
    "0001141391",  # AllianceBernstein
    "0001159157",  # Macquarie Group
    "0000049679",  # Principal Financial Group
    "0000051858",  # Raymond James Financial
    "0000023760",  # Royal Bank of Canada
    "0001273087",  # Sumitomo Mitsui Trust Holdings
    "0001364742",  # Blackstone Group
    "0000891541",  # Deutsche Bank
    "0000913383",  # PNC Financial Services
    "0001310018",  # Bridgewater Associates
    "0001345471",  # Elliott Investment Management
    "0001075857",  # Janus Henderson Group
    "0001368754",  # Man Group
    "0001172265",  # Point72 Asset Management
    "0000037321",  # Regions Financial
    "0001398348",  # Viking Global Investors
    "0000884776",  # AIG
    "0001167783",  # Baupost Group
    "0001555555",  # Tiger Global Management
    "0001593313",  # Marshall Wace
)
# fmt: on


@DataSource.register("sec_13f")
class Sec13FSource(DataSource):
    """SEC 13F institutional holdings data source (Phase 2 PR-I).

    Reads from a pre-fetched JSONL corpus on disk (no network at runtime)
    via offline ingest scripts/ingest_sec_13f.py.
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        start: str = "",
        end: Optional[str] = None,
        filers: Optional[List[str]] = None,
        min_position_pct: float = 0.5,
    ) -> None:
        super().__init__(source_type="sec_13f")
        self.universe: List[str] = universe if universe is not None else []
        self.start = start
        self.end = end
        self.filers: List[str] = filers if filers is not None else []
        self.min_position_pct = min_position_pct

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "Sec13FSource":
        return cls(
            universe=d.get("universe", []),
            start=d.get("start", ""),
            end=d.get("end"),
            filers=d.get("filers", []),
            min_position_pct=float(d.get("min_position_pct", 0.5)),
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": self.type,
            "universe": self.universe,
            "start": self.start,
            "filers": self.filers,
            "min_position_pct": self.min_position_pct,
        }
        if self.end is not None:
            result["end"] = self.end
        return result


# ---------------------------------------------------------------------------
# ModelArtifact (reserved for Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class ModelArtifact:
    """Model artifact reference (reserved for Phase 3)."""
    name: str = ""
    revision: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Any) -> "ModelArtifact":
        if not isinstance(d, dict):
            raise ManifestValidationError(
                f"model_artifact must be a dict, got {builtins_type_name(d)}"
            )
        return cls(name=d.get("name", ""), revision=d.get("revision"))

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"name": self.name}
        if self.revision is not None:
            result["revision"] = self.revision
        return result


# ---------------------------------------------------------------------------
# ComputeSpec
# ---------------------------------------------------------------------------

@dataclass
class ComputeSpec:
    """Compute resource specification."""
    mode: str = "inference"
    budget_seconds: int = 60
    gpu: bool = False

    @classmethod
    def from_dict(cls, d: Any) -> "ComputeSpec":
        if not isinstance(d, dict):
            raise ManifestValidationError(
                f"compute must be a dict, got {builtins_type_name(d)}"
            )
        return cls(
            mode=d.get("mode", "inference"),
            budget_seconds=d.get("budget_seconds", 60),
            gpu=d.get("gpu", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "budget_seconds": self.budget_seconds,
            "gpu": self.gpu,
        }


# ---------------------------------------------------------------------------
# EvaluatorSpec
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorSpec:
    """Evaluator specification with validated metrics."""
    evaluator_type: str = "portfolio"
    metrics: List[str] = field(default_factory=list)
    benchmark: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any) -> "EvaluatorSpec":
        if not isinstance(d, dict):
            raise ManifestValidationError(
                f"evaluator must be a dict, got {builtins_type_name(d)}"
            )
        return cls(
            evaluator_type=d.get("type", "portfolio"),
            metrics=d.get("metrics", []),
            benchmark=d.get("benchmark"),
            extras=d.get("extras", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": self.evaluator_type,
            "metrics": self.metrics,
        }
        if self.benchmark is not None:
            result["benchmark"] = self.benchmark
        if self.extras:
            result["extras"] = self.extras
        return result


# ---------------------------------------------------------------------------
# StrategyManifest
# ---------------------------------------------------------------------------

@dataclass
class StrategyManifest:
    """
    Top-level strategy manifest for v2.

    Fields:
      - name: strategy identifier
      - code_b64: base64-encoded Python source
      - data_sources: list of DataSource (discriminated union)
      - model_artifacts: list of ModelArtifact (reserved for Phase 3)
      - compute: ComputeSpec (mode, budget_seconds, gpu)
      - evaluator: EvaluatorSpec (type, metrics, benchmark, extras)
    """
    name: str = ""
    code_b64: str = ""
    data_sources: List[DataSource] = field(default_factory=list)
    model_artifacts: List[ModelArtifact] = field(default_factory=list)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    evaluator: EvaluatorSpec = field(default_factory=EvaluatorSpec)
    execution_mode: str = "structured"

    @classmethod
    def from_dict(cls, d: Any) -> "StrategyManifest":
        """Deserialize from dict."""
        if not isinstance(d, dict):
            raise ManifestValidationError(
                f"manifest must be a dict, got {builtins_type_name(d)}"
            )
        data_sources = [DataSource.from_dict(ds) for ds in d.get("data_sources", [])]
        model_artifacts = [ModelArtifact.from_dict(ma) for ma in d.get("model_artifacts", [])]
        compute = ComputeSpec.from_dict(d.get("compute", {}))
        evaluator = EvaluatorSpec.from_dict(d.get("evaluator", {}))
        return cls(
            name=d.get("name", ""),
            code_b64=d.get("code_b64", ""),
            data_sources=data_sources,
            model_artifacts=model_artifacts,
            compute=compute,
            evaluator=evaluator,
            execution_mode=d.get("execution_mode", "structured"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "code_b64": self.code_b64,
            "data_sources": [ds.to_dict() for ds in self.data_sources],
            "model_artifacts": [ma.to_dict() for ma in self.model_artifacts],
            "compute": self.compute.to_dict(),
            "evaluator": self.evaluator.to_dict(),
            "execution_mode": self.execution_mode,
        }

    def validate(self) -> List[str]:
        """
        Run ALL validation checks. Returns list of violation messages.
        Empty list means valid. Does NOT stop at first violation.
        """
        violations: List[str] = []

        # 1. name
        if not self.name or not isinstance(self.name, str):
            violations.append("name must be a non-empty string")

        # 2. code_b64
        if not self.code_b64 or not isinstance(self.code_b64, str):
            violations.append("code_b64 must be a non-empty string")
        else:
            try:
                decoded = base64.b64decode(self.code_b64)
                if len(decoded) > MAX_CODE_BYTES:
                    violations.append(
                        f"code_b64 decoded size {len(decoded)} bytes "
                        f"exceeds max {MAX_CODE_BYTES}"
                    )
            except Exception as e:
                violations.append(f"code_b64 is not valid base64: {e}")

        # 3. data_sources
        for i, ds in enumerate(self.data_sources):
            if isinstance(ds, OhlcvSource):
                violations.extend(_validate_ohlcv_source(ds, i))
            elif isinstance(ds, NewsSentimentSource):
                violations.extend(_validate_news_sentiment_source(ds, i))
            elif isinstance(ds, Sec13FSource):
                violations.extend(_validate_sec13f_source(ds, i))

        # 4. model_artifacts
        for i, ma in enumerate(self.model_artifacts):
            if not ma.name or not isinstance(ma.name, str):
                violations.append(
                    f"model_artifacts[{i}].name must be a non-empty string"
                )
            if ma.revision is not None and not isinstance(ma.revision, str):
                violations.append(
                    f"model_artifacts[{i}].revision must be a string or null"
                )

        # 5. compute
        if self.compute.mode not in VALID_COMPUTE_MODES:
            violations.append(
                f"compute.mode must be one of {sorted(VALID_COMPUTE_MODES)}, "
                f"got '{self.compute.mode}'"
            )
        if not isinstance(self.compute.budget_seconds, int) or isinstance(
            self.compute.budget_seconds, bool
        ):
            violations.append("compute.budget_seconds must be a non-negative integer")
        elif self.compute.budget_seconds < 0:
            violations.append("compute.budget_seconds must be a non-negative integer")
        if not isinstance(self.compute.gpu, bool):
            violations.append("compute.gpu must be a boolean")

        # 6. evaluator
        if self.evaluator.evaluator_type not in VALID_EVALUATOR_TYPES:
            violations.append(
                f"evaluator.type must be one of {sorted(VALID_EVALUATOR_TYPES)}, "
                f"got '{self.evaluator.evaluator_type}'"
            )
        if not isinstance(self.evaluator.metrics, list):
            violations.append("evaluator.metrics must be a list")
        else:
            for j, metric in enumerate(self.evaluator.metrics):
                if metric not in VALID_METRICS:
                    violations.append(
                        f"evaluator.metrics[{j}] '{metric}' is not valid. "
                        f"Allowed: {sorted(VALID_METRICS)}"
                    )
        if self.evaluator.benchmark is not None:
            if not isinstance(self.evaluator.benchmark, str):
                violations.append("evaluator.benchmark must be a string or null")
            elif not SYMBOL_REGEX.match(self.evaluator.benchmark):
                violations.append(
                    f"evaluator.benchmark '{self.evaluator.benchmark}' "
                    f"does not match symbol regex"
                )
        if not isinstance(self.evaluator.extras, dict):
            violations.append("evaluator.extras must be a dict")

        # 7. execution_mode
        if self.execution_mode not in VALID_EXECUTION_MODES:
            violations.append(
                f"execution_mode must be one of {sorted(VALID_EXECUTION_MODES)}, "
                f"got '{self.execution_mode}'"
            )

        return violations


# ---------------------------------------------------------------------------
# Helper validators
# ---------------------------------------------------------------------------

def _validate_ohlcv_source(source: OhlcvSource, index: int) -> List[str]:
    """Validate OhlcvSource fields."""
    violations: List[str] = []

    if not isinstance(source.universe, list) or len(source.universe) == 0:
        violations.append(
            f"data_sources[{index}].universe must be a non-empty list"
        )
    else:
        for j, symbol in enumerate(source.universe):
            if not isinstance(symbol, str):
                violations.append(
                    f"data_sources[{index}].universe[{j}] must be a string"
                )
            elif not SYMBOL_REGEX.match(symbol):
                violations.append(
                    f"data_sources[{index}].universe[{j}] '{symbol}' "
                    f"does not match symbol regex"
                )

    if not source.start or not isinstance(source.start, str):
        violations.append(
            f"data_sources[{index}].start must be a non-empty string"
        )
    elif not DATE_REGEX.match(source.start):
        violations.append(
            f"data_sources[{index}].start '{source.start}' "
            f"must be YYYY-MM-DD format"
        )

    if source.end is not None:
        if not isinstance(source.end, str):
            violations.append(
                f"data_sources[{index}].end must be a string or null"
            )
        elif not DATE_REGEX.match(source.end):
            violations.append(
                f"data_sources[{index}].end '{source.end}' "
                f"must be YYYY-MM-DD format"
            )

    return violations


def _validate_news_sentiment_source(source: "NewsSentimentSource", index: int) -> List[str]:
    """Validate NewsSentimentSource fields."""
    violations: List[str] = []

    # universe: optional list, but each entry must match symbol regex
    if not isinstance(source.universe, list):
        violations.append(
            f"data_sources[{index}].universe must be a list"
        )
    else:
        for j, symbol in enumerate(source.universe):
            if not isinstance(symbol, str):
                violations.append(
                    f"data_sources[{index}].universe[{j}] must be a string"
                )
            elif not SYMBOL_REGEX.match(symbol):
                violations.append(
                    f"data_sources[{index}].universe[{j}] '{symbol}' "
                    f"does not match symbol regex"
                )

    # start: required, YYYY-MM-DD
    if not source.start or not isinstance(source.start, str):
        violations.append(
            f"data_sources[{index}].start must be a non-empty string"
        )
    elif not DATE_REGEX.match(source.start):
        violations.append(
            f"data_sources[{index}].start '{source.start}' "
            f"must be YYYY-MM-DD format"
        )

    # end: optional, YYYY-MM-DD
    if source.end is not None:
        if not isinstance(source.end, str):
            violations.append(
                f"data_sources[{index}].end must be a string or null"
            )
        elif not DATE_REGEX.match(source.end):
            violations.append(
                f"data_sources[{index}].end '{source.end}' "
                f"must be YYYY-MM-DD format"
            )

    # source: must be in valid set
    if source.source not in VALID_NEWS_SOURCES:
        violations.append(
            f"data_sources[{index}].source must be one of {sorted(VALID_NEWS_SOURCES)}, "
            f"got '{source.source}'"
        )

    # min_relevance: float in [0.0, 1.0]
    if not isinstance(source.min_relevance, (int, float)):
        violations.append(
            f"data_sources[{index}].min_relevance must be a float, "
            f"got {type(source.min_relevance).__name__}"
        )
    else:
        mr = float(source.min_relevance)
        if mr < 0.0 or mr > 1.0:
            violations.append(
                f"data_sources[{index}].min_relevance must be in [0.0, 1.0], "
                f"got {mr}"
            )

    return violations


def _validate_sec13f_source(source: "Sec13FSource", index: int) -> List[str]:
    """Validate Sec13FSource fields."""
    violations: List[str] = []

    # universe: optional list, each entry must match symbol regex
    if not isinstance(source.universe, list):
        violations.append(
            f"data_sources[{index}].universe must be a list"
        )
    else:
        for j, symbol in enumerate(source.universe):
            if not isinstance(symbol, str):
                violations.append(
                    f"data_sources[{index}].universe[{j}] must be a string"
                )
            elif not SYMBOL_REGEX.match(symbol):
                violations.append(
                    f"data_sources[{index}].universe[{j}] '{symbol}' "
                    f"does not match symbol regex"
                )

    # start: required, YYYY-MM-DD
    if not source.start or not isinstance(source.start, str):
        violations.append(
            f"data_sources[{index}].start must be a non-empty string"
        )
    elif not DATE_REGEX.match(source.start):
        violations.append(
            f"data_sources[{index}].start '{source.start}' "
            f"must be YYYY-MM-DD format"
        )

    # end: optional, YYYY-MM-DD
    if source.end is not None:
        if not isinstance(source.end, str):
            violations.append(
                f"data_sources[{index}].end must be a string or null"
            )
        elif not DATE_REGEX.match(source.end):
            violations.append(
                f"data_sources[{index}].end '{source.end}' "
                f"must be YYYY-MM-DD format"
            )

    # filers: list[str] or list containing 'top_50'
    if not isinstance(source.filers, list):
        violations.append(
            f"data_sources[{index}].filers must be a list"
        )
    elif len(source.filers) == 0:
        violations.append(
            f"data_sources[{index}].filers must be a non-empty list. "
            f"Use ['top_50'] for the 50 largest asset managers."
        )
    else:
        for j, filer in enumerate(source.filers):
            if not isinstance(filer, str):
                violations.append(
                    f"data_sources[{index}].filers[{j}] must be a string"
                )
            elif filer == "top_50":
                # Allow the shortcut — validate no other entries when used
                if len(source.filers) > 1:
                    violations.append(
                        f"data_sources[{index}].filers: 'top_50' shortcut "
                        f"cannot be combined with other CIKs"
                    )
                break
            elif not filer.isdigit() or len(filer) != 10:
                violations.append(
                    f"data_sources[{index}].filers[{j}] '{filer}' "
                    f"must be a 10-digit CIK or 'top_50'"
                )

    # min_position_pct: float >= 0.0
    if not isinstance(source.min_position_pct, (int, float)):
        violations.append(
            f"data_sources[{index}].min_position_pct must be a float, "
            f"got {type(source.min_position_pct).__name__}"
        )
    else:
        mp = float(source.min_position_pct)
        if mp < 0.0:
            violations.append(
                f"data_sources[{index}].min_position_pct must be >= 0.0, "
                f"got {mp}"
            )

    return violations
