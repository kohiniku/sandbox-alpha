"""Cross-sectional strategy package.

Exposes:
  CROSS_SECTIONAL_STRATEGIES  — {name: callable} registry
  validate_weights            — contract: weight DataFrame validator
  validate_signals            — contract: signal DataFrame validator
  validate_scores             — contract: score DataFrame validator

All cross-sectional strategies are imported here so the registry is fully
populated at import time.
"""
from __future__ import annotations

# ── Populate the registry by importing strategies ────────────────────────
# Dual-import pattern for container flat-layout compatibility (tested by
# tests/test_container_flat_imports.py).
try:
    from .xs_momentum import CROSS_SECTIONAL_STRATEGIES as _xs_registry
except ImportError:
    from xs_momentum import CROSS_SECTIONAL_STRATEGIES as _xs_registry  # type: ignore[no-redef]

# Merge into the package-level registry
CROSS_SECTIONAL_STRATEGIES: dict = {}
CROSS_SECTIONAL_STRATEGIES.update(_xs_registry)

# ── Re-export contract validators ────────────────────────────────────────
try:
    from ._contract import validate_weights, validate_signals, validate_scores
except ImportError:
    from _contract import validate_weights, validate_signals, validate_scores  # type: ignore[no-redef]

__all__ = [
    "CROSS_SECTIONAL_STRATEGIES",
    "validate_weights",
    "validate_signals",
    "validate_scores",
]
