"""Portfolio evaluator plugin framework (PR-D, Phase 0)."""

from evaluators.base import Evaluator
from evaluators.dispatch import evaluate

# Import to trigger registration of built-in metrics
import evaluators.portfolio  # noqa: F401

__all__ = ["Evaluator", "evaluate"]
