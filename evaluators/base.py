"""Base evaluator class with plugin registry."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd


# Type alias for metric functions
MetricFn = Callable[
    [pd.DataFrame, Optional[pd.DataFrame], Optional[pd.Series], dict],
    Any,  # float or dict[str, float] for expanding metrics
]


class Evaluator:
    """Base class / registry for portfolio metrics.

    Usage::

        @Evaluator.register("my_metric")
        def my_metric(returns, weights, benchmark, config):
            ...
            return 1.23

    The registry is a simple class-level dict so new metrics can be added
    without modifying this file.
    """

    _registry: Dict[str, MetricFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[MetricFn], MetricFn]:
        """Decorator that registers *fn* under *name*."""

        def _decorator(fn: MetricFn) -> MetricFn:
            cls._registry[name] = fn
            return fn

        return _decorator

    @classmethod
    def get(cls, name: str) -> Optional[MetricFn]:
        return cls._registry.get(name)

    @classmethod
    def registered_names(cls) -> list[str]:
        return sorted(cls._registry.keys())
