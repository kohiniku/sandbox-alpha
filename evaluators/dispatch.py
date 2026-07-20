"""Dispatch layer: iterate over manifest evaluator spec and call registered metrics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from evaluators.base import Evaluator


def evaluate(
    spec: Any,
    returns: pd.DataFrame,
    weights: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    config: Optional[dict] = None,
) -> Dict[str, float]:
    """Evaluate all metrics listed in *spec.metrics*.

    Parameters
    ----------
    spec : duck-typed object with ``.metrics`` (list[str]) attribute.
    returns : DataFrame of asset returns (columns = assets, index = dates).
    weights : optional DataFrame of portfolio weights, same shape/index.
    benchmark : optional Series of benchmark returns, same index.
    config : optional dict passed through to metric functions (e.g. factors).

    Returns
    -------
    dict[str, float] — one entry per requested metric.  factor_exposure
    expands into multiple keys (factor_exposure_alpha, factor_exposure_MKT, ...).
    """
    if config is None:
        config = {}

    results: Dict[str, float] = {}
    metrics = getattr(spec, "metrics", [])

    for metric_name in metrics:
        fn = Evaluator.get(metric_name)
        if fn is None:
            print(f"[evaluators] WARNING: unknown metric '{metric_name}', skipping")
            continue
        try:
            value = fn(returns, weights, benchmark, config)
        except (ValueError, TypeError) as exc:
            print(f"[evaluators] ERROR in '{metric_name}': {exc}")
            import numpy as np
            results[metric_name] = np.nan
            continue

        if isinstance(value, dict):
            # Expanding metric (e.g. factor_exposure)
            results.update(value)
        else:
            results[metric_name] = value

    return results
