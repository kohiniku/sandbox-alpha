"""Single-name built-in strategy modules.

Each submodule exposes NAME + compute_signal(df, **params) -> (df, position_col).
The parent package (backtests.strategies) wraps these with attach_returns to
build STRATEGIES.
"""
# Dual-import for container flat-layout compatibility (see
# tests/test_container_flat_imports.py). In the Dockerfile the whole
# strategies/ tree is copied preserving structure, so relative imports
# succeed both in pytest (backtests.strategies._single_name...) and in
# script mode inside the container (strategies._single_name...).
from . import sma_crossover, mean_reversion, momentum, rsi

__all__ = ["sma_crossover", "mean_reversion", "momentum", "rsi"]
