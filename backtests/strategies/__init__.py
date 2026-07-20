"""Built-in strategy registry.

Each strategy module exposes:
  NAME                       registry key
  compute_signal(df, **p)    strategy-specific position logic only;
                             returns (df, position_col)

The trusted return pipeline (_pipeline.attach_returns) is applied here, so a
strategy module cannot diverge from the shared return/cost conventions.

Import contexts (same constraint as backtest_engine's metrics import):
  - pytest imports this as backtests.strategies
  - the container runs backtest_engine.py as a script, importing it as
    top-level "strategies" (/backtest/strategies after COPY backtests/)
Relative imports inside the package work in both contexts.

Note: the repo-root strategies/ directory (runtime dump of LLM-generated
strategy code, gitignored) is unrelated to this package and never on
sys.path for the engine.
"""

from . import mean_reversion, momentum, rsi, sma_crossover
from ._pipeline import attach_returns

_MODULES = (sma_crossover, mean_reversion, momentum, rsi)


def _make_runner(module):
    def run(df, **params):
        df, position_col = module.compute_signal(df, **params)
        return attach_returns(df, position_col)

    run.__name__ = f"run_{module.NAME}_strategy"
    run.__doc__ = module.compute_signal.__doc__
    return run


STRATEGIES = {module.NAME: _make_runner(module) for module in _MODULES}

# Stable callables for direct use (tests, notebooks); the engine re-exports these.
run_sma_crossover_strategy = STRATEGIES["sma_crossover"]
run_mean_reversion_strategy = STRATEGIES["mean_reversion"]
run_momentum_strategy = STRATEGIES["momentum"]
run_rsi_strategy = STRATEGIES["rsi"]
