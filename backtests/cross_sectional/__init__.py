"""Cross-sectional portfolio engine.

Exposes:
  run_cross_sectional_backtest  — full pipeline: score→weight→rebalance→returns→costs→metrics
  cross_metrics helpers         — portfolio_sharpe, portfolio_ir, portfolio_turnover,
                                  portfolio_max_drawdown, portfolio_hit_rate
"""

from __future__ import annotations

# Dual-import pattern for container flat-layout compatibility
try:
    from .portfolio import PortfolioBuilder
    from .costs import apply_transaction_costs
    from .cross_metrics import (
        portfolio_sharpe,
        portfolio_ir,
        portfolio_turnover,
        portfolio_max_drawdown,
        portfolio_hit_rate,
        metrics_dict,
    )
    from .engine import run_cross_sectional_backtest
except ImportError:
    from portfolio import PortfolioBuilder  # type: ignore[no-redef]
    from costs import apply_transaction_costs  # type: ignore[no-redef]
    from cross_metrics import (  # type: ignore[no-redef]
        portfolio_sharpe,
        portfolio_ir,
        portfolio_turnover,
        portfolio_max_drawdown,
        portfolio_hit_rate,
        metrics_dict,
    )
    from engine import run_cross_sectional_backtest  # type: ignore[no-redef]


__all__ = [
    "run_cross_sectional_backtest",
    "PortfolioBuilder",
    "apply_transaction_costs",
    "portfolio_sharpe",
    "portfolio_ir",
    "portfolio_turnover",
    "portfolio_max_drawdown",
    "portfolio_hit_rate",
    "metrics_dict",
]
