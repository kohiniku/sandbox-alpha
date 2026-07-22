"""Cross-sectional backtest engine.

Orchestrates the full pipeline:
  score/signal/weight → weight construction → rebalance → returns → costs → metrics → segments
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Dual-import for container flat-layout compatibility
try:
    from .portfolio import PortfolioBuilder
    from .costs import apply_transaction_costs
    from .cross_metrics import metrics_dict, portfolio_sharpe
except ImportError:
    from portfolio import PortfolioBuilder  # type: ignore[no-redef]
    from costs import apply_transaction_costs  # type: ignore[no-redef]
    from cross_metrics import metrics_dict, portfolio_sharpe  # type: ignore[no-redef]


def _build_benchmark_returns(panel: dict[str, pd.DataFrame], universe: list[str]) -> pd.Series:
    """Build equal-weight universe benchmark return series."""
    closes = {}
    for sym in universe:
        if sym in panel and "Close" in panel[sym].columns:
            closes[sym] = panel[sym]["Close"]
    if not closes:
        return pd.Series(dtype=float)
    close_df = pd.DataFrame(closes)
    # Align to common index
    close_df = close_df.sort_index()
    rets = close_df.pct_change()
    return rets.mean(axis=1)


def run_cross_sectional_backtest(
    raw_output: pd.DataFrame,
    return_type: str,
    universe: list[str],
    panel: dict[str, pd.DataFrame],
    config: dict,
) -> dict:
    """Run full cross-sectional backtest pipeline.

    Parameters
    ----------
    raw_output : DataFrame — output of user's generate_cross_signal
    return_type : str — "weights" | "signals" | "scores"
    universe : list[str] — symbols
    panel : dict[symbol → DataFrame] — OHLCV data
    config : dict — extras + train_end/val_end, construction_mode, etc.

    Returns
    -------
    dict with keys: in_sample, out_of_sample, holdout, walkforward,
    cross_sectional.  Each segment dict has metrics from metrics_dict().
    """
    # ── 1. Convert raw_output → weights ─────────────────────────────────
    weights: pd.DataFrame

    if return_type == "weights":
        if config.get("construction_mode") == "custom_weights":
            weights = raw_output.copy()
        else:
            # Direct weights from strategy — use as-is
            weights = raw_output.copy()
    elif return_type == "signals":
        weights = PortfolioBuilder.signals_to_equal_weights(raw_output)
    elif return_type == "scores":
        mode = config.get("construction_mode", "top_k")
        if mode == "top_k":
            k = config.get("top_k", 50)
            long_only = config.get("long_only", True)
            weights = PortfolioBuilder.top_k_weights(raw_output, k=k, long_only=long_only)
        elif mode == "quintile_ls":
            q = config.get("quintiles", 5)
            weights = PortfolioBuilder.quintile_ls_weights(raw_output, quintiles=q)
        elif mode == "zscore_continuous":
            thresh = config.get("zscore_threshold", 0.0)
            long_only = config.get("long_only", True)
            weights = PortfolioBuilder.zscore_continuous_weights(
                raw_output, threshold=thresh, long_only=long_only
            )
        else:
            # Fallback to top_k
            k = config.get("top_k", 50)
            long_only = config.get("long_only", True)
            weights = PortfolioBuilder.top_k_weights(raw_output, k=k, long_only=long_only)
    else:
        raise ValueError(
            f"Unknown return_type '{return_type}'. Must be weights, signals, or scores."
        )

    # ── 2. Apply rebalance calendar ─────────────────────────────────────
    rebalance = config.get("rebalance", "monthly")
    weights = PortfolioBuilder.apply_rebalance_calendar(weights, cadence=rebalance)

    # ── 3. Compute asset returns from panel ─────────────────────────────
    closes_for_symbols = {}
    for sym in universe:
        if sym in panel and "Close" in panel[sym].columns:
            closes_for_symbols[sym] = panel[sym]["Close"]
    if not closes_for_symbols:
        return _empty_result("No Close data available in panel", config)

    asset_returns = pd.DataFrame(closes_for_symbols).pct_change()

    # Align weights to asset_returns
    common_cols = sorted(set(weights.columns) & set(asset_returns.columns))
    missing_cols = sorted(set(weights.columns) - set(asset_returns.columns))
    weights = weights[common_cols]
    if not missing_cols:
        # If we have extra symbols in weights not in asset_returns,
        # zero them out (already handled by slicing)
        pass

    asset_returns = asset_returns[common_cols]

    # Fill weights forward to cover asset_returns dates; fill NaN with 0
    full_idx = asset_returns.index
    weights = weights.reindex(full_idx).ffill().fillna(0.0)

    # ── 4. Portfolio gross returns ──────────────────────────────────────
    # Portfolio return at t uses weights decided at t-1
    portfolio_ret = (weights.shift(1) * asset_returns).sum(axis=1).iloc[1:]
    if len(portfolio_ret) < 2:
        return _empty_result("Portfolio return series has fewer than 2 rows", config)

    # ── 5. Costs ────────────────────────────────────────────────────────
    cost_bps = config.get("cost_bps", 5.0)
    if isinstance(cost_bps, dict):
        costs = apply_transaction_costs(weights, cost_bps_map=cost_bps, default_bps=5.0)
    else:
        costs = apply_transaction_costs(weights, cost_bps_map=float(cost_bps))

    costs = costs.reindex(portfolio_ret.index, fill_value=0.0)

    # ── 6. Portfolio net returns ────────────────────────────────────────
    net_returns = portfolio_ret - costs

    # ── 7. Slice by train/val/holdout ───────────────────────────────────
    train_end = config.get("train_end")
    val_end = config.get("val_end")

    if train_end is None or val_end is None:
        return _empty_result("Missing train_end or val_end in config", config)

    # Ensure timestamps
    if isinstance(train_end, str):
        train_end = pd.Timestamp(train_end)
    if isinstance(val_end, str):
        val_end = pd.Timestamp(val_end)

    # Build equal-weight benchmark
    bench = _build_benchmark_returns(panel, universe)

    # Slice
    in_sample_ret = net_returns[net_returns.index <= train_end]
    out_of_sample_ret = net_returns[
        (net_returns.index > train_end) & (net_returns.index <= val_end)
    ]
    holdout_ret = net_returns[net_returns.index > val_end]

    in_sample_w = weights.reindex(in_sample_ret.index)
    out_of_sample_w = weights.reindex(out_of_sample_ret.index)
    holdout_w = weights.reindex(holdout_ret.index)

    bench_is = bench.reindex(in_sample_ret.index) if len(bench) > 0 else None
    bench_oos = bench.reindex(out_of_sample_ret.index) if len(bench) > 0 else None
    bench_ho = bench.reindex(holdout_ret.index) if len(bench) > 0 else None

    # ── 8. Compute metrics per segment ──────────────────────────────────
    is_metrics = metrics_dict(in_sample_ret, bench_is, in_sample_w, len(in_sample_ret))
    oos_metrics = metrics_dict(out_of_sample_ret, bench_oos, out_of_sample_w, len(out_of_sample_ret))
    ho_metrics = metrics_dict(holdout_ret, bench_ho, holdout_w, len(holdout_ret))

    # Add convenience percentage fields for downstream compatibility
    _add_pct_fields(is_metrics, in_sample_ret, "in_sample")
    _add_pct_fields(oos_metrics, out_of_sample_ret, "out_of_sample")
    _add_pct_fields(ho_metrics, holdout_ret, "holdout")

    # ── 9. Cross-sectional summary ──────────────────────────────────────
    n_active_avg = float((weights.abs() > 1e-10).sum(axis=1).mean())

    result = {
        "in_sample": is_metrics,
        "out_of_sample": oos_metrics,
        "holdout": ho_metrics,
        "walkforward": {
            "enabled": True,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "holdout_ratio": 0.2,
        },
        "cross_sectional": {
            "construction_mode": config.get("construction_mode", "top_k"),
            "rebalance": rebalance,
            "cost_bps": float(cost_bps) if isinstance(cost_bps, (int, float)) else str(cost_bps),
            "n_active_symbols_avg": round(n_active_avg, 1),
        },
    }

    return result


def _add_pct_fields(m: dict, rets: pd.Series, prefix: str) -> None:
    """Add max_drawdown_pct and total_return_pct for downstream compatibility."""
    if prefix + "_max_drawdown_pct" not in m:
        m[prefix + "_max_drawdown_pct"] = round(m["max_drawdown"] * 100, 2)
    if prefix + "_total_return_pct" not in m:
        m[prefix + "_total_return_pct"] = round(
            float((1 + rets).prod() - 1) * 100, 2
        )


def _empty_result(reason: str, config: dict) -> dict:
    """Return a minimal but valid result dict for edge cases."""
    return {
        "in_sample": {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                      "num_days": 0, "num_trades": 0, "ir": 0.0, "turnover": 0.0,
                      "hit_rate": 0.0},
        "out_of_sample": {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                          "num_days": 0, "num_trades": 0, "ir": 0.0, "turnover": 0.0,
                          "hit_rate": 0.0},
        "holdout": {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                    "num_days": 0, "num_trades": 0, "ir": 0.0, "turnover": 0.0,
                    "hit_rate": 0.0},
        "walkforward": {"enabled": True, "train_ratio": 0.6, "val_ratio": 0.2,
                        "holdout_ratio": 0.2},
        "cross_sectional": {
            "construction_mode": config.get("construction_mode", "top_k"),
            "rebalance": config.get("rebalance", "monthly"),
            "cost_bps": config.get("cost_bps", 5.0),
            "n_active_symbols_avg": 0.0,
        },
    }
