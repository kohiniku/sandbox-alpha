"""
Tests for overfitting guards — no network dependencies, synthetic data only.
Covers: 3-way split, deflation formula, holdout gate, cluster matching, replace-if-better.
"""
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.backtest_engine import split_walkforward
from autonomous_loop import (
    compute_effective_min_sharpe,
    _params_within_cluster,
    evaluate_result,
    MIN_SHARPE_BASE,
    MAX_DRAWDOWN_LIMIT,
    STRATEGY_TEMPLATES,
)


# ============================================================
# 3-way split boundaries
# ============================================================

class TestThreeWaySplit:
    def test_num_days_sum_correct(self):
        """Sum of train + val + holdout equals total length."""
        df = pd.DataFrame({"Close": range(252)}, index=pd.date_range("2020-01-02", periods=252, freq="B"))
        train, val, holdout = split_walkforward(df)
        assert len(train) + len(val) + len(holdout) == 252

    def test_chronological_order(self):
        """Train < val < holdout in time."""
        df = pd.DataFrame({"Close": range(252)}, index=pd.date_range("2020-01-02", periods=252, freq="B"))
        train, val, holdout = split_walkforward(df)
        assert train.index.max() < val.index.min()
        assert val.index.max() < holdout.index.min()

    def test_exact_ratios_100(self):
        """60/20/20 on 100 rows."""
        df = pd.DataFrame({"Close": range(100)}, index=range(100))
        train, val, holdout = split_walkforward(df)
        assert len(train) == 60
        assert len(val) == 20
        assert len(holdout) == 20


# ============================================================
# Deflation formula
# ============================================================

class TestDeflationFormula:
    def test_N2_T252_returns_sqrt_2ln2(self):
        """N_family=2, T_val=252 → threshold = max(0.5, sqrt(2*ln2)*sqrt(1)) = sqrt(2*ln2) ≈ 1.177"""
        result = compute_effective_min_sharpe(N_family=2, T_val=252)
        expected = math.sqrt(2 * math.log(2))
        assert result == pytest.approx(expected, rel=1e-6)
        assert result > MIN_SHARPE_BASE  # 1.177 > 0.5

    def test_N1_clamped_to_2(self):
        """N_family=1 → clamped to 2, same as N=2."""
        result = compute_effective_min_sharpe(N_family=1, T_val=252)
        assert result == pytest.approx(math.sqrt(2 * math.log(2)), rel=1e-6)

    def test_large_N_increases_threshold(self):
        """More trials → higher deflated threshold."""
        t2 = compute_effective_min_sharpe(N_family=2, T_val=252)
        t100 = compute_effective_min_sharpe(N_family=100, T_val=252)
        assert t100 > t2

    def test_small_T_increases_threshold(self):
        """Fewer trading days → higher threshold."""
        t252 = compute_effective_min_sharpe(N_family=2, T_val=252)
        t63 = compute_effective_min_sharpe(N_family=2, T_val=63)
        assert t63 > t252

    def test_base_floor_kicks_in(self):
        """When N_family=2 and T_val is large enough, floor of 0.5 may dominate."""
        result = compute_effective_min_sharpe(N_family=2, T_val=10000)
        # sqrt(2*ln2)*sqrt(252/10000) ≈ 1.177 * 0.159 ≈ 0.187, floored to 0.5
        assert result == 0.5


# ============================================================
# Holdout gate
# ============================================================

def _make_synthetic_result(val_sharpe, val_return, val_max_dd, val_days,
                           holdout_sharpe, holdout_return, holdout_days,
                           strategy="sma_crossover", symbol="AAPL",
                           params=None):
    """Build a synthetic backtest result dict matching the new engine output."""
    if params is None:
        params = {"fast_window": 10, "slow_window": 30}
    return {
        "strategy": strategy,
        "symbol": symbol,
        "params": params,
        "walkforward": {"enabled": True, "train_ratio": 0.6, "val_ratio": 0.2, "holdout_ratio": 0.2},
        "in_sample": {"sharpe_ratio": 1.0, "total_return_pct": 20.0, "max_drawdown_pct": -10.0,
                      "num_days": 378, "num_trades": 10},
        "out_of_sample": {"sharpe_ratio": val_sharpe, "total_return_pct": val_return,
                          "max_drawdown_pct": val_max_dd, "num_days": val_days, "num_trades": 5},
        "holdout": {"sharpe_ratio": holdout_sharpe, "total_return_pct": holdout_return,
                    "max_drawdown_pct": -5.0, "num_days": holdout_days, "num_trades": 3},
    }


class TestHoldoutGate:
    def test_negative_holdout_rejected_despite_good_val(self):
        """Candidate that passes validation but has negative holdout Sharpe should be rejected."""
        result = _make_synthetic_result(
            val_sharpe=1.5, val_return=25.0, val_max_dd=-10.0, val_days=252,
            holdout_sharpe=-0.3, holdout_return=-5.0, holdout_days=252,
        )
        hyp = {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 10, "slow_window": 30}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        assert verdict == "rejected"
        assert evaluation["gate_results"]["validation"] is True
        assert evaluation["gate_results"]["holdout"] is False

    def test_positive_holdout_negative_return_rejected(self):
        """Holdout Sharpe > 0 but holdout return <= 0 → rejected."""
        result = _make_synthetic_result(
            val_sharpe=1.5, val_return=25.0, val_max_dd=-10.0, val_days=252,
            holdout_sharpe=0.2, holdout_return=-1.0, holdout_days=252,
        )
        hyp = {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 10, "slow_window": 30}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        assert verdict == "rejected"
        assert evaluation["gate_results"]["holdout"] is False

    def test_all_good_holdout_passes(self):
        """Positive val + positive holdout → adopted."""
        result = _make_synthetic_result(
            val_sharpe=1.5, val_return=25.0, val_max_dd=-10.0, val_days=252,
            holdout_sharpe=0.8, holdout_return=15.0, holdout_days=252,
        )
        hyp = {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 10, "slow_window": 30}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        assert verdict == "adopted"
        assert evaluation["gate_results"]["holdout"] is True
        assert "cluster_id" in evaluation


# ============================================================
# Cluster matching
# ============================================================

class TestClusterMatching:
    def test_params_within_15pct_same_cluster(self):
        """Params within 15% range → same cluster."""
        p1 = {"fast_window": 10, "slow_window": 30}
        p2 = {"fast_window": 11, "slow_window": 32}  # within 15%
        result = _params_within_cluster(p1, p2, STRATEGY_TEMPLATES)
        assert result is True

    def test_params_outside_15pct_different_cluster(self):
        """Params far apart → different cluster."""
        p1 = {"fast_window": 10, "slow_window": 30}
        p2 = {"fast_window": 20, "slow_window": 60}  # 100% diff
        result = _params_within_cluster(p1, p2, STRATEGY_TEMPLATES)
        assert result is False

    def test_exact_same_params_same_cluster(self):
        """Identical params → same cluster."""
        p1 = {"window": 20, "threshold": 2.0}
        p2 = {"window": 20, "threshold": 2.0}
        result = _params_within_cluster(p1, p2, STRATEGY_TEMPLATES)
        assert result is True

    def test_extra_param_not_matching(self):
        """Differing keys → different cluster."""
        p1 = {"fast_window": 10}
        p2 = {"fast_window": 10, "slow_window": 30}
        result = _params_within_cluster(p1, p2, STRATEGY_TEMPLATES)
        assert result is False

    def test_list_params_one_step_same_cluster(self):
        """List params one step apart → same cluster."""
        p1 = {"rsi_window": 14, "oversold": 30, "overbought": 70}
        p2 = {"rsi_window": 14, "oversold": 25, "overbought": 70}  # one step in list
        result = _params_within_cluster(p1, p2, STRATEGY_TEMPLATES)
        assert result is True

    def test_list_params_two_steps_different_cluster(self):
        """List params two steps apart → different cluster."""
        p1 = {"rsi_window": 14, "oversold": 30, "overbought": 70}
        p2 = {"rsi_window": 14, "oversold": 20, "overbought": 70}  # two steps
        result = _params_within_cluster(p1, p2, STRATEGY_TEMPLATES)
        assert result is False


# ============================================================
# Replace-if-better logic
# ============================================================

class TestReplaceIfBetter:
    def test_duplicate_cluster_rejected_when_worse(self):
        """New candidate in existing cluster but with worse holdout Sharpe → rejected as duplicate_cluster."""
        result_new = _make_synthetic_result(
            val_sharpe=1.5, val_return=25.0, val_max_dd=-10.0, val_days=252,
            holdout_sharpe=0.5, holdout_return=10.0, holdout_days=252,
        )
        hyp_new = {"strategy": "sma_crossover", "symbol": "AAPL",
                   "params": {"fast_window": 11, "slow_window": 32}}

        # Existing adopted entry with same cluster (within 15%) and higher holdout Sharpe
        incumbent = {
            "hypothesis": {"strategy": "sma_crossover", "symbol": "AAPL",
                          "params": {"fast_window": 10, "slow_window": 30}},
            "evaluation": {"holdout_sharpe": 1.2, "total_return_pct": 30.0,
                          "sharpe_ratio": 2.0, "max_drawdown_pct": -5.0},
            "verdict": "adopted",
            "cluster_id": "abc12345",
        }
        knowledge = {
            "tested_combinations": [],
            "adopted": [incumbent],
            "rejected": [],
            "tested": [],
        }
        verdict, evaluation = evaluate_result(hyp_new, result_new, knowledge)
        assert verdict == "rejected"
        assert evaluation["gate_results"]["cluster"] == "duplicate_cluster"

    def test_cluster_replace_when_better_holdout(self):
        """New candidate in existing cluster with higher holdout Sharpe → replaces incumbent."""
        result_new = _make_synthetic_result(
            val_sharpe=1.5, val_return=25.0, val_max_dd=-10.0, val_days=252,
            holdout_sharpe=2.0, holdout_return=40.0, holdout_days=252,
        )
        hyp_new = {"strategy": "sma_crossover", "symbol": "AAPL",
                   "params": {"fast_window": 11, "slow_window": 32}}

        incumbent = {
            "hypothesis": {"strategy": "sma_crossover", "symbol": "AAPL",
                          "params": {"fast_window": 10, "slow_window": 30}},
            "evaluation": {"holdout_sharpe": 1.2, "total_return_pct": 30.0,
                          "sharpe_ratio": 2.0, "max_drawdown_pct": -5.0},
            "verdict": "adopted",
            "cluster_id": "abc12345",
        }
        knowledge = {
            "tested_combinations": [],
            "adopted": [incumbent],
            "rejected": [],
            "tested": [],
        }
        verdict, evaluation = evaluate_result(hyp_new, result_new, knowledge)
        assert verdict == "adopted"
        assert evaluation["gate_results"]["cluster"] == "replaced"
        # Incumbent moved to superseded
        assert len(knowledge.get("superseded", [])) == 1
        # Old adopted removed
        assert len(knowledge["adopted"]) == 0

    def test_new_cluster_when_params_different(self):
        """Params outside ±15% → new cluster, adopted normally."""
        result_new = _make_synthetic_result(
            val_sharpe=1.5, val_return=25.0, val_max_dd=-10.0, val_days=252,
            holdout_sharpe=1.0, holdout_return=15.0, holdout_days=252,
        )
        hyp_new = {"strategy": "sma_crossover", "symbol": "AAPL",
                   "params": {"fast_window": 25, "slow_window": 90}}  # very different

        incumbent = {
            "hypothesis": {"strategy": "sma_crossover", "symbol": "AAPL",
                          "params": {"fast_window": 5, "slow_window": 20}},
            "evaluation": {"holdout_sharpe": 1.2, "total_return_pct": 30.0,
                          "sharpe_ratio": 2.0, "max_drawdown_pct": -5.0},
            "verdict": "adopted",
            "cluster_id": "old_cluster",
        }
        knowledge = {
            "tested_combinations": [],
            "adopted": [incumbent],
            "rejected": [],
            "tested": [],
        }
        verdict, evaluation = evaluate_result(hyp_new, result_new, knowledge)
        assert verdict == "adopted"
        assert evaluation["gate_results"]["cluster"] == "new"
        # Incumbent preserved, new one adopted too
        assert len(knowledge["adopted"]) == 1


# ============================================================
# error handling
# ============================================================

class TestErrorHandling:
    def test_error_result_rejected(self):
        result = {"error": "something went wrong"}
        hyp = {"strategy": "sma_crossover", "symbol": "AAPL", "params": {}}
        knowledge = {"tested_combinations": [], "adopted": [], "rejected": [], "tested": []}
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        assert verdict == "error"
        assert "error" in evaluation