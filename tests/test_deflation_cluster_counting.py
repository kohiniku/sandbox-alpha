"""
Regression test for issue #66: Deflation threshold's N_family conflates 
refine-loop iterations with independent exploration.

The problem: when the review loop refines parameters of a promising family 
(e.g., rsi_window 19→30→50), each refinement is counted as an independent 
trial, inflating N_family and raising the deflation bar unfairly.

The fix: count distinct parameter clusters instead of raw trial count.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_loop import (
    compute_effective_min_sharpe,
    evaluate_result,
    update_family_aggregates,
    STRATEGY_TEMPLATES,
)


def _make_refinement_entry(strategy, symbol, params, val_sharpe):
    """Create a synthetic evaluation record for a parameter refinement."""
    return {
        "hypothesis": {
            "strategy": strategy,
            "symbol": symbol,
            "params": params,
        },
        "evaluation": {
            "verdict": "rejected",
            "sharpe_ratio": val_sharpe,
            "gate_results": {
                "validation": True,
                "deflation": False,  # failed deflation
                "holdout": True,
            },
        },
        "tested_at": "2026-07-30T12:00:00Z",
    }


class TestDeflationClusterCounting:
    def test_refinement_loop_should_not_inflate_n_family(self):
        """
        When the review loop refines parameters within the same cluster 
        (e.g., fast_window 10→11→12, all within ±15%), 
        N_family should count this as ~1 effective trial, not 3.
        
        This test reproduces the issue from #66: parameters were refined 
        multiple times but the deflation bar kept rising because each 
        refinement was counted as an independent trial.
        """
        knowledge = {
            "tested_combinations": [
                # Three refinements of the same SMA strategy, all within ±15% cluster
                {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 10, "slow_window": 30}},
                {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 11, "slow_window": 32}},
                {"strategy": "sma_crossover", "symbol": "AAPL", "params": {"fast_window": 12, "slow_window": 34}},
            ],
            "adopted": [],
            "rejected": [],
            "families": {},
        }
        
        # Update family aggregates for each refinement
        for tc in knowledge["tested_combinations"]:
            record = _make_refinement_entry(
                tc["strategy"], tc["symbol"], tc["params"], val_sharpe=2.0
            )
            update_family_aggregates(knowledge, record)
        
        fam_key = "sma_crossover|AAPL"
        fam = knowledge["families"][fam_key]
        
        # Raw trial count is 3
        assert fam["n_trials"] == 3
        
        # But the effective N for deflation should be closer to 1 (one cluster)
        # because all three are refinements within the same parameter neighborhood
        
        # Compute what the deflation threshold would be with raw count (WRONG)
        T_val = 252
        threshold_raw = compute_effective_min_sharpe(fam["n_trials"], T_val)
        
        # Compute what it should be with cluster count (CORRECT)
        # For now, we'll check that the family has cluster information
        # The actual implementation will need to track clusters
        
        # The threshold with N=1 cluster should be much lower than with N=3 trials
        threshold_1_cluster = compute_effective_min_sharpe(1, T_val)
        
        # The raw threshold (N=3) is punishing the refinement
        # Expected: sqrt(2*ln(3)) * sqrt(252/252) ≈ 1.48
        # vs N=1: sqrt(2*ln(1)) * sqrt(252/252) = 0, floored to 0.5
        
        # This test will FAIL on main because the code doesn't track clusters yet
        # After the fix, we should be able to get the cluster count
        assert "param_clusters" in fam or "distinct_clusters" in fam, \
            "Family should track parameter clusters for deflation calculation"
        
        # The effective N for deflation should be the cluster count, not raw trials
        effective_n = fam.get("distinct_clusters", fam["n_trials"])
        assert effective_n < fam["n_trials"], \
            f"Effective N ({effective_n}) should be less than raw trials ({fam['n_trials']}) for refinements"
        
        threshold_effective = compute_effective_min_sharpe(effective_n, T_val)
        assert threshold_effective < threshold_raw, \
            f"Effective threshold ({threshold_effective:.3f}) should be lower than raw ({threshold_raw:.3f})"

    def test_distinct_parameter_regions_should_count_separately(self):
        """
        When trials explore genuinely different parameter regions 
        (not within ±15% or one-step), they should count as separate trials.
        """
        knowledge = {
            "tested_combinations": [
                # Three genuinely different parameter regions
                {"strategy": "sma", "symbol": "AAPL", "params": {"fast_window": 5, "slow_window": 20}},
                {"strategy": "sma", "symbol": "AAPL", "params": {"fast_window": 50, "slow_window": 200}},
                {"strategy": "sma", "symbol": "AAPL", "params": {"fast_window": 100, "slow_window": 400}},
            ],
            "adopted": [],
            "rejected": [],
            "families": {},
        }
        
        for tc in knowledge["tested_combinations"]:
            record = _make_refinement_entry(
                tc["strategy"], tc["symbol"], tc["params"], val_sharpe=1.0
            )
            update_family_aggregates(knowledge, record)
        
        fam_key = "sma|AAPL"
        fam = knowledge["families"][fam_key]
        
        assert fam["n_trials"] == 3
        
        # All three are in different clusters, so effective N should be 3
        effective_n = fam.get("distinct_clusters", fam["n_trials"])
        assert effective_n == 3, \
            f"Genuinely different parameter regions should count as 3 clusters, got {effective_n}"
