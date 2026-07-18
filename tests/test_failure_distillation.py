"""
Tests for failure-knowledge distillation — no network dependencies.
Covers: family aggregate updates, migration rebuild, exhausted-cluster pre-block,
        prompt family lines / EXHAUSTED markers / gate-reason strings.
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_loop import (
    _check_exhausted_cluster,
    update_family_aggregates,
    _rebuild_families_from_history,
    _family_key,
    STRATEGY_TEMPLATES,
)
from llm_hypothesis import _build_prompt, _build_knowledge_summary


# ============================================================
# Helpers
# ============================================================

def _make_rejected_entry(strategy, symbol, params, val_sharpe=-0.5,
                         gate_validation=False, gate_holdout=True,
                         cluster_gate="", tested_at="2026-07-18T00:00:00"):
    """Build a synthetic rejected entry for testing."""
    ev = {
        "verdict": "rejected",
        "sharpe_ratio": val_sharpe,
        "total_return_pct": -5.0,
        "max_drawdown_pct": -30.0,
        "gate_results": {
            "validation": gate_validation,
            "holdout": gate_holdout,
        },
    }
    if cluster_gate:
        ev["gate_results"]["cluster"] = cluster_gate
    return {
        "hypothesis": {
            "strategy": strategy,
            "symbol": symbol,
            "params": params,
        },
        "evaluation": ev,
        "verdict": "rejected",
        "tested_at": tested_at,
    }


def _make_adopted_entry(strategy, symbol, params, val_sharpe=1.5,
                        holdout_sharpe=0.8, tested_at="2026-07-18T00:00:00"):
    """Build a synthetic adopted entry for testing."""
    return {
        "hypothesis": {
            "strategy": strategy,
            "symbol": symbol,
            "params": params,
        },
        "evaluation": {
            "verdict": "adopted",
            "sharpe_ratio": val_sharpe,
            "total_return_pct": 25.0,
            "max_drawdown_pct": -10.0,
            "holdout_sharpe": holdout_sharpe,
            "holdout_return_pct": 15.0,
            "gate_results": {
                "validation": True,
                "deflation": True,
                "holdout": True,
                "cluster": "new",
            },
            "effective_min_sharpe": 0.5,
            "cluster_id": "test1234",
        },
        "verdict": "adopted",
        "tested_at": tested_at,
    }


# ============================================================
# Family key helper
# ============================================================

class TestFamilyKey:
    def test_family_key_form(self):
        assert _family_key("sma_crossover", "AAPL") == "sma_crossover|AAPL"


# ============================================================
# Aggregate update on adopted
# ============================================================

class TestAggregateUpdateOnAdopted:
    def test_adopted_increments_trials_and_tracks_best_sharpe(self):
        knowledge = {"families": {}}
        entry = _make_adopted_entry("sma_crossover", "AAPL",
                                     {"fast_window": 10, "slow_window": 30},
                                     val_sharpe=1.5)
        update_family_aggregates(knowledge, entry)

        fam = knowledge["families"]["sma_crossover|AAPL"]
        assert fam["n_trials"] == 1
        assert fam["best_val_sharpe"] == 1.5
        assert fam["best_params"] == {"fast_window": 10, "slow_window": 30}
        # Adopted should NOT increment gate_failures
        assert fam["gate_failures"]["validation"] == 0
        assert fam["gate_failures"]["holdout"] == 0

    def test_adopted_updates_best_only_when_higher(self):
        knowledge = {"families": {}}
        e1 = _make_adopted_entry("sma_crossover", "AAPL",
                                  {"fast_window": 10, "slow_window": 30},
                                  val_sharpe=0.8)
        e2 = _make_adopted_entry("sma_crossover", "AAPL",
                                  {"fast_window": 12, "slow_window": 35},
                                  val_sharpe=1.6)
        update_family_aggregates(knowledge, e1)
        update_family_aggregates(knowledge, e2)

        fam = knowledge["families"]["sma_crossover|AAPL"]
        assert fam["n_trials"] == 2
        assert fam["best_val_sharpe"] == 1.6
        assert fam["best_params"] == {"fast_window": 12, "slow_window": 35}


# ============================================================
# Aggregate update on rejected
# ============================================================

class TestAggregateUpdateOnRejected:
    def test_rejected_increments_trials_and_gate_failures(self):
        knowledge = {"families": {}}
        entry = _make_rejected_entry("sma_crossover", "AAPL",
                                      {"fast_window": 10, "slow_window": 30},
                                      val_sharpe=-0.3,
                                      gate_validation=False, gate_holdout=True)
        update_family_aggregates(knowledge, entry)

        fam = knowledge["families"]["sma_crossover|AAPL"]
        assert fam["n_trials"] == 1
        assert fam["gate_failures"]["validation"] == 1

    def test_rejected_holdout_failure_counted(self):
        knowledge = {"families": {}}
        entry = _make_rejected_entry("sma_crossover", "AAPL",
                                      {"fast_window": 10, "slow_window": 30},
                                      val_sharpe=1.2,
                                      gate_validation=True, gate_holdout=False)
        update_family_aggregates(knowledge, entry)

        fam = knowledge["families"]["sma_crossover|AAPL"]
        assert fam["gate_failures"]["holdout"] == 1

    def test_duplicate_cluster_counted(self):
        knowledge = {"families": {}}
        entry = _make_rejected_entry("sma_crossover", "AAPL",
                                      {"fast_window": 11, "slow_window": 32},
                                      val_sharpe=1.0,
                                      gate_validation=True, gate_holdout=True,
                                      cluster_gate="duplicate_cluster")
        update_family_aggregates(knowledge, entry)

        fam = knowledge["families"]["sma_crossover|AAPL"]
        assert fam["gate_failures"]["duplicate_cluster"] == 1


# ============================================================
# Migration rebuild from legacy knowledge without "families"
# ============================================================

class TestMigrationRebuild:
    def test_rebuild_from_adopted_rejected_superseded(self):
        knowledge = {
            "adopted": [
                _make_adopted_entry("sma_crossover", "AAPL",
                                     {"fast_window": 10, "slow_window": 30},
                                     val_sharpe=1.5),
            ],
            "rejected": [
                _make_rejected_entry("sma_crossover", "AAPL",
                                      {"fast_window": 5, "slow_window": 20},
                                      val_sharpe=-0.5,
                                      gate_validation=False),
                _make_rejected_entry("mean_reversion", "MSFT",
                                      {"window": 30, "threshold": 2.0},
                                      val_sharpe=0.3,
                                      gate_validation=True, gate_holdout=False),
            ],
            "superseded": [
                _make_adopted_entry("sma_crossover", "NVDA",
                                     {"fast_window": 8, "slow_window": 25},
                                     val_sharpe=0.7),
            ],
        }
        families = _rebuild_families_from_history(knowledge)

        assert "sma_crossover|AAPL" in families
        assert families["sma_crossover|AAPL"]["n_trials"] == 2  # 1 adopted + 1 rejected
        assert families["sma_crossover|AAPL"]["best_val_sharpe"] == 1.5
        assert families["sma_crossover|AAPL"]["gate_failures"]["validation"] == 1

        assert "mean_reversion|MSFT" in families
        assert families["mean_reversion|MSFT"]["n_trials"] == 1
        assert families["mean_reversion|MSFT"]["gate_failures"]["holdout"] == 1

        assert "sma_crossover|NVDA" in families
        assert families["sma_crossover|NVDA"]["n_trials"] == 1

    def test_rebuild_empty_knowledge(self):
        families = _rebuild_families_from_history({"adopted": [], "rejected": [], "superseded": []})
        assert families == {}


# ============================================================
# Exhausted-cluster pre-block
# ============================================================

class TestExhaustedCluster:
    def test_triggers_at_3_negative_failures(self):
        """3 rejected entries in same cluster with best Sharpe < 0 → exhausted."""
        rejected = [
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 10, "slow_window": 30},
                                  val_sharpe=-0.8, gate_validation=False),
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 11, "slow_window": 32},
                                  val_sharpe=-0.3, gate_validation=False),
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 9, "slow_window": 28},
                                  val_sharpe=-0.5, gate_validation=False),
        ]
        knowledge = {"rejected": rejected}
        hypothesis = {"strategy": "sma_crossover", "symbol": "AAPL",
                       "params": {"fast_window": 10, "slow_window": 30}}
        exhausted, count, best = _check_exhausted_cluster(hypothesis, knowledge)
        assert exhausted is True
        assert count == 3
        assert best == pytest.approx(-0.3)

    def test_does_not_trigger_at_2_failures(self):
        """Only 2 failures → not exhausted."""
        rejected = [
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 10, "slow_window": 30},
                                  val_sharpe=-0.8, gate_validation=False),
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 11, "slow_window": 32},
                                  val_sharpe=-0.3, gate_validation=False),
        ]
        knowledge = {"rejected": rejected}
        hypothesis = {"strategy": "sma_crossover", "symbol": "AAPL",
                       "params": {"fast_window": 10, "slow_window": 30}}
        exhausted, count, best = _check_exhausted_cluster(hypothesis, knowledge)
        assert exhausted is False
        assert count == 2

    def test_does_not_trigger_when_best_sharpe_positive(self):
        """3 failures but best Sharpe >= 0 → NOT exhausted (there's a signal)."""
        rejected = [
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 10, "slow_window": 30},
                                  val_sharpe=-0.8, gate_validation=False),
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 11, "slow_window": 32},
                                  val_sharpe=0.2, gate_validation=False),
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 9, "slow_window": 28},
                                  val_sharpe=-0.5, gate_validation=False),
        ]
        knowledge = {"rejected": rejected}
        hypothesis = {"strategy": "sma_crossover", "symbol": "AAPL",
                       "params": {"fast_window": 10, "slow_window": 30}}
        exhausted, count, best = _check_exhausted_cluster(hypothesis, knowledge)
        assert exhausted is False
        assert best == 0.2

    def test_params_outside_cluster_not_counted(self):
        """Rejected entries with params far from hypothesis → not in cluster → not exhausted."""
        rejected = [
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 25, "slow_window": 90},
                                  val_sharpe=-0.8, gate_validation=False),
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 26, "slow_window": 92},
                                  val_sharpe=-0.5, gate_validation=False),
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 27, "slow_window": 93},
                                  val_sharpe=-0.3, gate_validation=False),
        ]
        knowledge = {"rejected": rejected}
        hypothesis = {"strategy": "sma_crossover", "symbol": "AAPL",
                       "params": {"fast_window": 5, "slow_window": 20}}
        exhausted, count, best = _check_exhausted_cluster(hypothesis, knowledge)
        assert exhausted is False


# ============================================================
# Prompt content assertions
# ============================================================

def _synthetic_knowledge():
    """Build a knowledge dict with families, adopted, rejected for prompt testing."""
    return {
        "families": {
            "sma_crossover|AAPL": {
                "n_trials": 5,
                "best_val_sharpe": -0.2,
                "best_params": {"fast_window": 12, "slow_window": 35},
                "gate_failures": {"validation": 3, "deflation": 0, "holdout": 1,
                                  "duplicate_cluster": 1, "exhausted_cluster": 0},
                "last_tried": "2026-07-18T00:00:00",
            },
            "mean_reversion|MSFT": {
                "n_trials": 2,
                "best_val_sharpe": 1.2,
                "best_params": {"window": 30, "threshold": 2.0},
                "gate_failures": {"validation": 0, "deflation": 0, "holdout": 1,
                                  "duplicate_cluster": 0, "exhausted_cluster": 0},
                "last_tried": "2026-07-18T00:00:00",
            },
        },
        "adopted": [
            _make_adopted_entry("mean_reversion", "MSFT",
                                 {"window": 30, "threshold": 2.0},
                                 val_sharpe=1.2, holdout_sharpe=0.9),
        ],
        "rejected": [
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 10, "slow_window": 30},
                                  val_sharpe=-0.8, gate_validation=False,
                                  gate_holdout=True),
            _make_rejected_entry("sma_crossover", "AAPL",
                                  {"fast_window": 15, "slow_window": 40},
                                  val_sharpe=0.6, gate_validation=True,
                                  gate_holdout=False),
        ],
        "tested_combinations": [
            {"strategy": "sma_crossover", "symbol": "AAPL",
             "params": {"fast_window": 10, "slow_window": 30}},
        ],
    }


TEMPLATES_SIMPLE = {
    "sma_crossover": {
        "description": "SMA cross",
        "param_space": {"fast_window": range(5, 30), "slow_window": range(20, 100)},
    },
    "mean_reversion": {
        "description": "Mean rev",
        "param_space": {"window": range(10, 60), "threshold": [1.0, 1.5, 2.0, 2.5, 3.0]},
    },
}

class TestPromptContent:
    def test_prompt_contains_family_lines(self):
        knowledge = _synthetic_knowledge()
        messages = _build_prompt(knowledge, TEMPLATES_SIMPLE)
        user_content = messages[1]["content"]

        assert "sma_crossover on AAPL" in user_content
        assert "mean_reversion on MSFT" in user_content
        assert "5 trials" in user_content
        assert "best val Sharpe -0.20" in user_content

    def test_prompt_contains_exhausted_marker(self):
        """Family with n>=3 and best Sharpe < 0 → EXHAUSTED marker in prompt."""
        knowledge = _synthetic_knowledge()
        messages = _build_prompt(knowledge, TEMPLATES_SIMPLE)
        user_content = messages[1]["content"]

        assert "EXHAUSTED" in user_content
        assert "do not propose params near previous attempts" in user_content

    def test_prompt_contains_gate_reason_strings(self):
        """Rejected entries should show which gate they failed at."""
        knowledge = _synthetic_knowledge()
        messages = _build_prompt(knowledge, TEMPLATES_SIMPLE)
        user_content = messages[1]["content"]

        assert "rejected at:" in user_content
        assert "validation failed" in user_content
        assert "holdout failed" in user_content

    def test_prompt_contains_failure_learning_instructions(self):
        knowledge = _synthetic_knowledge()
        messages = _build_prompt(knowledge, TEMPLATES_SIMPLE)
        user_content = messages[1]["content"]

        assert "respond to WHY" in user_content
        assert "Holdout failures" in user_content
        assert "overfitting" in user_content

    def test_compact_prompt_under_target_tokens(self):
        """The prompt should be compact — rough token estimate ≤ ~2500 chars."""
        knowledge = _synthetic_knowledge()
        messages = _build_prompt(knowledge, TEMPLATES_SIMPLE)
        user_content = messages[1]["content"]
        # Rough: 1 token ≈ 4 chars, 2000 tokens ≈ 8000 chars. Staying well under.
        assert len(user_content) < 8000, f"Prompt too long: {len(user_content)} chars"


# ============================================================
# Knowledge summary direct test
# ============================================================

class TestKnowledgeSummaryDirect:
    def test_empty_knowledge_no_crash(self):
        knowledge = {"families": {}, "adopted": [], "rejected": [], "tested_combinations": []}
        result = _build_knowledge_summary(knowledge)
        assert "none yet" in result
        assert "Family aggregates: none yet" in result

    def test_exhausted_marker_not_set_for_non_exhausted(self):
        """Family with n=2, best Sharpe >= 0 → no EXHAUSTED marker."""
        knowledge = {
            "families": {
                "sma_crossover|AAPL": {
                    "n_trials": 2,
                    "best_val_sharpe": 0.5,
                    "best_params": {},
                    "gate_failures": {"validation": 1, "deflation": 0, "holdout": 0,
                                      "duplicate_cluster": 0, "exhausted_cluster": 0},
                    "last_tried": "",
                }
            },
            "adopted": [],
            "rejected": [],
        }
        result = _build_knowledge_summary(knowledge)
        assert "EXHAUSTED" not in result
