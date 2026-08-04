"""
Regression tests for issue #72: Review loop has no param schema contract.

The review loop's LLM generates refine proposals with invented param names
(e.g. lookback_days, volatility_scaling, rsi_period) that don't match the
runner's schema. Before the fix, these were added to the backlog unconditionally,
guaranteed to fail at the runner with HTTP 400.

Two fixes:
1. The judge prompt includes valid param schemas so the LLM knows what's valid.
2. apply_verdict validates refine params against STRATEGY_TEMPLATES before
   adding to backlog; invalid proposals are downgraded to keep.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_constants import FamilyLifecycle, REFINE_CAP
import strategy_review as sr
from autonomous_loop import STRATEGY_TEMPLATES


# ============================================================================
# Helpers (same pattern as test_review_verdict.py)
# ============================================================================

def _make_family(key, family_type="single", lifecycle=FamilyLifecycle.CANDIDATE,
                 best_val_sharpe=0.5, kill_reason="", refine_count=0,
                 n_trials=5, gate_failures=None):
    return {
        "n_trials": n_trials,
        "best_val_sharpe": best_val_sharpe,
        "best_params": {},
        "gate_failures": gate_failures or {"validation": 0, "deflation": 0, "holdout": 0,
                                            "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": "2026-07-01T00:00:00",
        "family_type": family_type,
        "lifecycle": lifecycle,
        "refine_count": refine_count,
        "kill_reason": kill_reason,
    }


def _make_report(family_key="sma_crossover|AAPL", family_type="single",
                 flags=None, baseline_val_sharpe=0.5, baseline_val_turnover=30.0,
                 fold_sharpes=None, cost0_val_sharpe=0.7):
    report = {
        "family_key": family_key,
        "family_type": family_type,
        "diagnosed_at": "2026-07-20T00:00:00",
        "flags": flags or {"cost_bound": False, "no_signal": False, "unstable": False,
                           "regime_dependent": False, "high_turnover": False},
        "baseline": {
            "val_sharpe": baseline_val_sharpe,
            "val_turnover": baseline_val_turnover,
        },
        "cost_free": {"val_sharpe": cost0_val_sharpe},
        "folds_available": True,
    }
    if fold_sharpes is not None:
        report["baseline"]["fold_sharpes"] = fold_sharpes
    return report


def _make_knowledge(families=None):
    return {
        "families": families or {},
        "rejected": [],
        "near_misses": [],
        "near_misses_cross": [],
        "review_state": {},
        "reviews": [],
        "adopted": [],
        "tested_combinations": [],
        "errors": [],
    }


# ============================================================================
# 1. Preflight validation: invalid params must NOT create backlog entries
# ============================================================================

class TestRefineParamSchemaValidation:
    """apply_verdict must validate refine params against STRATEGY_TEMPLATES."""

    def test_invented_param_name_downgraded_to_keep(self, tmp_path, monkeypatch):
        """LLM proposes lookback_days (not in schema) → keep, no backlog entry."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "momentum|META"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog_path = tmp_path / "backlog.json"
        backlog = Backlog(str(backlog_path))
        # LLM invents "lookback_days" and "volatility_scaling" — not in momentum schema
        verdict = {
            "verdict": "refine",
            "rationale": "cost_bound flag, try longer lookback",
            "refine_proposal": {
                "params": {"lookback_days": 60, "volatility_scaling": True},
                "change_summary": " Longer lookback with vol scaling",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        # Must be downgraded to keep
        assert result == "keep"

        # No backlog entry should be created
        bl2 = Backlog(str(backlog_path))
        data = bl2.load()
        assert len(data["entries"]) == 0

        # refine_count must NOT be incremented
        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["refine_count"] == 0

    def test_extra_keys_in_params_downgraded_to_keep(self, tmp_path, monkeypatch):
        """LLM includes 'symbol' and 'strategy' meta-keys → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AMZN"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog_path = tmp_path / "backlog.json"
        backlog = Backlog(str(backlog_path))
        # LLM includes meta-keys + correct param names
        verdict = {
            "verdict": "refine",
            "rationale": "try longer windows",
            "refine_proposal": {
                "params": {"symbol": "AMZN", "strategy": "sma_crossover",
                           "fast_window": 20, "slow_window": 50},
                "change_summary": "longer windows",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "keep"
        bl2 = Backlog(str(backlog_path))
        data = bl2.load()
        assert len(data["entries"]) == 0

    def test_out_of_range_value_downgraded_to_keep(self, tmp_path, monkeypatch):
        """LLM proposes fast_window=300 (out of range 5..29) → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog_path = tmp_path / "backlog.json"
        backlog = Backlog(str(backlog_path))
        verdict = {
            "verdict": "refine",
            "rationale": "try much longer windows",
            "refine_proposal": {
                "params": {"fast_window": 300, "slow_window": 50},
                "change_summary": "very long fast window",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "keep"
        bl2 = Backlog(str(backlog_path))
        data = bl2.load()
        assert len(data["entries"]) == 0

    def test_rsi_period_invented_name_downgraded(self, tmp_path, monkeypatch):
        """LLM proposes rsi_period (not in rsi schema — should be rsi_window) → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "rsi|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog_path = tmp_path / "backlog.json"
        backlog = Backlog(str(backlog_path))
        verdict = {
            "verdict": "refine",
            "rationale": "adjust RSI period",
            "refine_proposal": {
                "params": {"rsi_period": 30},
                "change_summary": "longer RSI period",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "keep"
        bl2 = Backlog(str(backlog_path))
        data = bl2.load()
        assert len(data["entries"]) == 0

    def test_valid_params_still_accepted(self, tmp_path, monkeypatch):
        """Valid params (matching schema exactly) still create backlog entry."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog_path = tmp_path / "backlog.json"
        backlog = Backlog(str(backlog_path))
        # Valid: fast_window in range(5,30), slow_window in range(20,100)
        verdict = {
            "verdict": "refine",
            "rationale": "longer windows to reduce noise",
            "refine_proposal": {
                "params": {"fast_window": 25, "slow_window": 90},
                "change_summary": "longer windows",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "refine"
        bl2 = Backlog(str(backlog_path))
        data = bl2.load()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["spec"]["params"] == {"fast_window": 25, "slow_window": 90}

    def test_valid_rsi_params_accepted(self, tmp_path, monkeypatch):
        """Valid RSI params (all 3 required keys with valid values) accepted."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "rsi|META"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog_path = tmp_path / "backlog.json"
        backlog = Backlog(str(backlog_path))
        verdict = {
            "verdict": "refine",
            "rationale": "adjust RSI thresholds",
            "refine_proposal": {
                "params": {"rsi_window": 14, "oversold": 25, "overbought": 75},
                "change_summary": "wider RSI bands",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "refine"
        bl2 = Backlog(str(backlog_path))
        data = bl2.load()
        assert len(data["entries"]) == 1


# ============================================================================
# 2. Prompt includes schema information
# ============================================================================

class TestJudgePromptIncludesSchema:
    """_build_judge_prompt must include valid param schemas for the family's strategy."""

    def test_prompt_contains_param_space_for_family_strategy(self):
        """Prompt for sma_crossover|AAPL includes fast_window and slow_window."""
        report = _make_report(family_key="sma_crossover|AAPL")
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        messages = sr._build_judge_prompt(report, family, knowledge)
        # Concatenate all message content
        full_text = " ".join(m["content"] for m in messages)

        assert "fast_window" in full_text
        assert "slow_window" in full_text

    def test_prompt_contains_param_space_for_momentum(self):
        """Prompt for momentum|META includes lookback and hold_period."""
        report = _make_report(family_key="momentum|META")
        family = _make_family("momentum|META")
        knowledge = _make_knowledge()

        messages = sr._build_judge_prompt(report, family, knowledge)
        full_text = " ".join(m["content"] for m in messages)

        assert "lookback" in full_text
        assert "hold_period" in full_text

    def test_prompt_contains_param_space_for_rsi(self):
        """Prompt for rsi|AAPL includes rsi_window, oversold, overbought."""
        report = _make_report(family_key="rsi|AAPL")
        family = _make_family("rsi|AAPL")
        knowledge = _make_knowledge()

        messages = sr._build_judge_prompt(report, family, knowledge)
        full_text = " ".join(m["content"] for m in messages)

        assert "rsi_window" in full_text
        assert "oversold" in full_text
        assert "overbought" in full_text

    def test_prompt_does_not_include_unrelated_strategy_params(self):
        """Prompt for sma_crossover should NOT include rsi_window or lookback."""
        report = _make_report(family_key="sma_crossover|AAPL")
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        messages = sr._build_judge_prompt(report, family, knowledge)
        full_text = " ".join(m["content"] for m in messages)

        # These belong to other strategies, not sma_crossover
        assert "rsi_window" not in full_text
        assert "lookback" not in full_text

    def test_prompt_includes_range_constraints(self):
        """Prompt includes range info (e.g. '5..29') so LLM knows bounds."""
        report = _make_report(family_key="sma_crossover|AAPL")
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        messages = sr._build_judge_prompt(report, family, knowledge)
        full_text = " ".join(m["content"] for m in messages)

        # fast_window: range(5, 30) → should show bounds
        assert "5" in full_text  # start of range
        assert "29" in full_text  # end of range (stop - 1)
