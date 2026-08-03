"""
Tests for issue #72: Review loop param schema contract.

The review loop's LLM judge must:
1. Include valid param schemas in the prompt (so the LLM knows what names to use).
2. Validate refine params against STRATEGY_TEMPLATES before adding to backlog
   (so invalid params are caught and the refine slot is not wasted).

Regression tests verify:
- Prompt contains param schema for the family's strategy.
- Invalid params (wrong keys, extra keys, meta-keys) are rejected at apply_verdict.
- Valid params still pass through.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_constants import FamilyLifecycle, REFINE_CAP
from autonomous_loop import STRATEGY_TEMPLATES

# Module under test
import strategy_review as sr


# ============================================================================
# Helpers
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


def _make_report(family_key="momentum|AAPL", family_type="single",
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


def _make_knowledge(families=None, rejected=None, near_misses=None,
                    near_misses_cross=None, review_state=None):
    return {
        "families": families or {},
        "rejected": rejected or [],
        "near_misses": near_misses or [],
        "near_misses_cross": near_misses_cross or [],
        "review_state": review_state or {},
        "reviews": [],
        "adopted": [],
        "tested_combinations": [],
        "errors": [],
    }


# ============================================================================
# 1. Prompt includes param schema
# ============================================================================

class TestPromptIncludesSchema:
    def test_prompt_contains_strategy_param_names(self, tmp_path, monkeypatch):
        """Judge prompt for momentum family must mention lookback and hold_period."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        captured_messages = []

        def _capture(messages):
            captured_messages.append(messages)
            return {"verdict": "keep", "rationale": "fine", "refine_proposal": None}

        monkeypatch.setattr(sr, "_call_review_llm", _capture)

        report = _make_report(family_key="momentum|AAPL")
        family = _make_family("momentum|AAPL")
        knowledge = _make_knowledge()

        sr.judge_family(report, family, knowledge)
        assert len(captured_messages) == 1
        user_content = captured_messages[0][1]["content"]

        # The valid param names for momentum must appear in the prompt
        assert "lookback" in user_content
        assert "hold_period" in user_content

    def test_prompt_contains_rsi_param_names(self, tmp_path, monkeypatch):
        """Judge prompt for rsi family must mention rsi_window, oversold, overbought."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        captured_messages = []

        def _capture(messages):
            captured_messages.append(messages)
            return {"verdict": "keep", "rationale": "fine", "refine_proposal": None}

        monkeypatch.setattr(sr, "_call_review_llm", _capture)

        report = _make_report(family_key="rsi|SPY")
        family = _make_family("rsi|SPY")
        knowledge = _make_knowledge()

        sr.judge_family(report, family, knowledge)
        user_content = captured_messages[0][1]["content"]

        assert "rsi_window" in user_content
        assert "oversold" in user_content
        assert "overbought" in user_content

    def test_prompt_contains_sma_crossover_param_names(self, tmp_path, monkeypatch):
        """Judge prompt for sma_crossover family must mention fast_window, slow_window."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        captured_messages = []

        def _capture(messages):
            captured_messages.append(messages)
            return {"verdict": "keep", "rationale": "fine", "refine_proposal": None}

        monkeypatch.setattr(sr, "_call_review_llm", _capture)

        report = _make_report(family_key="sma_crossover|AAPL")
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        sr.judge_family(report, family, knowledge)
        user_content = captured_messages[0][1]["content"]

        assert "fast_window" in user_content
        assert "slow_window" in user_content


# ============================================================================
# 2. Preflight validation rejects invalid params
# ============================================================================

class TestRefineParamValidation:
    def test_invalid_param_keys_rejected(self, tmp_path, monkeypatch):
        """Refine with wrong param names (e.g. lookback_days) → downgraded to keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge
        from backlog import Backlog

        family_key = "momentum|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "cost_bound — try longer lookback",
            "refine_proposal": {
                # WRONG keys: lookback_days instead of lookback, target_vol is bogus
                "params": {"lookback_days": 60, "target_vol": 0.3},
                "change_summary": "Longer lookback to reduce cost",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        # Should be downgraded to keep — invalid params
        assert result == "keep"

        # No backlog entry should be created
        data = backlog.load()
        assert len(data["entries"]) == 0

    def test_extra_meta_keys_rejected(self, tmp_path, monkeypatch):
        """Refine with meta-keys (symbol, strategy) mixed in → downgraded to keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AMZN"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "try longer windows",
            "refine_proposal": {
                # Includes meta-keys symbol and strategy alongside valid params
                "params": {"symbol": "AMZN", "strategy": "sma_crossover",
                           "fast_window": 20, "slow_window": 50},
                "change_summary": "Longer windows",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "keep"
        data = backlog.load()
        assert len(data["entries"]) == 0

    def test_wrong_strategy_param_set_rejected(self, tmp_path, monkeypatch):
        """Refine with rsi params for a momentum family → downgraded to keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge
        from backlog import Backlog

        family_key = "momentum|MSFT"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "try different period",
            "refine_proposal": {
                # rsi_period is not a valid momentum param
                "params": {"rsi_period": 30},
                "change_summary": "Different period",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "keep"
        data = backlog.load()
        assert len(data["entries"]) == 0

    def test_valid_params_still_pass(self, tmp_path, monkeypatch):
        """Refine with correct param names and valid values → still works."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "momentum|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "cost_bound — try longer lookback",
            "refine_proposal": {
                "params": {"lookback": 30, "hold_period": 10},
                "change_summary": "Longer lookback to reduce cost impact",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "refine"
        data = backlog.load()
        assert len(data["entries"]) == 1

    def test_unknown_strategy_skips_validation(self, tmp_path, monkeypatch):
        """If strategy is not in STRATEGY_TEMPLATES, skip schema validation (pass through)."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge
        from backlog import Backlog

        family_key = "custom_strategy|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "try something",
            "refine_proposal": {
                "params": {"custom_param": 42},
                "change_summary": "Custom change",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        # Unknown strategy → skip validation → refine succeeds
        assert result == "refine"


# ============================================================================
# 3. _validate_refine_params helper unit tests
# ============================================================================

class TestValidateRefineParams:
    def test_valid_momentum_params(self):
        ok, err = sr._validate_refine_params("momentum", {"lookback": 20, "hold_period": 5})
        assert ok is True
        assert err is None

    def test_wrong_keys(self):
        ok, err = sr._validate_refine_params("momentum", {"lookback_days": 20, "hold_period": 5})
        assert ok is False
        assert "lookback_days" in err or "expected" in err.lower()

    def test_extra_keys(self):
        ok, err = sr._validate_refine_params("momentum",
                                              {"lookback": 20, "hold_period": 5, "extra": 1})
        assert ok is False

    def test_missing_keys(self):
        ok, err = sr._validate_refine_params("momentum", {"lookback": 20})
        assert ok is False

    def test_unknown_strategy_passes(self):
        """Unknown strategy → validation passes (no schema to check against)."""
        ok, err = sr._validate_refine_params("nonexistent", {"anything": 1})
        assert ok is True
        assert err is None

    def test_rsi_valid(self):
        ok, err = sr._validate_refine_params("rsi",
                                              {"rsi_window": 14, "oversold": 30, "overbought": 70})
        assert ok is True

    def test_rsi_invalid_value(self):
        ok, err = sr._validate_refine_params("rsi",
                                              {"rsi_window": 14, "oversold": 99, "overbought": 70})
        assert ok is False
        assert "oversold" in err

    def test_meta_keys_rejected(self):
        """Including 'symbol' or 'strategy' as a param key is rejected."""
        ok, err = sr._validate_refine_params("sma_crossover",
                                              {"fast_window": 10, "slow_window": 50, "symbol": "AAPL"})
        assert ok is False
