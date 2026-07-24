"""
Tests for PR-C: strategy_review.py verdict stage.

Covers: judge_family, apply_verdict, fail-open, prompt content,
CLI --dry-run on judge path, disk round-trip for verdicts.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_constants import FamilyLifecycle, REFINE_CAP

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


def _make_near_miss(strategy="sma_crossover", symbol="AAPL", params=None,
                    val_sharpe=0.7, date="2026-07-01T00:00:00",
                    holdout_sharpe=-0.1, failed_gate="holdout"):
    return {
        "id": "nm1",
        "strategy": strategy,
        "symbol": symbol,
        "params": params or {"fast_window": 15, "slow_window": 40},
        "val_sharpe": val_sharpe,
        "deflated_threshold": 0.8,
        "holdout_sharpe": holdout_sharpe,
        "failed_gate": failed_gate,
        "date": date,
    }


# ============================================================================
# 1. Verdict application — kill
# ============================================================================

class TestApplyVerdictKill:
    def test_kill_sets_lifecycle_and_reason(self, tmp_path, monkeypatch):
        """Kill verdict sets lifecycle=KILLED and kill_reason."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, n_trials=5)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {"verdict": "kill", "rationale": "no signal, consistently bad", "refine_proposal": None}
        sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        # Disk round-trip
        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["lifecycle"] == FamilyLifecycle.KILLED
        assert fam["kill_reason"] == "auto: no signal, consistently bad"

    def test_kill_downgraded_insufficient_evidence(self, tmp_path, monkeypatch):
        """Kill verdict with n_trials < 3 → keep (downgraded)."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, n_trials=2)},  # < MIN_TRIALS_FOR_KILL
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {"verdict": "kill", "rationale": "should be killed", "refine_proposal": None}
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "keep"

        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["lifecycle"] != FamilyLifecycle.KILLED
        assert fam["kill_reason"] == ""  # not set

    def test_kill_allowed_at_boundary(self, tmp_path, monkeypatch):
        """Kill allowed at exactly MIN_TRIALS_FOR_KILL (3)."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, n_trials=3)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {"verdict": "kill", "rationale": "enough evidence", "refine_proposal": None}
        sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["lifecycle"] == FamilyLifecycle.KILLED
        assert "auto: enough evidence" in fam["kill_reason"]


# ============================================================================
# 2. Verdict application — refine
# ============================================================================

class TestApplyVerdictRefine:
    def test_refine_increments_count_and_sets_lifecycle(self, tmp_path, monkeypatch):
        """Refine sets lifecycle=REFINING, increments refine_count."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, refine_count=1)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "cost_bound flag, try longer windows",
            "refine_proposal": {
                "params": {"fast_window": 30, "slow_window": 80},
                "change_summary": "Longer windows to reduce cost impact",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "refine"

        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["refine_count"] == 2
        assert fam["lifecycle"] == FamilyLifecycle.REFINING

    def test_refine_creates_backlog_entry(self, tmp_path, monkeypatch):
        """Refine creates a proper backlog entry with correct shape."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge
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
            "rationale": "high turnover — reduce trade frequency",
            "refine_proposal": {
                "params": {"fast_window": 40, "slow_window": 100},
                "change_summary": "Longer windows for fewer trades",
            },
        }
        sr.apply_verdict(family_key, verdict, knowledge, backlog)

        # Disk round-trip on backlog
        bl2 = Backlog(str(backlog_path))
        data = bl2.load()
        entries = data["entries"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["type"] == "param"
        assert entry["priority"] == 0.95
        assert entry["source"]["kind"] == "review_refine"
        assert entry["source"]["ref"] == family_key
        assert entry["spec"]["strategy"] == "sma_crossover"
        assert entry["spec"]["symbol"] == "AAPL"
        assert entry["spec"]["params"] == {"fast_window": 40, "slow_window": 100}
        assert "id" in entry
        assert len(entry["id"]) > 0  # uuid assigned
        assert entry["status"] == "pending"
        assert entry["eval_plan"] == {"extra_criteria": []}

    def test_refine_cap_kills_family(self, tmp_path, monkeypatch):
        """Refine at refine_count == REFINE_CAP → killed with cap reason."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, refine_count=REFINE_CAP, n_trials=1)},  # n_trials=1 ignored by cap
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "try again",
            "refine_proposal": {
                "params": {"fast_window": 50, "slow_window": 120},
                "change_summary": "final attempt",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "kill"

        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["lifecycle"] == FamilyLifecycle.KILLED
        assert fam["kill_reason"] == "auto: refine cap exhausted"

    def test_cross_refine_downgraded_to_keep(self, tmp_path, monkeypatch):
        """Cross family refine → keep (downgraded)."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "manifest:xs_mom|universe:abc"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, family_type="cross")},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "try different params",
            "refine_proposal": {
                "params": {"universe_size": 10},
                "change_summary": "Bigger universe",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "keep"

        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        # Should not be KILLED or REFINING
        assert fam["lifecycle"] == FamilyLifecycle.CANDIDATE


# ============================================================================
# 3. Verdict application — keep
# ============================================================================

class TestApplyVerdictKeep:
    def test_keep_records_only(self, tmp_path, monkeypatch):
        """Keep verdict does not change lifecycle or refine_count."""
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

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {"verdict": "keep", "rationale": "needs more data", "refine_proposal": None}
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "keep"

        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["lifecycle"] == FamilyLifecycle.CANDIDATE
        assert fam["refine_count"] == 0
        assert fam["kill_reason"] == ""

    def test_keep_appends_review_summary(self, tmp_path, monkeypatch):
        """Every verdict appends to knowledge['reviews']."""
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

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {"verdict": "keep", "rationale": "needs more data", "refine_proposal": None}
        sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        reloaded = load_knowledge()
        assert len(reloaded["reviews"]) == 1
        assert reloaded["reviews"][0]["family_key"] == family_key
        assert reloaded["reviews"][0]["llm_verdict"] == "keep"
        assert reloaded["reviews"][0]["applied_verdict"] == "keep"
        assert reloaded["reviews"][0]["rationale"] == "needs more data"
        assert "at" in reloaded["reviews"][0]


# ============================================================================
# 4. Fail-open (judge_family)
# ============================================================================

class TestFailOpen:
    def test_exception_fail_open(self, tmp_path, monkeypatch):
        """LLM exception → keep with llm_failure rationale."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        def _fail(*args, **kwargs):
            raise RuntimeError("API down")

        monkeypatch.setattr(sr, "_call_review_llm", _fail)

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "llm_failure" in verdict["rationale"]
        assert "API down" in verdict["rationale"]
        assert verdict["refine_proposal"] is None

    def test_empty_content_fail_open(self, tmp_path, monkeypatch):
        """Empty LLM response → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        def _return_empty(*args, **kwargs):
            raise json.JSONDecodeError("empty", "", 0)

        monkeypatch.setattr(sr, "_call_review_llm", _return_empty)

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "llm_failure" in verdict["rationale"]

    def test_not_a_dict_fail_open(self, tmp_path, monkeypatch):
        """LLM returns a list instead of dict → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm", lambda _: ["not", "a", "dict"])

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "llm_failure: not_a_dict" == verdict["rationale"]

    def test_missing_verdict_fail_open(self, tmp_path, monkeypatch):
        """LLM response missing verdict field → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm", lambda _: {"rationale": "no verdict here"})

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "invalid_verdict" in verdict["rationale"]

    def test_invalid_verdict_value_fail_open(self, tmp_path, monkeypatch):
        """LLM returns unknown verdict value → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm", lambda _: {"verdict": "banana", "rationale": "wtf"})

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "banana" in verdict["rationale"]

    def test_refine_missing_proposal_fail_open(self, tmp_path, monkeypatch):
        """Refine verdict without refine_proposal → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm",
                            lambda _: {"verdict": "refine", "rationale": "try again", "refine_proposal": None})

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "missing_refine_proposal" in verdict["rationale"]

    def test_refine_bad_params_shape_fail_open(self, tmp_path, monkeypatch):
        """Refine with empty params dict → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm",
                            lambda _: {"verdict": "refine", "rationale": "try",
                                       "refine_proposal": {"params": {}, "change_summary": "x"}})

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "bad_refine_params" in verdict["rationale"]

    def test_refine_params_bad_type_fail_open(self, tmp_path, monkeypatch):
        """Refine with non-int/float/str/bool param value → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm",
                            lambda _: {"verdict": "refine", "rationale": "try",
                                       "refine_proposal": {"params": {"fast_window": [1, 2, 3]}, "change_summary": "x"}})

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "bad_param_type" in verdict["rationale"]

    def test_refine_proposal_not_dict_fail_open(self, tmp_path, monkeypatch):
        """Refine with refine_proposal as string → keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm",
                            lambda _: {"verdict": "refine", "rationale": "try",
                                       "refine_proposal": "not a dict"})

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "keep"
        assert "refine_proposal_not_dict" in verdict["rationale"]

    def test_valid_refine_passes(self, tmp_path, monkeypatch):
        """Valid refine verdict passes through."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm",
                            lambda _: {"verdict": "refine", "rationale": "cost bound",
                                       "refine_proposal": {"params": {"fast_window": 30, "slow_window": 80},
                                                           "change_summary": "Longer windows"}})

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "refine"
        assert verdict["refine_proposal"]["params"]["fast_window"] == 30

    def test_valid_kill_passes(self, tmp_path, monkeypatch):
        """Valid kill verdict passes through."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        monkeypatch.setattr(sr, "_call_review_llm",
                            lambda _: {"verdict": "kill", "rationale": "no signal", "refine_proposal": None})

        report = _make_report()
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        verdict = sr.judge_family(report, family, knowledge)
        assert verdict["verdict"] == "kill"
        assert verdict["refine_proposal"] is None


# ============================================================================
# 5. Prompt content
# ============================================================================

class TestPromptContent:
    def test_prompt_contains_flags(self, tmp_path, monkeypatch):
        """Prompt includes active flags from the report."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        # Capture the messages sent to the LLM
        captured_messages = []

        def _capture(messages):
            captured_messages.append(messages)
            return {"verdict": "keep", "rationale": "fine", "refine_proposal": None}

        monkeypatch.setattr(sr, "_call_review_llm", _capture)

        report = _make_report(
            family_key="sma_crossover|AAPL",
            flags={"cost_bound": True, "no_signal": False, "unstable": False,
                   "regime_dependent": False, "high_turnover": False},
        )
        family = _make_family("sma_crossover|AAPL")
        knowledge = _make_knowledge()

        sr.judge_family(report, family, knowledge)
        assert len(captured_messages) == 1
        user_content = captured_messages[0][1]["content"]  # user message

        # Flags should appear
        assert "cost_bound" in user_content

    def test_prompt_excludes_holdout(self, tmp_path, monkeypatch):
        """Prompt does not contain holdout measurement numbers (e.g. holdout_sharpe)."""
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
        # holdout measurement numbers (holdout_sharpe, holdout_*) must not be in prompt
        # (gate_failures.holdout count is fine — it's a counter, not a measurement)
        assert "holdout_sharpe" not in user_content.lower()

    def test_prompt_contains_no_arithmetic_instruction(self, tmp_path, monkeypatch):
        """Prompt tells LLM not to recompute arithmetic."""
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
        system_content = captured_messages[0][0]["content"]
        assert "do not recompute" in system_content.lower()

    def test_prompt_contains_family_aggregates(self, tmp_path, monkeypatch):
        """Prompt includes n_trials, best_val_sharpe, gate_failures, refine_count."""
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
        family = _make_family("sma_crossover|AAPL", n_trials=7, best_val_sharpe=0.9,
                              refine_count=2,
                              gate_failures={"validation": 1, "deflation": 0, "holdout": 2,
                                             "duplicate_cluster": 0, "exhausted_cluster": 0})
        knowledge = _make_knowledge()

        sr.judge_family(report, family, knowledge)
        user_content = captured_messages[0][1]["content"]

        assert "n_trials=7" in user_content
        assert "best_val_sharpe=0.9" in user_content
        assert "refine_count=2" in user_content
        assert "validation=1" in user_content
        assert "holdout=2" in user_content

    def test_prompt_includes_near_misses(self, tmp_path, monkeypatch):
        """Prompt includes last 5 near-miss entries for the family."""
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
        knowledge = _make_knowledge(
            near_misses=[
                _make_near_miss(strategy="sma_crossover", symbol="AAPL",
                                params={"fast_window": 10, "slow_window": 30},
                                val_sharpe=0.3, date="2026-07-10T00:00:00",
                                failed_gate="validation"),
                _make_near_miss(strategy="sma_crossover", symbol="AAPL",
                                params={"fast_window": 15, "slow_window": 40},
                                val_sharpe=0.5, date="2026-07-12T00:00:00",
                                failed_gate="holdout"),
            ],
        )

        sr.judge_family(report, family, knowledge)
        user_content = captured_messages[0][1]["content"]

        assert "near-miss" in user_content.lower()
        assert "fast_window" in user_content
        assert "slow_window" in user_content
        assert "holdout" in user_content  # "holdout" as a failed_gate value is expected


# ============================================================================
# 6. CLI --dry-run on judge path
# ============================================================================

class TestCLIDryRunJudge:
    def test_dry_run_zero_llm_calls(self, tmp_path, monkeypatch):
        """--dry-run makes zero LLM calls."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge

        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            rejected=[
                {
                    "hypothesis": {
                        "id": "test1",
                        "strategy": "sma_crossover",
                        "symbol": "AAPL",
                        "params": {"fast_window": 10, "slow_window": 50},
                    },
                    "evaluation": {
                        "sharpe_ratio": 0.5,
                        "gate_results": {"validation": False},
                    },
                    "verdict": "rejected",
                    "tested_at": "2026-07-15T00:00:00",
                },
            ],
        )
        save_knowledge(knowledge)

        # Track LLM calls
        llm_calls = []

        def _track_llm(*args):
            llm_calls.append(1)
            return {"verdict": "keep", "rationale": "ok", "refine_proposal": None}

        monkeypatch.setattr(sr, "_call_review_llm", _track_llm)

        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "strategy_review.py"),
             "--dry-run", "--max-families", "1"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "KNOWLEDGE_FILE": str(tmp_path / "knowledge.json")},
        )
        assert result.returncode == 0
        assert llm_calls == [], f"Expected 0 LLM calls, got {len(llm_calls)}"
        assert "DRY_RUN" in result.stdout

    def test_dry_run_no_writes(self, tmp_path, monkeypatch):
        """--dry-run writes nothing to knowledge.json or backlog.json."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge

        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            rejected=[
                {
                    "hypothesis": {
                        "id": "test1",
                        "strategy": "sma_crossover",
                        "symbol": "AAPL",
                        "params": {"fast_window": 10, "slow_window": 50},
                    },
                    "evaluation": {
                        "sharpe_ratio": 0.5,
                        "gate_results": {"validation": False},
                    },
                    "verdict": "rejected",
                    "tested_at": "2026-07-15T00:00:00",
                },
            ],
        )
        save_knowledge(knowledge)

        # Record initial knowledge mtime
        kpath = tmp_path / "knowledge.json"
        mtime_before = kpath.stat().st_mtime if kpath.exists() else 0

        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "strategy_review.py"),
             "--dry-run", "--max-families", "1"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "KNOWLEDGE_FILE": str(kpath)},
        )
        assert result.returncode == 0

        # Knowledge file should not be modified
        mtime_after = kpath.stat().st_mtime
        assert mtime_after == mtime_before, "knowledge.json was modified in dry-run"

    def test_no_judge_skips_llm(self, tmp_path, monkeypatch):
        """--no-judge skips LLM calls entirely."""
        # This test needs a runner URL but we'll mock diagnose_family to avoid HTTP
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge

        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            near_misses=[
                _make_near_miss(strategy="sma_crossover", symbol="AAPL",
                                date="2026-07-15T00:00:00"),
            ],
        )
        save_knowledge(knowledge)

        # Mock diagnose_family to return a valid report
        report = _make_report()
        monkeypatch.setattr(sr, "diagnose_family", lambda fk, kn, ru: (report, None))

        # Track LLM calls
        llm_calls = []

        def _track_llm(*args):
            llm_calls.append(1)
            return {"verdict": "keep", "rationale": "ok", "refine_proposal": None}

        monkeypatch.setattr(sr, "_call_review_llm", _track_llm)

        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "strategy_review.py"),
             "--no-judge", "--family", "sma_crossover|AAPL", "--max-families", "1"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ,
                 "KNOWLEDGE_FILE": str(tmp_path / "knowledge.json"),
                 "SANDBOX_RUNNER_URL": "http://fake:9000",
                 },
        )
        assert result.returncode == 0
        assert llm_calls == [], f"--no-judge should skip LLM calls, got {len(llm_calls)}"

    def test_cli_help_shows_no_judge(self):
        """--no-judge appears in help output."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "strategy_review.py"),
             "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--no-judge" in result.stdout


# ============================================================================
# 7. REVIEW_SUMMARY extended output
# ============================================================================

class TestReviewSummary:
    def test_summary_includes_verdict_counts(self, tmp_path, monkeypatch, capsys):
        """REVIEW_SUMMARY with --no-judge only shows diagnosed/errors/skipped."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge

        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            near_misses=[
                _make_near_miss(strategy="sma_crossover", symbol="AAPL",
                                date="2026-07-15T00:00:00"),
            ],
        )
        save_knowledge(knowledge)

        report = _make_report()
        monkeypatch.setattr(sr, "diagnose_family", lambda fk, kn, ru: (report, None))

        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "strategy_review.py"),
             "--no-judge", "--family", "sma_crossover|AAPL", "--max-families", "1"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ,
                 "KNOWLEDGE_FILE": str(tmp_path / "knowledge.json"),
                 "SANDBOX_RUNNER_URL": "http://fake:9000",
                 },
        )
        assert result.returncode == 0
        # --no-judge summary should NOT have kill/refine/keep/failopen
        assert "kill=" not in result.stdout
        assert "diagnosed=" in result.stdout

    def test_full_summary_includes_counts(self, tmp_path, monkeypatch):
        """REVIEW_SUMMARY with judge path includes kill/refine/keep/failopen."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge

        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            near_misses=[
                _make_near_miss(strategy="sma_crossover", symbol="AAPL",
                                date="2026-07-15T00:00:00"),
            ],
        )
        save_knowledge(knowledge)

        # Mock _post_json for diagnosis (needs evidence from near_misses)
        def _mock_post_json(url, payload, timeout=180):
            return {
                "out_of_sample": {"sharpe_ratio": 0.5, "turnover": 30.0},
                "cv": {"folds": [{"val": {"sharpe_ratio": 0.3}}, {"val": {"sharpe_ratio": 0.5}}, {"val": {"sharpe_ratio": 0.7}}]},
            }
        monkeypatch.setattr(sr, "_post_json", _mock_post_json)

        # Mock LLM to return keep
        monkeypatch.setattr(sr, "_call_review_llm",
                            lambda _: {"verdict": "keep", "rationale": "ok", "refine_proposal": None})

        # Call main() directly (in-process) to test summary output
        import io
        import sys as _sys
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://fake:9000")
        captured = io.StringIO()
        old_stdout = _sys.stdout
        _sys.stdout = captured

        try:
            # Simulate sys.argv
            monkeypatch.setattr(_sys, "argv", [
                "strategy_review.py", "--family", "sma_crossover|AAPL",
                "--max-families", "1",
            ])
            sr.main()
        finally:
            _sys.stdout = old_stdout

        output = captured.getvalue()
        assert "kill=0" in output
        assert "refine=0" in output
        assert "keep=1" in output
        assert "failopen=0" in output


# ============================================================================
# 8. Review-feedback fixes: audit trail, refine duplicate, downgrade suffix
# ============================================================================

class TestAuditTrail:
    def test_summary_records_applied_verdict_on_kill_downgrade(self, tmp_path, monkeypatch):
        """When kill is downgraded to keep, summary has llm_verdict=kill, applied_verdict=keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, n_trials=2)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {"verdict": "kill", "rationale": "should be killed", "refine_proposal": None}
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "keep"
        reloaded = load_knowledge()
        summary = reloaded["reviews"][0]
        assert summary["llm_verdict"] == "kill"
        assert summary["applied_verdict"] == "keep"
        assert "downgraded: insufficient evidence" in summary["rationale"]

    def test_summary_records_applied_verdict_on_cross_downgrade(self, tmp_path, monkeypatch):
        """When cross refine is downgraded, summary records llm_verdict=refine, applied_verdict=keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "manifest:xs_mom|universe:abc"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, family_type="cross")},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "try different params",
            "refine_proposal": {"params": {"universe_size": 10}, "change_summary": "Bigger"},
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "keep"
        reloaded = load_knowledge()
        summary = reloaded["reviews"][0]
        assert summary["llm_verdict"] == "refine"
        assert summary["applied_verdict"] == "keep"
        assert "downgraded: cross refine unavailable" in summary["rationale"]

    def test_summary_records_applied_verdict_on_refine_cap(self, tmp_path, monkeypatch):
        """When refine cap triggers, summary has llm_verdict=refine, applied_verdict=kill."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, refine_count=REFINE_CAP)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {
            "verdict": "refine",
            "rationale": "try again",
            "refine_proposal": {
                "params": {"fast_window": 50, "slow_window": 120},
                "change_summary": "final attempt",
            },
        }
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "kill"
        reloaded = load_knowledge()
        summary = reloaded["reviews"][0]
        assert summary["llm_verdict"] == "refine"
        assert summary["applied_verdict"] == "kill"
        assert summary["rationale"] == "refine cap exhausted"


class TestRefineDuplicate:
    def test_duplicate_refine_does_not_increment_count(self, tmp_path, monkeypatch):
        """Duplicate refine entry → refine_count unchanged, applied_verdict=keep."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, refine_count=1)},
        )
        save_knowledge(knowledge)

        backlog_path = tmp_path / "backlog.json"
        backlog = Backlog(str(backlog_path))

        # First refine: should succeed
        verdict = {
            "verdict": "refine",
            "rationale": "cost bound",
            "refine_proposal": {
                "params": {"fast_window": 30, "slow_window": 80},
                "change_summary": "Longer windows",
            },
        }
        result1 = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)
        assert result1 == "refine"

        # Second refine with same params: duplicate → should be rejected
        result2 = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        assert result2 == "keep"
        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        # refine_count should NOT have incremented from 2 to 3
        assert fam["refine_count"] == 2
        assert fam["lifecycle"] == FamilyLifecycle.REFINING  # unchanged from first
        # backlog should still have only 1 entry
        bl2 = Backlog(str(backlog_path))
        data = bl2.load()
        assert len(data["entries"]) == 1

    def test_duplicate_refine_prints_review_refine_duplicate(self, tmp_path, monkeypatch, capsys):
        """Duplicate refine emits REVIEW_REFINE_DUPLICATE line."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge
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
            "rationale": "cost bound",
            "refine_proposal": {
                "params": {"fast_window": 30, "slow_window": 80},
                "change_summary": "Longer windows",
            },
        }
        # First: accepted
        sr.apply_verdict(family_key, verdict, knowledge, backlog)

        # Second: duplicate
        sr.apply_verdict(family_key, verdict, knowledge, backlog)

        captured = capsys.readouterr()
        assert "REVIEW_REFINE_DUPLICATE sma_crossover|AAPL" in captured.out

    def test_duplicate_refine_records_keep_in_summary(self, tmp_path, monkeypatch):
        """Duplicate refine records applied_verdict=keep with suffix."""
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
            "rationale": "cost bound",
            "refine_proposal": {
                "params": {"fast_window": 30, "slow_window": 80},
                "change_summary": "Longer windows",
            },
        }
        # First: accepted
        sr.apply_verdict(family_key, verdict, knowledge, backlog)

        # Clear reviews for clean check
        knowledge["reviews"] = []
        save_knowledge(knowledge)

        # Second: duplicate
        sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        reloaded = load_knowledge()
        summary = reloaded["reviews"][0]
        assert summary["llm_verdict"] == "refine"
        assert summary["applied_verdict"] == "keep"
        assert "refine duplicate" in summary["rationale"]


class TestDowngradeSuffix:
    def test_kill_downgrade_print_contains_suffix(self, tmp_path, monkeypatch, capsys):
        """Kill downgraded to keep prints the suffixed rationale."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, n_trials=2)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {"verdict": "kill", "rationale": "bad", "refine_proposal": None}
        sr.apply_verdict(family_key, verdict, knowledge, backlog)

        captured = capsys.readouterr()
        assert "downgraded: insufficient evidence" in captured.out

    def test_downgrade_suffix_in_summary_rationale(self, tmp_path, monkeypatch):
        """Summary rationale includes the downgrade suffix when kill is downgraded."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        family_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={family_key: _make_family(family_key, n_trials=2)},
        )
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = {"verdict": "kill", "rationale": "bad", "refine_proposal": None}
        sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        reloaded = load_knowledge()
        summary = reloaded["reviews"][0]
        assert summary["rationale"] == "bad (downgraded: insufficient evidence)"
