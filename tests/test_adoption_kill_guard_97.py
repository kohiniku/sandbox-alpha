"""
Regression tests for issue #97: the review loop could kill a family whose
member is currently adopted.

Adoption and the review/kill lifecycle were two independent state machines
over the same families with no shared contract. A later rejected trial of an
adopted family re-admitted it to review, and a kill verdict applied
lifecycle=KILLED with no adoption check — permanently poisoning the adopted
parameter cluster via KILLED_SKIP in the validation loop.

Fix:
- adopted records now carry a ``family_key`` join key (stamped at creation,
  backfilled for legacy records in load_knowledge);
- select_candidates skips families that have an adopted member;
- apply_verdict's kill branch (LLM kill and refine-cap auto-kill) downgrades
  to keep with a REVIEW_KILL_BLOCKED_ADOPTED marker when the family has an
  adopted member.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_loop import FamilyLifecycle, load_knowledge, save_knowledge
from backlog import Backlog
import strategy_review as sr


def _make_knowledge(families=None, adopted=None, rejected=None):
    return {
        "families": families or {},
        "adopted": adopted or [],
        "rejected": rejected or [],
        "near_misses": [],
        "near_misses_cross": [],
        "review_state": {},
        "reviews": [],
        "tested_combinations": [],
        "errors": [],
        "iterations": 0,
        "tested": [],
        "superseded": [],
    }


def _single_family(n_trials=3, best_val_sharpe=1.2, lifecycle=None):
    fam = {
        "family_type": "single",
        "n_trials": n_trials,
        "best_val_sharpe": best_val_sharpe,
        "best_params": {"rsi_window": 14, "oversold": 30, "overbought": 70},
        "refine_count": 0,
        "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                          "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": "2026-08-09T17:10:28",
    }
    if lifecycle:
        fam["lifecycle"] = lifecycle
    return fam


def _adopted_entry(strategy="rsi", symbol="TLT", family_key=None):
    entry = {
        "hypothesis": {"strategy": strategy, "symbol": symbol, "params": {}},
        "verdict": "adopted",
        "tested_at": "2026-08-09T17:10:28",
        "family": None,
    }
    if family_key:
        entry["family_key"] = family_key
    return entry


def _rejected_entry(strategy="rsi", symbol="TLT", tested_at="2026-08-10T02:44:00"):
    return {
        "hypothesis": {"strategy": strategy, "symbol": symbol,
                       "params": {"rsi_window": 27, "oversold": 20, "overbought": 80}},
        "verdict": "rejected",
        "tested_at": tested_at,
    }


def _kill_verdict(rationale="no signal"):
    return {"verdict": "kill", "rationale": rationale}


# ============================================================================
# select_candidates: adopted families must not be re-admitted to review
# ============================================================================

class TestSelectCandidatesAdoptionGuard:
    def test_adopted_family_excluded_legacy_record(self):
        """Legacy adopted record (no family_key) still excludes the family."""
        knowledge = _make_knowledge(
            families={"rsi|TLT": _single_family()},
            adopted=[_adopted_entry()],  # no family_key → derived from hypothesis
            rejected=[_rejected_entry()],
        )
        assert sr.select_candidates(knowledge, "2026-08-10T03:00:00") == []

    def test_adopted_family_excluded_with_family_key(self):
        """New-format adopted record (family_key stamped) excludes the family."""
        knowledge = _make_knowledge(
            families={"rsi|TLT": _single_family()},
            adopted=[_adopted_entry(family_key="rsi|TLT")],
            rejected=[_rejected_entry()],
        )
        assert sr.select_candidates(knowledge, "2026-08-10T03:00:00") == []

    def test_non_adopted_family_still_selected(self):
        """The guard must not suppress legitimate review of non-adopted families."""
        knowledge = _make_knowledge(
            families={
                "rsi|TLT": _single_family(),
                "sma_crossover|GLD": _single_family(n_trials=2, best_val_sharpe=0.9),
            },
            adopted=[_adopted_entry()],  # only rsi|TLT is adopted
            rejected=[_rejected_entry(), _rejected_entry("sma_crossover", "GLD")],
        )
        candidates = sr.select_candidates(knowledge, "2026-08-10T03:00:00")
        assert "sma_crossover|GLD" in candidates
        assert "rsi|TLT" not in candidates


# ============================================================================
# apply_verdict: kill must be blocked for adopted families
# ============================================================================

class TestApplyVerdictAdoptionGuard:
    def test_kill_blocked_for_adopted_family(self):
        """Kill verdict on adopted family → keep + REVIEW_KILL_BLOCKED_ADOPTED marker."""
        knowledge = _make_knowledge(
            families={"rsi|TLT": _single_family(n_trials=3)},
            adopted=[_adopted_entry(family_key="rsi|TLT")],
        )
        result = sr.apply_verdict("rsi|TLT", _kill_verdict(), knowledge, Backlog("/tmp/b97.json"))
        assert result == "keep"
        fam = knowledge["families"]["rsi|TLT"]
        assert fam.get("lifecycle") != FamilyLifecycle.KILLED
        assert fam.get("kill_blocked_by_adoption")  # marker recorded

    def test_kill_blocked_for_adopted_family_legacy_adopted(self):
        """Legacy adopted record (no family_key) also blocks the kill."""
        knowledge = _make_knowledge(
            families={"rsi|TLT": _single_family(n_trials=3)},
            adopted=[_adopted_entry()],  # no family_key
        )
        result = sr.apply_verdict("rsi|TLT", _kill_verdict(), knowledge, Backlog("/tmp/b97b.json"))
        assert result == "keep"
        assert knowledge["families"]["rsi|TLT"].get("lifecycle") != FamilyLifecycle.KILLED

    def test_kill_still_applies_for_non_adopted_family(self):
        """Non-adopted family with enough trials still gets killed."""
        knowledge = _make_knowledge(
            families={"mean_reversion|MSFT": _single_family(n_trials=3)},
        )
        result = sr.apply_verdict("mean_reversion|MSFT", _kill_verdict(), knowledge, Backlog("/tmp/b97c.json"))
        assert result == "kill"
        assert knowledge["families"]["mean_reversion|MSFT"]["lifecycle"] == FamilyLifecycle.KILLED

    def test_refine_cap_autokill_blocked_for_adopted_cross_family(self):
        """Refine-cap auto-kill is also blocked for an adopted cross family."""
        family_key = "manifest:xs_mom|universe:abc123"
        fam = _single_family(n_trials=1, lifecycle=FamilyLifecycle.REFINING)
        fam["family_type"] = "cross"
        fam["refine_count"] = sr.REFINE_CAP
        fam["last_manifest_spec"] = {"name": "xs_mom"}
        knowledge = _make_knowledge(
            families={family_key: fam},
            adopted=[_adopted_entry(strategy="manifest:xs_mom", symbol="universe:abc123",
                                    family_key=family_key)],
        )
        verdict = {"verdict": "refine", "rationale": "cap hit",
                   "refine_proposal": {"params": {}, "change_summary": "x"}}
        result = sr.apply_verdict(family_key, verdict, knowledge, Backlog("/tmp/b97d.json"))
        assert result == "keep"
        assert knowledge["families"][family_key].get("lifecycle") != FamilyLifecycle.KILLED
        assert knowledge["families"][family_key].get("kill_blocked_by_adoption")


# ============================================================================
# load_knowledge: legacy adopted records get family_key backfilled
# ============================================================================

class TestAdoptedFamilyKeyBackfill:
    def test_backfill_on_load(self, tmp_path, monkeypatch):
        kf = tmp_path / "knowledge.json"
        kf.write_text(json.dumps(_make_knowledge(adopted=[_adopted_entry()])))
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
        data = load_knowledge()
        entry = data["adopted"][0]
        assert entry["family_key"] == "rsi|TLT"

    def test_stamped_family_key_preserved_on_load(self, tmp_path, monkeypatch):
        kf = tmp_path / "knowledge.json"
        kf.write_text(json.dumps(_make_knowledge(
            adopted=[_adopted_entry(family_key="rsi|TLT")])))
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
        data = load_knowledge()
        assert data["adopted"][0]["family_key"] == "rsi|TLT"

    def test_cross_adopted_family_key_derivation(self, tmp_path, monkeypatch):
        kf = tmp_path / "knowledge.json"
        kf.write_text(json.dumps(_make_knowledge(
            adopted=[_adopted_entry(strategy="manifest:xs_mom", symbol="universe:abc123")])))
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
        data = load_knowledge()
        assert data["adopted"][0]["family_key"] == "manifest:xs_mom|universe:abc123"
