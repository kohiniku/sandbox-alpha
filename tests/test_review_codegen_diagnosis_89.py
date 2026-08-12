"""
Regression tests for issue #89: review loop permanently re-selects codegen
families it cannot diagnose.

codegen families (execution_mode=code proposals) carry no params and their
code payload is not persisted in knowledge.json, so the param-based single
diagnosis path (_diagnose_single_family) raises ValueError
("no params recoverable from evidence") on EVERY attempt. Because the
validation loop keeps feeding them new evidence, the #77 error backoff
(last_diag_error_at >= newest_failure) never holds and the family is
re-selected every review run — REVIEW_DIAG_ERROR every cycle, and the whole
class stuck candidate forever with no verdicts.

Fix (this PR):
1. Codegen families get a baseline-only diagnosis path (mirroring cross
   families): zero HTTP calls, baseline from the recorded evaluation /
   near-miss val_sharpe, so the LLM judge can finally apply verdicts.
2. apply_verdict downgrades refine → keep for codegen families (no params,
   no runner schema to refine against), same pattern as cross families.
3. Diagnosis errors are classified permanent vs transient: permanent-class
   errors (e.g. "no params recoverable from evidence") mark the family
   ineligible regardless of new evidence, so ANY future permanent class
   stops burning review capacity. A successful diagnosis clears the marker.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_constants import FamilyLifecycle
from backlog import Backlog, BacklogStatus

# Module under test (import after path setup)
import strategy_review as sr


# ============================================================================
# Helpers
# ============================================================================

def _make_family(key, family_type="single", lifecycle=FamilyLifecycle.CANDIDATE,
                 best_val_sharpe=0.5, n_trials=1, refine_count=0):
    return {
        "n_trials": n_trials,
        "best_val_sharpe": best_val_sharpe,
        "best_params": {},
        "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                          "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": "2026-01-01T00:00:00",
        "family_type": family_type,
        "lifecycle": lifecycle,
        "refine_count": refine_count,
        "kill_reason": "",
    }


def _make_codegen_near_miss(symbol="SPY", val_sharpe=0.4,
                            date="2026-08-07T21:11:00"):
    """Near-miss entry shape produced by autonomous_loop._classify_near_miss
    for a codegen (execution_mode=code) backlog entry — params is empty."""
    return {
        "id": "bl_codegen1",
        "strategy": "codegen",
        "symbol": symbol,
        "params": {},
        "val_sharpe": val_sharpe,
        "deflated_threshold": 0.8,
        "holdout_sharpe": -0.1,
        "failed_gate": "holdout",
        "date": date,
    }


def _make_codegen_rejected(symbol="SPY", val_sharpe=0.35,
                           tested_at="2026-08-07T21:11:00"):
    """Rejected record shape produced by save_result for a codegen entry."""
    return {
        "hypothesis": {
            "id": "bl_codegen2",
            "strategy": "codegen",
            "symbol": symbol,
            "params": {},
            "description": f"Backlog code: xyz on {symbol}",
            "generated_at": "2026-08-07T20:00:00",
        },
        "backtest_result": {
            "out_of_sample": {"sharpe_ratio": val_sharpe},
        },
        "evaluation": {
            "sharpe_ratio": val_sharpe,
            "gate_results": {"validation": False},
        },
        "verdict": "rejected",
        "tested_at": tested_at,
    }


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
# 1. Codegen diagnosis: baseline-only, zero HTTP calls, no error
# ============================================================================

class TestCodegenDiagnosis:
    def test_codegen_near_miss_diagnoses_baseline_only(self):
        """A codegen family with near-miss evidence diagnoses successfully
        (baseline from the recorded near-miss val_sharpe) instead of raising
        'no params recoverable from evidence'."""
        knowledge = _make_knowledge(
            families={"codegen|SPY": _make_family("codegen|SPY")},
            near_misses=[_make_codegen_near_miss(val_sharpe=0.4)],
        )
        report, error = sr.diagnose_family("codegen|SPY", knowledge, "http://fake:9000")
        assert error is None, f"codegen family must not error: {error}"
        assert report["diagnosis_scope"] == "baseline_only"
        assert report["baseline"]["val_sharpe"] == 0.4
        assert report["cost_free"] is None
        assert report["flags"] == []

    def test_codegen_rejected_uses_recorded_evaluation(self):
        """A codegen family with a rejected record uses the recorded
        evaluation.sharpe_ratio as baseline (not the near-miss fallback)."""
        knowledge = _make_knowledge(
            families={"codegen|AMD": _make_family("codegen|AMD")},
            rejected=[_make_codegen_rejected(symbol="AMD", val_sharpe=0.35)],
        )
        report, error = sr.diagnose_family("codegen|AMD", knowledge, "http://fake:9000")
        assert error is None
        assert report["baseline"]["source"] == "recorded_evaluation"
        assert report["baseline"]["val_sharpe"] == 0.35

    def test_codegen_diagnosis_makes_no_runner_calls(self, monkeypatch):
        """Baseline-only means zero HTTP calls to the runner."""
        calls = []

        def _boom(url, payload, timeout=180):
            calls.append(url)
            raise AssertionError(f"unexpected runner call: {url}")

        monkeypatch.setattr(sr, "_post_json", _boom)
        knowledge = _make_knowledge(
            families={"codegen|SPY": _make_family("codegen|SPY")},
            near_misses=[_make_codegen_near_miss()],
        )
        report, error = sr.diagnose_family("codegen|SPY", knowledge, "http://fake:9000")
        assert error is None
        assert calls == []

    def test_param_family_still_diagnoses_via_runner(self, monkeypatch):
        """Invariant guard: regular param families keep the HTTP diagnosis
        path — the codegen branch must not swallow them."""
        monkeypatch.setattr(sr, "_post_json",
                            lambda url, payload, timeout=180: {
                                "out_of_sample": {"sharpe_ratio": 0.6,
                                                  "turnover": 30.0},
                                "folds": [0.5, 0.6, 0.7],
                            })
        knowledge = _make_knowledge(
            families={"sma_crossover|AAPL": _make_family("sma_crossover|AAPL")},
            rejected=[{
                "hypothesis": {"id": "t1", "strategy": "sma_crossover",
                               "symbol": "AAPL",
                               "params": {"fast_window": 10, "slow_window": 50}},
                "evaluation": {"sharpe_ratio": 0.6,
                               "gate_results": {"validation": False}},
                "tested_at": "2026-08-01T00:00:00",
            }],
        )
        report, error = sr.diagnose_family("sma_crossover|AAPL", knowledge,
                                           "http://fake:9000")
        assert error is None
        assert report["family_type"] == "single"
        # Real param diagnosis: flags dict (not the baseline-only [] list),
        # cost_free measured, no baseline_only scope marker.
        assert report.get("diagnosis_scope") != "baseline_only"
        assert isinstance(report["flags"], dict)
        assert report["cost_free"] is not None


# ============================================================================
# 2. apply_verdict: refine downgraded to keep for codegen families
# ============================================================================

class TestCodegenRefineDowngrade:
    def _run_apply(self, verdict_dict, family=None, backlog_path=None):
        family = family or _make_family("codegen|SPY", n_trials=5)
        knowledge = _make_knowledge(families={"codegen|SPY": family})
        bl = Backlog(str(backlog_path or "/tmp/nonexistent-backlog.json"))
        result = sr.apply_verdict("codegen|SPY", verdict_dict, knowledge, bl)
        return result, family, knowledge, bl

    def test_codegen_refine_downgraded_to_keep(self, tmp_path):
        """refine on a codegen family must be downgraded to keep (no params /
        no runner schema to refine against) — never create a param backlog
        entry for strategy='codegen'."""
        verdict = {
            "verdict": "refine",
            "rationale": "改善の余地あり",
            "refine_proposal": {"params": {"window": 20}, "change_summary": "窓を広げる"},
        }
        result, family, knowledge, bl = self._run_apply(
            verdict, backlog_path=tmp_path / "backlog.json")
        assert result == "keep"
        assert family.get("lifecycle") != FamilyLifecycle.REFINING
        assert family.get("refine_count", 0) == 0
        # Disk round-trip: no param entry may have been queued
        entries = bl.load()["entries"]
        assert entries == []
        # The rationale must say why
        summary = knowledge["reviews"][-1]
        assert summary["applied_verdict"] == "keep"
        assert "codegenファミリーはrefine不可" in summary["rationale"]

    def test_codegen_kill_still_gated_by_trials(self, tmp_path):
        """Invariant guard: codegen kill verdicts still respect the
        MIN_TRIALS_FOR_KILL gate — the refine downgrade must not loosen it."""
        bl = Backlog(str(tmp_path / "backlog.json"))
        family = _make_family("codegen|SPY", n_trials=1)  # below threshold
        knowledge = _make_knowledge(families={"codegen|SPY": family})
        result = sr.apply_verdict("codegen|SPY",
                                  {"verdict": "kill", "rationale": "全foldマイナス",
                                   "refine_proposal": None},
                                  knowledge, bl)
        assert result == "keep"
        assert family.get("lifecycle") != FamilyLifecycle.KILLED

    def test_codegen_kill_applies_at_sufficient_trials(self, tmp_path):
        """codegen family with enough trials can still be killed — the point
        of the fix is that verdicts become REACHABLE, not that codegen
        families are immortal."""
        bl = Backlog(str(tmp_path / "backlog.json"))
        family = _make_family("codegen|SPY", n_trials=7)
        knowledge = _make_knowledge(families={"codegen|SPY": family})
        result = sr.apply_verdict("codegen|SPY",
                                  {"verdict": "kill", "rationale": "全foldマイナス",
                                   "refine_proposal": None},
                                  knowledge, bl)
        assert result == "kill"
        assert family.get("lifecycle") == FamilyLifecycle.KILLED


# ============================================================================
# 3. Permanent vs transient diagnosis errors
# ============================================================================

class TestPermanentDiagError:
    def test_permanent_error_family_not_reselected_with_new_evidence(self):
        """A family whose last diagnosis error is a PERMANENT class error
        (e.g. codegen 'no params recoverable') must NOT be re-selected even
        when new evidence arrives — the #77 backoff must actually hold."""
        key = "codegen|SPY"
        knowledge = _make_knowledge(
            families={key: _make_family(key)},
            near_misses=[_make_codegen_near_miss(date="2026-08-08T00:00:00")],
            review_state={"reviewed": {
                key: {
                    "last_diag_error_at": "2026-08-07T21:11:00",
                    "diag_error_permanent": True,
                }
            }},
        )
        assert sr._family_has_new_evidence(key, knowledge) is False
        candidates = sr.select_candidates(knowledge, "2026-08-08T01:00:00",
                                          max_families=3)
        assert key not in candidates

    def test_transient_error_still_re_admitted_on_new_evidence(self):
        """Invariant guard: a TRANSIENT diagnosis error (no permanent marker)
        must keep the #77 behavior — new evidence re-admits the family."""
        key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={key: _make_family(key)},
            rejected=[{
                "hypothesis": {"id": "t1", "strategy": "sma_crossover",
                               "symbol": "AAPL",
                               "params": {"fast_window": 10, "slow_window": 50}},
                "evaluation": {"sharpe_ratio": 0.5,
                               "gate_results": {"validation": False}},
                "tested_at": "2026-08-08T00:00:00",
            }],
            review_state={"reviewed": {
                key: {"last_diag_error_at": "2026-08-07T21:11:00"},
            }},
        )
        assert sr._family_has_new_evidence(key, knowledge) is True
        candidates = sr.select_candidates(knowledge, "2026-08-08T01:00:00",
                                          max_families=3)
        assert key in candidates

    def test_record_diag_error_marks_permanent_class(self):
        """_record_diag_error stamps diag_error_permanent for permanent-class
        error messages and leaves transient errors unmarked."""
        knowledge = _make_knowledge()
        sr._record_diag_error(knowledge, "codegen|SPY", "2026-08-08T00:00:00",
                              error_msg="no params recoverable from evidence for codegen|SPY")
        reviewed = knowledge["review_state"]["reviewed"]["codegen|SPY"]
        assert reviewed["diag_error_permanent"] is True

        knowledge2 = _make_knowledge()
        sr._record_diag_error(knowledge2, "rsi|AAPL", "2026-08-08T00:00:00",
                              error_msg="Runner HTTP 503: boom")
        reviewed2 = knowledge2["review_state"]["reviewed"]["rsi|AAPL"]
        assert reviewed2.get("diag_error_permanent") is None

    def test_successful_diagnosis_clears_permanent_marker(self):
        """Invariant guard: a successful diagnosis clears the permanent
        marker and stale error timestamp, so a later codegen diagnosis path
        can bring the family back into the pool."""
        key = "codegen|SPY"
        knowledge = _make_knowledge(
            review_state={"reviewed": {
                key: {"last_diag_error_at": "2026-08-07T21:11:00",
                      "diag_error_permanent": True},
            }},
        )
        sr._update_review_state(knowledge, key, "2026-08-08T00:00:00")
        reviewed = knowledge["review_state"]["reviewed"][key]
        assert reviewed.get("diag_error_permanent") is None
        assert reviewed.get("last_diag_error_at") is None
        assert reviewed["last_diagnosed_at"] == "2026-08-08T00:00:00"
