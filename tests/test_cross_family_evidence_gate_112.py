"""
Regression tests for issue #112: cross-sectional families permanently locked
out of the review candidate pool by the evidence gate.

Cross families are structurally one-shot — they get exactly one trial, so
after diagnosis ``newest_evidence < last_diagnosed_at`` permanently. The
evidence gate must bypass this for cross families that have a persisted
manifest spec (``last_manifest_spec``) and haven't exhausted refine attempts.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_constants import FamilyLifecycle, REFINE_CAP
import strategy_review as sr


# ---------------------------------------------------------------------------
# Helpers (mirrors test_strategy_review.py patterns)
# ---------------------------------------------------------------------------

def _make_family(key, family_type="single", lifecycle=FamilyLifecycle.CANDIDATE,
                 best_val_sharpe=0.5, refine_count=0,
                 last_manifest_spec=None):
    fam = {
        "n_trials": 1,
        "best_val_sharpe": best_val_sharpe,
        "best_params": {},
        "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                          "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": "2026-01-01T00:00:00",
        "family_type": family_type,
        "lifecycle": lifecycle,
        "refine_count": refine_count,
    }
    if last_manifest_spec is not None:
        fam["last_manifest_spec"] = last_manifest_spec
    return fam


def _make_near_miss_cross(strategy="manifest:xs_mom", symbol="universe:abc",
                          date="2026-08-01T06:00:00", val_sharpe=3.0):
    return {
        "id": "nm_cross_1",
        "strategy": strategy,
        "symbol": symbol,
        "params": {"universe_size": 100},
        "val_sharpe": val_sharpe,
        "deflated_threshold": 0.8,
        "holdout_sharpe": -0.1,
        "failed_gate": "holdout",
        "date": date,
    }


def _make_knowledge(families=None, near_misses_cross=None, review_state=None):
    return {
        "families": families or {},
        "rejected": [],
        "near_misses": [],
        "near_misses_cross": near_misses_cross or [],
        "review_state": review_state or {},
        "reviews": [],
        "adopted": [],
        "tested_combinations": [],
        "errors": [],
    }


CROSS_KEY = "manifest:xs_mom|universe:abc"
MANIFEST_SPEC = {"name": "xs_mom", "code_b64": "xxx", "universe_size": 100,
                 "evaluator": {"extras": {}}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCrossFamilyEvidenceGate:
    """Issue #112: cross families must be re-selectable after diagnosis."""

    def test_cross_family_eligible_after_diagnosis_with_manifest_spec(self):
        """After diagnosis, a cross family with last_manifest_spec and
        refine_count < REFINE_CAP should still be eligible via the bypass."""
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=3.0,
                    last_manifest_spec=MANIFEST_SPEC,
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-01T06:00:00"),
            ],
            review_state={
                "last_review_at": "2026-08-10T00:00:00",
                "reviewed": {
                    CROSS_KEY: {
                        # Diagnosis happened AFTER the evidence timestamp
                        "last_diagnosed_at": "2026-08-02T07:00:00",
                    },
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert CROSS_KEY in candidates, (
            "Cross family with manifest_spec should be eligible via bypass "
            "even after diagnosis consumed its only evidence"
        )

    def test_cross_family_blocked_when_no_manifest_spec(self):
        """Cross family without last_manifest_spec cannot be refined — bypass
        must NOT apply (refine would fail at apply_verdict anyway)."""
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=3.0,
                    # No last_manifest_spec
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-01T06:00:00"),
            ],
            review_state={
                "reviewed": {
                    CROSS_KEY: {
                        "last_diagnosed_at": "2026-08-02T07:00:00",
                    },
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert CROSS_KEY not in candidates

    def test_cross_family_blocked_at_refine_cap(self):
        """Cross family at REFINE_CAP should not get bypass — no more refines
        possible, so re-diagnosis is pointless."""
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=3.0,
                    last_manifest_spec=MANIFEST_SPEC,
                    refine_count=REFINE_CAP,
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-01T06:00:00"),
            ],
            review_state={
                "reviewed": {
                    CROSS_KEY: {
                        "last_diagnosed_at": "2026-08-02T07:00:00",
                    },
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert CROSS_KEY not in candidates

    def test_bypass_does_not_loop_at_same_refine_count(self):
        """After one bypass diagnosis, the family should not be re-selected
        at the same refine_count (prevents #89-style infinite loop)."""
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=3.0,
                    last_manifest_spec=MANIFEST_SPEC,
                    refine_count=0,
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-01T06:00:00"),
            ],
            review_state={
                "reviewed": {
                    CROSS_KEY: {
                        "last_diagnosed_at": "2026-08-02T07:00:00",
                        # First bypass diagnosis already happened at rc=0
                        "last_bypass_refine_count": 0,
                    },
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert CROSS_KEY not in candidates, (
            "Bypass should not re-select at the same refine_count"
        )

    def test_bypass_available_after_refine_count_changes(self):
        """After a refine (refine_count incremented), the bypass becomes
        available again for the next diagnosis cycle."""
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=3.0,
                    last_manifest_spec=MANIFEST_SPEC,
                    refine_count=1,  # Was refined once
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-01T06:00:00"),
            ],
            review_state={
                "reviewed": {
                    CROSS_KEY: {
                        "last_diagnosed_at": "2026-08-05T07:00:00",
                        # Previous bypass was at rc=0, now rc=1
                        "last_bypass_refine_count": 0,
                    },
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert CROSS_KEY in candidates, (
            "Bypass should re-open when refine_count changed since last bypass"
        )

    def test_bypass_stamps_marker_on_selection(self):
        """select_candidates should stamp last_bypass_refine_count when a
        cross family is selected via the bypass path."""
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=3.0,
                    last_manifest_spec=MANIFEST_SPEC,
                    refine_count=0,
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-01T06:00:00"),
            ],
            review_state={
                "reviewed": {
                    CROSS_KEY: {
                        "last_diagnosed_at": "2026-08-02T07:00:00",
                    },
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert CROSS_KEY in candidates
        # Verify the bypass marker was stamped
        reviewed = knowledge["review_state"]["reviewed"][CROSS_KEY]
        assert reviewed.get("last_bypass_refine_count") == 0

    def test_normal_evidence_path_still_works_for_cross(self):
        """If genuinely new evidence exists (newest > last_diagnosed), the
        normal path should select the family — bypass is not needed."""
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=3.0,
                    last_manifest_spec=MANIFEST_SPEC,
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-10T06:00:00"),
            ],
            review_state={
                "reviewed": {
                    CROSS_KEY: {
                        "last_diagnosed_at": "2026-08-02T07:00:00",
                    },
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert CROSS_KEY in candidates
        # Bypass marker should NOT be stamped when normal path works
        reviewed = knowledge["review_state"]["reviewed"][CROSS_KEY]
        assert "last_bypass_refine_count" not in reviewed

    def test_single_families_unaffected_by_bypass(self):
        """Single families must NOT use the cross-family bypass."""
        single_key = "sma_crossover|AAPL"
        knowledge = _make_knowledge(
            families={
                single_key: _make_family(
                    single_key, family_type="single",
                    best_val_sharpe=0.8,
                    last_manifest_spec=MANIFEST_SPEC,  # irrelevant for single
                ),
            },
            near_misses_cross=[],
            review_state={
                "near_misses": [],
                "reviewed": {
                    single_key: {
                        "last_diagnosed_at": "2026-08-10T07:00:00",
                    },
                },
            },
        )
        # No rejected or near_miss evidence newer than diagnosis
        knowledge["rejected"] = [{
            "hypothesis": {"strategy": "sma_crossover", "symbol": "AAPL",
                          "params": {"fast": 10}},
            "tested_at": "2026-08-01T00:00:00",
        }]
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert single_key not in candidates

    def test_cross_family_diag_error_permanent_blocks_bypass(self):
        """Permanent diagnosis error must block even the bypass path."""
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=3.0,
                    last_manifest_spec=MANIFEST_SPEC,
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-01T06:00:00"),
            ],
            review_state={
                "reviewed": {
                    CROSS_KEY: {
                        "last_diagnosed_at": "2026-08-02T07:00:00",
                        "diag_error_permanent": True,
                    },
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-08-15T00:00:00",
                                          max_families=3)
        assert CROSS_KEY not in candidates

    def test_high_sharpe_cross_prioritized_over_low_sharpe_single(self):
        """Cross families with high val_sharpe should be selected over
        low-sharpe single families when both are eligible (integration)."""
        single_key = "rsi|QCOM"
        knowledge = _make_knowledge(
            families={
                CROSS_KEY: _make_family(
                    CROSS_KEY, family_type="cross",
                    best_val_sharpe=4.3,
                    last_manifest_spec=MANIFEST_SPEC,
                ),
                single_key: _make_family(
                    single_key, family_type="single",
                    best_val_sharpe=1.3,
                ),
            },
            near_misses_cross=[
                _make_near_miss_cross(date="2026-08-01T06:00:00", val_sharpe=4.3),
            ],
            review_state={
                "reviewed": {
                    CROSS_KEY: {
                        "last_diagnosed_at": "2026-08-02T07:00:00",
                    },
                    single_key: {
                        "last_diagnosed_at": "2026-08-15T00:00:00",
                    },
                },
            },
        )
        # Add fresh evidence for the single family so it's eligible normally
        knowledge["rejected"] = [{
            "hypothesis": {"strategy": "rsi", "symbol": "QCOM",
                          "params": {"period": 14}},
            "tested_at": "2026-08-16T00:00:00",
        }]
        candidates = sr.select_candidates(knowledge, "2026-08-17T00:00:00",
                                          max_families=3)
        assert CROSS_KEY in candidates
        assert single_key in candidates
        # Cross family (sharpe 4.3) should come before single (sharpe 1.3)
        assert candidates.index(CROSS_KEY) < candidates.index(single_key)
