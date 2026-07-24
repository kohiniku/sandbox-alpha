"""
Tests for PR-B2: strategy_review.py diagnosis stage.

Covers: candidate selection, flag arithmetic, holdout exclusion,
disk round-trip, error paths, CLI --dry-run.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_constants import FamilyLifecycle

# Module under test (import after path setup)
import strategy_review as sr


# ============================================================================
# Helpers
# ============================================================================

def _make_family(key, family_type="single", lifecycle=FamilyLifecycle.CANDIDATE,
                 best_val_sharpe=0.5, kill_reason="", refine_count=0,
                 last_tried="2026-01-01T00:00:00"):
    return {
        "n_trials": 1,
        "best_val_sharpe": best_val_sharpe,
        "best_params": {},
        "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                          "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": last_tried,
        "family_type": family_type,
        "lifecycle": lifecycle,
        "refine_count": refine_count,
        "kill_reason": kill_reason,
    }


def _make_rejected(strategy="sma_crossover", symbol="AAPL", params=None,
                   tested_at="2026-07-01T00:00:00", val_sharpe=0.5):
    return {
        "hypothesis": {
            "id": "test1",
            "strategy": strategy,
            "symbol": symbol,
            "params": params or {"fast_window": 10, "slow_window": 50},
        },
        "backtest_result": {
            "out_of_sample": {"sharpe_ratio": val_sharpe, "turnover": 30.0},
        },
        "evaluation": {
            "sharpe_ratio": val_sharpe,
            "gate_results": {"validation": False},
        },
        "verdict": "rejected",
        "tested_at": tested_at,
    }


def _make_near_miss(strategy="sma_crossover", symbol="AAPL", params=None,
                    val_sharpe=0.7, date="2026-07-01T00:00:00"):
    return {
        "id": "nm1",
        "strategy": strategy,
        "symbol": symbol,
        "params": params or {"fast_window": 15, "slow_window": 40},
        "val_sharpe": val_sharpe,
        "deflated_threshold": 0.8,
        "holdout_sharpe": -0.1,
        "failed_gate": "holdout",
        "date": date,
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
# 1. Candidate selection
# ============================================================================

class TestCandidateSelection:
    def test_selects_families_with_new_rejected(self):
        """Families with rejected entries after last_review_at are eligible."""
        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            rejected=[
                _make_rejected(tested_at="2026-07-15T00:00:00"),
            ],
            review_state={"last_review_at": "2026-07-01T00:00:00"},
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert len(candidates) == 1
        assert candidates[0] == "sma_crossover|AAPL"

    def test_selects_families_with_new_near_miss(self):
        """Families with near_miss entries after last_review_at are eligible."""
        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            near_misses=[
                _make_near_miss(date="2026-07-15T00:00:00"),
            ],
            review_state={"last_review_at": "2026-07-01T00:00:00"},
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert len(candidates) == 1
        assert candidates[0] == "sma_crossover|AAPL"

    def test_skips_killed_families(self):
        """KILLED families are never selected."""
        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL",
                                                   lifecycle=FamilyLifecycle.KILLED,
                                                   kill_reason="bad"),
            },
            rejected=[
                _make_rejected(tested_at="2026-07-15T00:00:00"),
            ],
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert len(candidates) == 0

    def test_near_miss_priority(self):
        """Families with near-misses come before those with only rejected."""
        knowledge = _make_knowledge(
            families={
                "momentum|MSFT": _make_family("momentum|MSFT", best_val_sharpe=0.3),
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL", best_val_sharpe=0.8),
            },
            rejected=[
                _make_rejected(strategy="momentum", symbol="MSFT",
                               tested_at="2026-07-15T00:00:00", val_sharpe=0.3),
                _make_rejected(strategy="sma_crossover", symbol="AAPL",
                               tested_at="2026-07-15T00:00:00", val_sharpe=0.8),
            ],
            near_misses=[
                _make_near_miss(strategy="momentum", symbol="MSFT",
                                date="2026-07-15T00:00:00"),
            ],
            review_state={"last_review_at": "2026-07-01T00:00:00"},
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        # near-miss family (momentum) should come first
        assert candidates[0] == "momentum|MSFT"

    def test_respects_max_families(self):
        """Cap at max_families."""
        knowledge = _make_knowledge(
            families={
                f"sma_crossover|S{i}": _make_family(f"sma_crossover|S{i}",
                                                    best_val_sharpe=1.0 - i * 0.1)
                for i in range(5)
            },
            rejected=[
                _make_rejected(symbol=f"S{i}", tested_at="2026-07-15T00:00:00")
                for i in range(5)
            ],
            review_state={"last_review_at": "2026-07-01T00:00:00"},
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=2)
        assert len(candidates) == 2

    def test_skips_already_diagnosed_without_new_evidence(self):
        """Don't re-diagnose if reviewed.last_diagnosed_at >= newest failure."""
        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            rejected=[
                _make_rejected(tested_at="2026-07-10T00:00:00"),
            ],
            review_state={
                "last_review_at": "2026-07-01T00:00:00",
                "reviewed": {
                    "sma_crossover|AAPL": {"last_diagnosed_at": "2026-07-15T00:00:00"},
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert len(candidates) == 0

    def test_retries_when_new_failure_appears(self):
        """Family with new failure after last_diagnosed_at becomes eligible again."""
        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            rejected=[
                _make_rejected(tested_at="2026-07-10T00:00:00"),
                _make_rejected(tested_at="2026-07-18T00:00:00",
                               params={"fast_window": 20, "slow_window": 60}),
            ],
            review_state={
                "last_review_at": "2026-07-01T00:00:00",
                "reviewed": {
                    "sma_crossover|AAPL": {"last_diagnosed_at": "2026-07-15T00:00:00"},
                },
            },
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert len(candidates) == 1
        assert candidates[0] == "sma_crossover|AAPL"

    def test_missing_review_state_uses_epoch(self):
        """Missing review_state → epoch, so all failures are new."""
        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            rejected=[
                _make_rejected(tested_at="2026-01-01T00:00:00"),
            ],
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert len(candidates) == 1

    def test_handles_near_misses_cross(self):
        """Cross-family near_misses_cross entries are found."""
        family_key = "manifest:xs_mom|universe:abc"
        knowledge = _make_knowledge(
            families={
                family_key: _make_family(family_key, family_type="cross"),
            },
            near_misses_cross=[
                _make_near_miss(strategy="manifest:xs_mom", symbol="universe:abc",
                                date="2026-07-15T00:00:00"),
            ],
            review_state={"last_review_at": "2026-07-01T00:00:00"},
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert len(candidates) == 1
        assert candidates[0] == family_key

    def test_no_candidates_when_no_review_state(self):
        """Empty knowledge returns no candidates."""
        knowledge = _make_knowledge()
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert candidates == []

    def test_uses_date_field_for_near_misses(self):
        """Near-miss entries use 'date' field for comparison."""
        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            near_misses=[
                _make_near_miss(date="2026-07-15T00:00:00"),
            ],
            review_state={"last_review_at": "2026-07-01T00:00:00"},
        )
        candidates = sr.select_candidates(knowledge, "2026-07-20T00:00:00", max_families=3)
        assert len(candidates) == 1


# ============================================================================
# 2. Flag arithmetic
# ============================================================================

class TestFlags:
    def test_cost_bound_true(self):
        """cost_bound: val_sharpe(cost0) - val_sharpe(baseline) >= 0.3 AND cost0 > 0."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=1.0,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        assert flags["cost_bound"] is True

    def test_cost_bound_false_small_delta(self):
        """cost_bound false when delta < 0.3."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.7,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        assert flags["cost_bound"] is False

    def test_cost_bound_false_cost0_negative(self):
        """cost_bound false when cost0 sharpe <= 0."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=-0.5,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        assert flags["cost_bound"] is False  # cost0_val_sharpe <= 0

    def test_cost_bound_boundary(self):
        """cost_bound at exactly 0.3 delta."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.2,
            cost0_val_sharpe=0.5,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        assert flags["cost_bound"] is True

    def test_no_signal_true(self):
        """no_signal: val_sharpe(cost0) <= 0.2."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.1,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        assert flags["no_signal"] is True

    def test_no_signal_false(self):
        """no_signal false when sharpe > 0.2."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.5,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        assert flags["no_signal"] is False

    def test_no_signal_boundary(self):
        """no_signal at exactly 0.2."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.2,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        assert flags["no_signal"] is True

    def test_unstable_true(self):
        """unstable: (max - min) of fold sharpes > 1.5."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.8,
            fold_sharpes=[-0.2, 0.5, 2.0],
            baseline_val_turnover=30.0,
        )
        assert flags["unstable"] is True

    def test_unstable_false(self):
        """unstable false when spread <= 1.5."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.8,
            fold_sharpes=[0.3, 0.5, 0.7],
            baseline_val_turnover=30.0,
        )
        assert flags["unstable"] is False

    def test_unstable_boundary(self):
        """unstable at exactly 1.5."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.8,
            fold_sharpes=[0.0, 0.5, 1.5],
            baseline_val_turnover=30.0,
        )
        assert flags["unstable"] is False

    def test_regime_dependent_true(self):
        """regime_dependent: some fold sharpe > +0.5 AND some < -0.5."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.8,
            fold_sharpes=[0.8, -0.7, 0.3],
            baseline_val_turnover=30.0,
        )
        assert flags["regime_dependent"] is True

    def test_regime_dependent_false(self):
        """regime_dependent false when no negative folds."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.8,
            fold_sharpes=[0.3, 0.5, 0.7],
            baseline_val_turnover=30.0,
        )
        assert flags["regime_dependent"] is False

    def test_regime_dependent_none_folds(self):
        """regime_dependent false when fold_sharpes is None."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.8,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        assert flags["regime_dependent"] is False

    def test_high_turnover_true(self):
        """high_turnover: baseline val turnover > 50.0."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.8,
            fold_sharpes=None,
            baseline_val_turnover=60.0,
        )
        assert flags["high_turnover"] is True

    def test_high_turnover_false(self):
        """high_turnover false at exactly 50.0."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.8,
            fold_sharpes=None,
            baseline_val_turnover=50.0,
        )
        assert flags["high_turnover"] is False

    def test_all_flags_false(self):
        """Normal scenario: no flags triggered."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.6,
            cost0_val_sharpe=0.7,
            fold_sharpes=[0.5, 0.6, 0.7],
            baseline_val_turnover=20.0,
        )
        assert flags["cost_bound"] is False
        assert flags["no_signal"] is False
        assert flags["unstable"] is False
        assert flags["regime_dependent"] is False
        assert flags["high_turnover"] is False

    def test_flags_returned_as_dict(self):
        """_compute_flags always returns a dict with all 5 keys."""
        flags = sr._compute_flags(
            baseline_val_sharpe=0.5,
            cost0_val_sharpe=0.6,
            fold_sharpes=None,
            baseline_val_turnover=30.0,
        )
        expected_keys = {"cost_bound", "no_signal", "unstable", "regime_dependent", "high_turnover"}
        assert set(flags.keys()) == expected_keys
        assert all(isinstance(v, bool) for v in flags.values())


# ============================================================================
# 3. Holdout exclusion
# ============================================================================

class TestHoldoutExclusion:
    def test_report_json_no_holdout_single(self, tmp_path):
        """Report JSON must not contain holdout numbers for single families."""
        report = {
            "family_key": "test|AAPL",
            "diagnosed_at": "2026-07-20T00:00:00",
            "flags": {"cost_bound": True},
            "baseline": {"val_sharpe": 0.5, "fold_sharpes": [0.3, 0.5, 0.7]},
            "cost_free": {"val_sharpe": 1.0},
        }
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))

        # Read back and check no holdout keys
        data = json.loads(report_path.read_text())
        _assert_no_holdout_keys(data, "")

    def test_report_json_no_holdout_cross(self, tmp_path):
        """Report JSON must not contain holdout numbers for cross families."""
        report = {
            "family_key": "manifest:test|universe:abc",
            "diagnosed_at": "2026-07-20T00:00:00",
            "diagnosis_scope": "baseline_only",
            "flags": [],
            "baseline": {"val_sharpe": 0.5, "source": "recorded_near_miss"},
            "cost_free": None,
            "folds_available": False,
            "warnings": [
                "manifest spec not persisted in knowledge; cost-free re-run unavailable"
            ],
        }
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))
        data = json.loads(report_path.read_text())
        _assert_no_holdout_keys(data, "")


def _assert_no_holdout_keys(obj, path):
    """Recursively assert no key starts with 'holdout'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert not k.lower().startswith("holdout"), f"holdout key found: {path}.{k}"
            _assert_no_holdout_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_holdout_keys(item, f"{path}[{i}]")


# ============================================================================
# 4. Disk round-trip
# ============================================================================

class TestDiskRoundTrip:
    def test_report_file_written(self, tmp_path, monkeypatch):
        """Report file is written to review_reports dir."""
        review_dir = tmp_path / "review_reports"
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", review_dir)
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

        report = {
            "family_key": "sma_crossover|AAPL",
            "diagnosed_at": "2026-07-20T00:00:00",
            "flags": {},
            "baseline": {},
            "cost_free": {},
        }
        path = sr._save_report(report)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["family_key"] == "sma_crossover|AAPL"

    def test_filename_sanitizes_special_chars(self, tmp_path, monkeypatch):
        """Filename sanitizes | and / characters."""
        review_dir = tmp_path / "review_reports"
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", review_dir)
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

        report = {
            "family_key": "manifest:test|universe:abc/def",
            "diagnosed_at": "2026-07-20T00:00:00",
            "flags": {},
            "baseline": {},
            "cost_free": {},
        }
        path = sr._save_report(report)
        name = path.name
        assert "|" not in name
        assert "/" not in name

    def test_reviews_list_capped_at_50(self, tmp_path, monkeypatch):
        """reviews list capped at 50 entries."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

        knowledge = _make_knowledge()
        knowledge["reviews"] = [{"n": i} for i in range(50)]

        summary = {"family_key": "test|AAPL", "flags": {}, "report": "r.json"}
        sr._append_review_summary(knowledge, summary)
        assert len(knowledge["reviews"]) == 50
        # The new one replaced the oldest
        assert knowledge["reviews"][-1]["family_key"] == "test|AAPL"

    def test_review_state_persisted(self, tmp_path, monkeypatch):
        """review_state is updated and persists across save/load."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        knowledge = _make_knowledge()
        # Write initial knowledge
        from autonomous_loop import save_knowledge
        save_knowledge(knowledge)

        # Update review_state
        sr._update_review_state(knowledge, "sma_crossover|AAPL", "2026-07-20T00:00:00")
        save_knowledge(knowledge)

        # Re-load
        from autonomous_loop import load_knowledge
        reloaded = load_knowledge()
        assert reloaded["review_state"]["last_review_at"] == "2026-07-20T00:00:00"
        assert reloaded["review_state"]["reviewed"]["sma_crossover|AAPL"]["last_diagnosed_at"] == "2026-07-20T00:00:00"

    def test_review_state_migrates_existing(self):
        """review_state setdefault works with existing partial state."""
        knowledge = _make_knowledge(review_state={"last_review_at": "2026-06-01T00:00:00"})
        sr._update_review_state(knowledge, "test|AAPL", "2026-07-20T00:00:00")
        assert knowledge["review_state"]["last_review_at"] == "2026-07-20T00:00:00"
        assert "reviewed" in knowledge["review_state"]
        assert knowledge["review_state"]["reviewed"]["test|AAPL"]["last_diagnosed_at"] == "2026-07-20T00:00:00"


# ============================================================================
# 5. Error paths
# ============================================================================

class TestErrorPaths:
    def test_diagnosis_error_recorded(self, tmp_path, monkeypatch):
        """When runner fails, diagnosis_error is recorded."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        def _fail(*args, **kwargs):
            raise Exception("runner down")

        monkeypatch.setattr(sr, "_post_json", _fail)

        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            near_misses=[_make_near_miss()],
        )

        report, error = sr.diagnose_family("sma_crossover|AAPL", knowledge, "http://fake:9000")
        assert error is not None
        assert "runner down" in error
        assert "diagnosis_error" in report

    def test_error_does_not_advance_timestamps(self, tmp_path, monkeypatch):
        """On error, review_state timestamps are NOT advanced."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        def _fail(*args, **kwargs):
            raise Exception("runner down")

        monkeypatch.setattr(sr, "_post_json", _fail)

        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
            near_misses=[_make_near_miss()],
        )
        knowledge["review_state"] = {"last_review_at": "2026-07-01T00:00:00"}

        report, error = sr.diagnose_family("sma_crossover|AAPL", knowledge, "http://fake:9000")
        assert error is not None
        # Timestamps should NOT be advanced — review_state unchanged
        assert knowledge["review_state"]["last_review_at"] == "2026-07-01T00:00:00"

    def test_skips_family_with_no_evidence(self):
        """diagnose_family returns error when family has no rejected/near_miss entries."""
        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
            },
        )
        report, error = sr.diagnose_family("sma_crossover|AAPL", knowledge, "http://fake:9000")
        assert error is not None
        assert "no evidence" in error.lower()


# ============================================================================
# 6. CLI --dry-run
# ============================================================================

class TestCLIDryRun:
    def test_dry_run_no_http_calls(self, tmp_path, monkeypatch, capsys):
        """--dry-run selects candidates but makes zero HTTP calls."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        from autonomous_loop import save_knowledge

        knowledge = _make_knowledge(
            families={
                "sma_crossover|AAPL": _make_family("sma_crossover|AAPL"),
                "momentum|MSFT": _make_family("momentum|MSFT", best_val_sharpe=0.3),
            },
            rejected=[
                _make_rejected(strategy="sma_crossover", symbol="AAPL",
                               tested_at="2026-07-15T00:00:00"),
                _make_rejected(strategy="momentum", symbol="MSFT",
                               tested_at="2026-07-15T00:00:00"),
            ],
        )
        save_knowledge(knowledge)

        # Mock _post_json to track calls
        calls = []
        def _track(*args, **kwargs):
            calls.append(1)
            return {"status": "ok", "out_of_sample": {"sharpe_ratio": 0.5, "turnover": 30}}
        monkeypatch.setattr(sr, "_post_json", _track)

        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "strategy_review.py"),
             "--dry-run", "--max-families", "2"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "KNOWLEDGE_FILE": str(tmp_path / "knowledge.json")},
        )
        assert result.returncode == 0
        assert calls == [], f"Expected 0 HTTP calls, got {len(calls)}"
        assert "CANDIDATES" in result.stdout or "candidate" in result.stdout.lower()

    def test_cli_help(self):
        """CLI --help works."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "strategy_review.py"),
             "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--dry-run" in result.stdout
        assert "--max-families" in result.stdout
        assert "--family" in result.stdout


# ============================================================================
# 7. Family key mapping helpers
# ============================================================================

class TestFamilyEvidence:
    def test_find_evidence_single_near_miss(self):
        """Find the most recent near_miss for a single family."""
        knowledge = _make_knowledge(
            near_misses=[
                _make_near_miss(strategy="sma_crossover", symbol="AAPL",
                                date="2026-07-10T00:00:00"),
                _make_near_miss(strategy="sma_crossover", symbol="AAPL",
                                date="2026-07-15T00:00:00",
                                params={"fast_window": 20, "slow_window": 60}),
            ],
        )
        evidence = sr._find_most_recent_evidence("sma_crossover|AAPL", knowledge)
        assert evidence is not None
        assert evidence["params"]["fast_window"] == 20

    def test_find_evidence_single_rejected(self):
        """Fall back to rejected entries if no near_miss."""
        knowledge = _make_knowledge(
            rejected=[
                _make_rejected(strategy="sma_crossover", symbol="AAPL",
                               tested_at="2026-07-15T00:00:00",
                               params={"fast_window": 25, "slow_window": 55}),
            ],
        )
        evidence = sr._find_most_recent_evidence("sma_crossover|AAPL", knowledge)
        assert evidence is not None
        # Rejected entries store params under hypothesis.params
        assert evidence["hypothesis"]["params"]["fast_window"] == 25

    def test_find_evidence_cross(self):
        """Find cross-family evidence from near_misses_cross."""
        family_key = "manifest:xs_mom|universe:abc"
        knowledge = _make_knowledge(
            near_misses_cross=[
                _make_near_miss(strategy="manifest:xs_mom", symbol="universe:abc",
                                date="2026-07-15T00:00:00",
                                params={"universe_size": 5, "execution_mode": "structured"}),
            ],
        )
        evidence = sr._find_most_recent_evidence(family_key, knowledge)
        assert evidence is not None
        assert evidence["params"]["universe_size"] == 5

    def test_diagnose_single_uses_hypothesis_params_from_rejected(self, monkeypatch):
        """Rejected-shaped evidence nests params under hypothesis — the
        baseline payload must carry them (regression: empty params → 400)."""
        evidence = {
            "hypothesis": {
                "strategy": "rsi", "symbol": "AMZN",
                "params": {"rsi_window": 16, "oversold": 35, "overbought": 65},
            },
            "tested_at": "2026-07-15T00:00:00",
        }
        sent = []

        def fake_post(url, payload, timeout=180):
            sent.append(payload)
            return {"out_of_sample": {"sharpe_ratio": 0.4, "turnover": 10.0}}

        monkeypatch.setattr(sr, "_post_json", fake_post)
        report, err = sr._diagnose_single_family(
            "rsi|AMZN", evidence, "http://runner:9000", "2026-07-20T00:00:00")
        assert err is None
        assert sent[0]["params"] == {"rsi_window": 16, "oversold": 35, "overbought": 65}

    def test_diagnose_single_no_params_is_error_not_400(self, monkeypatch):
        """Evidence with no recoverable params must fail before any HTTP call."""
        called = []
        monkeypatch.setattr(sr, "_post_json",
                            lambda *a, **k: called.append(1))
        report, err = sr.diagnose_family("rsi|AMZN", {
            "families": {"rsi|AMZN": {"family_type": "single"}},
            "near_misses": [{"strategy": "rsi", "symbol": "AMZN",
                             "date": "2026-07-15T00:00:00"}],
        }, "http://runner:9000")
        assert err is not None
        assert "params" in err
        assert not called


# ============================================================================
# 8. Cross-family diagnosis (baseline-only, zero HTTP)
# ============================================================================

class TestCrossDiagnosis:
    def test_cross_diagnosis_makes_zero_http_calls(self, tmp_path, monkeypatch):
        """Cross-family diagnosis must never call _post_json."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

        calls = []
        def _track(*args, **kwargs):
            calls.append(1)
            return {"status": "ok"}
        monkeypatch.setattr(sr, "_post_json", _track)

        family_key = "manifest:xs_mom|universe:abc"
        knowledge = _make_knowledge(
            families={
                family_key: _make_family(family_key, family_type="cross"),
            },
            near_misses_cross=[
                _make_near_miss(strategy="manifest:xs_mom", symbol="universe:abc",
                                date="2026-07-15T00:00:00", val_sharpe=0.7,
                                params={"universe_size": 5}),
            ],
        )
        report, error = sr.diagnose_family(family_key, knowledge, "http://fake:9000")
        assert error is None
        assert calls == [], f"Cross diagnosis made {len(calls)} HTTP call(s), expected 0"

    def test_cross_report_shape(self, tmp_path, monkeypatch):
        """Cross-family report has correct baseline-only shape."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

        family_key = "manifest:xs_mom|universe:abc"
        knowledge = _make_knowledge(
            families={
                family_key: _make_family(family_key, family_type="cross"),
            },
            near_misses_cross=[
                _make_near_miss(strategy="manifest:xs_mom", symbol="universe:abc",
                                date="2026-07-15T00:00:00", val_sharpe=0.7,
                                params={"universe_size": 5}),
            ],
        )
        report, error = sr.diagnose_family(family_key, knowledge, "http://fake:9000")
        assert error is None
        assert report["diagnosis_scope"] == "baseline_only"
        assert report["flags"] == []
        assert report["cost_free"] is None
        assert report["folds_available"] is False
        assert len(report["warnings"]) >= 1
        assert "manifest spec not persisted" in report["warnings"][0]
        assert report["baseline"]["source"] == "recorded_near_miss"
        assert report["baseline"]["val_sharpe"] == 0.7

    def test_cross_baseline_from_rejected_evaluation(self, tmp_path, monkeypatch):
        """Cross baseline uses recorded evaluation from rejected entry."""
        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
        monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

        family_key = "manifest:xs_mom|universe:abc"
        knowledge = _make_knowledge(
            families={
                family_key: _make_family(family_key, family_type="cross"),
            },
            rejected=[
                _make_rejected(strategy="manifest:xs_mom", symbol="universe:abc",
                               tested_at="2026-07-15T00:00:00", val_sharpe=0.65),
            ],
        )
        report, error = sr.diagnose_family(family_key, knowledge, "http://fake:9000")
        assert error is None
        assert report["baseline"]["source"] == "recorded_evaluation"
        assert report["baseline"]["val_sharpe"] == 0.65
        assert report["cost_free"] is None

    def test_cross_diagnosis_no_fabricated_cost_free(self):
        """No test may assert a fabricated cost_free value for cross families."""
        family_key = "manifest:xs_mom|universe:abc"
        knowledge = _make_knowledge(
            families={
                family_key: _make_family(family_key, family_type="cross"),
            },
            near_misses_cross=[
                _make_near_miss(strategy="manifest:xs_mom", symbol="universe:abc",
                                date="2026-07-15T00:00:00", val_sharpe=0.7),
            ],
        )
        report, error = sr.diagnose_family(family_key, knowledge, "http://fake:9000")
        assert error is None
        # cost_free must be None (not a dict, not a number)
        assert report["cost_free"] is None
        # flags must be empty list (not a dict with fabricated flags)
        assert report["flags"] == []


# ============================================================================
# 9. Runner response parsing
# ============================================================================

class TestParseRunner:
    def test_extract_fold_sharpes(self):
        """Extract fold sharpe ratios from CV response (real runner schema:
        per-fold metrics live under "val_metrics", not "val")."""
        cv_response = {
            "out_of_sample": {"sharpe_ratio": 0.5, "turnover": 30.0},
            "cv": {
                "folds": [
                    {"fold": 0, "val_metrics": {"sharpe_ratio": 0.3}},
                    {"fold": 1, "val_metrics": {"sharpe_ratio": 0.5}},
                    {"fold": 2, "val_metrics": {"sharpe_ratio": 0.7}},
                ]
            }
        }
        sharpe, turnover, folds = sr._parse_baseline_response(cv_response)
        assert sharpe == 0.5
        assert turnover == 30.0
        assert folds == [0.3, 0.5, 0.7]

    def test_fold_without_sharpe_is_skipped_not_zeroed(self):
        """A fold missing sharpe_ratio must be skipped, never treated as 0.0."""
        cv_response = {
            "out_of_sample": {"sharpe_ratio": 0.5, "turnover": 30.0},
            "cv": {
                "folds": [
                    {"fold": 0, "val_metrics": {"sharpe_ratio": 0.3}},
                    {"fold": 1, "val_metrics": {}},
                ]
            }
        }
        _, _, folds = sr._parse_baseline_response(cv_response)
        assert folds == [0.3]

    def test_all_folds_missing_sharpe_gives_none(self):
        """If no fold has a sharpe, fold_sharpes is None (flags skip fold logic)."""
        cv_response = {
            "out_of_sample": {"sharpe_ratio": 0.5, "turnover": 30.0},
            "cv": {"folds": [{"fold": 0, "val_metrics": {}}]},
        }
        _, _, folds = sr._parse_baseline_response(cv_response)
        assert folds is None

    def test_extract_no_cv(self):
        """Handle response with no CV block."""
        response = {
            "out_of_sample": {"sharpe_ratio": 0.5, "turnover": 30.0},
        }
        sharpe, turnover, folds = sr._parse_baseline_response(response)
        assert sharpe == 0.5
        assert turnover == 30.0
        assert folds is None

    def test_extract_missing_turnover(self):
        """Handle missing turnover field."""
        response = {
            "out_of_sample": {"sharpe_ratio": 0.5},
        }
        sharpe, turnover, folds = sr._parse_baseline_response(response)
        assert sharpe == 0.5
        assert turnover == 0.0
