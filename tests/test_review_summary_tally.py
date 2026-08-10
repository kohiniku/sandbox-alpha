"""
Issue #85: REVIEW_SUMMARY tallies pre-downgrade LLM verdict, contradicting
the per-item applied verdict it summarizes.

``strategy_review.py::main`` captured ``result = apply_verdict(...)`` (the
applied / post-downgrade verdict) but then tallied kill/refine/keep counts
off ``verdict_dict["verdict"]`` (the raw LLM verdict). Whenever the
``n_trials < MIN_TRIALS_FOR_KILL`` downgrade fires (kill -> keep), or the
cross-refine downgrade fires (refine -> keep), or the refine cap fires
(refine -> kill), the summary line contradicts the REVIEW_VERDICT lines
printed for the same run.

These tests drive the full ``main()`` path in-process (mocked diagnosis +
mocked LLM) and assert REVIEW_SUMMARY counts the *applied* verdict.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_constants import FamilyLifecycle, REFINE_CAP

# Module under test
import strategy_review as sr


# ============================================================================
# Helpers (same shapes as tests/test_review_verdict.py)
# ============================================================================

def _make_family(key, family_type="single", lifecycle=FamilyLifecycle.CANDIDATE,
                 best_val_sharpe=0.5, kill_reason="", refine_count=0,
                 n_trials=5):
    return {
        "n_trials": n_trials,
        "best_val_sharpe": best_val_sharpe,
        "best_params": {},
        "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                          "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": "2026-07-01T00:00:00",
        "family_type": family_type,
        "lifecycle": lifecycle,
        "refine_count": refine_count,
        "kill_reason": kill_reason,
    }


def _make_report(family_key="sma_crossover|AAPL", family_type="single"):
    return {
        "family_key": family_key,
        "family_type": family_type,
        "diagnosed_at": "2026-07-20T00:00:00",
        "flags": {"cost_bound": False, "no_signal": False, "unstable": False,
                  "regime_dependent": False, "high_turnover": False},
        "baseline": {"val_sharpe": 0.5, "val_turnover": 30.0},
        "cost_free": {"val_sharpe": 0.7},
        "folds_available": True,
    }


def _make_knowledge(families):
    return {
        "families": families,
        "rejected": [],
        "near_misses": [],
        "near_misses_cross": [],
        "review_state": {},
        "reviews": [],
        "adopted": [],
        "tested_combinations": [],
        "errors": [],
    }


def _run_main(tmp_path, monkeypatch, families, llm_verdicts):
    """Run sr.main() in-process over the given families with mocked
    diagnosis and LLM. Returns captured stdout.

    ``llm_verdicts`` maps family_key -> verdict dict returned by the judge.
    """
    monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
    monkeypatch.setattr(sr, "REVIEW_REPORTS_DIR", tmp_path / "review_reports")
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

    from autonomous_loop import save_knowledge

    knowledge = _make_knowledge(families)
    save_knowledge(knowledge)

    # Mock diagnosis: always succeed with a neutral report
    monkeypatch.setattr(
        sr, "diagnose_family",
        lambda fk, kn, ru: (_make_report(family_key=fk), None),
    )
    # Mock LLM judge (single-family runs: one verdict for the only family)
    assert len(llm_verdicts) == 1, "test helper supports single-family runs"
    _verdict = next(iter(llm_verdicts.values()))
    monkeypatch.setattr(sr, "_call_review_llm", lambda _msgs: _verdict)

    monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://fake:9000")

    import io
    import sys as _sys
    captured = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = captured
    try:
        assert len(families) == 1, "test helper supports single-family runs"
        fk = next(iter(families))
        monkeypatch.setattr(_sys, "argv", ["strategy_review.py",
                                           "--family", fk, "--max-families", "1"])
        sr.main()
    finally:
        _sys.stdout = old_stdout

    return captured.getvalue()


def _summary_line(output):
    for line in output.splitlines():
        if line.startswith("REVIEW_SUMMARY") and "kill=" in line:
            return line
    raise AssertionError(f"no REVIEW_SUMMARY with counts in output:\n{output}")


def _run_single(tmp_path, monkeypatch, family_key, family, llm_verdict):
    """Convenience: one family, one LLM verdict."""
    output = _run_main(tmp_path, monkeypatch, {family_key: family},
                       {family_key: llm_verdict})
    return output


# ============================================================================
# 1. Core regression: kill downgraded to keep must be counted as keep
# ============================================================================

class TestSummaryTalliesAppliedVerdict:
    def test_downgraded_kill_counted_as_keep(self, tmp_path, monkeypatch):
        """LLM says kill but n_trials < MIN_TRIALS_FOR_KILL -> applied keep.

        REVIEW_SUMMARY must show keep=1 kill=0 (pre-fix it showed kill=1).
        Reproduces the 2026-08-06 production run where two keep verdicts
        were summarized as kill=2.
        """
        family_key = "sma_crossover|AAPL"
        family = _make_family(family_key, n_trials=2)  # < MIN_TRIALS_FOR_KILL
        verdict = {"verdict": "kill", "rationale": "no signal, consistently bad",
                   "refine_proposal": None}

        output = _run_single(tmp_path, monkeypatch, family_key, family, verdict)

        # Per-item line says keep (this was already correct pre-fix)
        assert f"REVIEW_VERDICT {family_key} verdict=keep" in output
        # Summary must agree: applied verdict, not raw LLM verdict
        line = _summary_line(output)
        assert "kill=0" in line, line
        assert "keep=1" in line, line
        assert "refine=0" in line, line

    def test_downgraded_cross_refine_counted_as_keep(self, tmp_path, monkeypatch):
        """Cross family refine is downgraded to keep -> counted as keep."""
        family_key = "manifest:xs_mom|universe:abc"
        family = _make_family(family_key, family_type="cross", n_trials=1)
        verdict = {"verdict": "refine", "rationale": "try different universe",
                   "refine_proposal": {"params": {"universe_size": 10},
                                       "change_summary": "Bigger universe"}}

        output = _run_single(tmp_path, monkeypatch, family_key, family, verdict)

        assert f"REVIEW_VERDICT {family_key} verdict=keep" in output
        line = _summary_line(output)
        assert "refine=0" in line, line
        assert "keep=1" in line, line
        assert "kill=0" in line, line

    def test_refine_cap_applied_kill_counted_as_kill(self, tmp_path, monkeypatch):
        """Refine verdict at REFINE_CAP is applied as kill -> counted as kill."""
        family_key = "sma_crossover|AAPL"
        family = _make_family(family_key, refine_count=REFINE_CAP, n_trials=1)
        verdict = {"verdict": "refine", "rationale": "try again",
                   "refine_proposal": {"params": {"fast_window": 29, "slow_window": 99},
                                       "change_summary": "final attempt"}}

        output = _run_single(tmp_path, monkeypatch, family_key, family, verdict)

        assert f"REVIEW_VERDICT {family_key} verdict=kill" in output
        line = _summary_line(output)
        assert "kill=1" in line, line
        assert "refine=0" in line, line
        assert "keep=0" in line, line

    def test_undowngraded_verdicts_still_counted(self, tmp_path, monkeypatch):
        """Guard: a real kill (enough trials) still counts as kill=1.

        Ensures the fix tallies applied verdicts without losing
        non-downgraded ones.
        """
        family_key = "sma_crossover|AAPL"
        family = _make_family(family_key, n_trials=5)  # >= MIN_TRIALS_FOR_KILL
        verdict = {"verdict": "kill", "rationale": "consistently bad",
                   "refine_proposal": None}

        output = _run_single(tmp_path, monkeypatch, family_key, family, verdict)

        assert f"REVIEW_VERDICT {family_key} verdict=kill" in output
        line = _summary_line(output)
        assert "kill=1" in line, line
        assert "keep=0" in line, line
        assert "refine=0" in line, line

    def test_failopen_still_counted_separately(self, tmp_path, monkeypatch):
        """Guard: fail-open keep (llm_failure rationale) counts as failopen,
        not as a plain keep — the fix must not break this branch."""
        family_key = "sma_crossover|AAPL"
        family = _make_family(family_key, n_trials=5)
        verdict = {"verdict": "keep", "rationale": "llm_failure: exception: boom",
                   "refine_proposal": None}

        output = _run_single(tmp_path, monkeypatch, family_key, family, verdict)

        line = _summary_line(output)
        assert "failopen=1" in line, line
        assert "keep=0" in line, line
        assert "kill=0" in line, line

    def test_summary_consistent_with_review_verdict_lines(self, tmp_path, monkeypatch):
        """The summary must be internally consistent with the per-item
        REVIEW_VERDICT lines: count of verdict=X lines == summary tallies."""
        family_key = "mean_reversion|MSFT"
        family = _make_family(family_key, n_trials=1)
        verdict = {"verdict": "kill", "rationale": "bad", "refine_proposal": None}

        output = _run_single(tmp_path, monkeypatch, family_key, family, verdict)

        verdict_lines = [l for l in output.splitlines()
                         if l.startswith(f"REVIEW_VERDICT {family_key}")]
        assert len(verdict_lines) == 1
        applied = "keep" if "verdict=keep" in verdict_lines[0] else \
                  "kill" if "verdict=kill" in verdict_lines[0] else "refine"
        line = _summary_line(output)
        assert f"{applied}=1" in line, (line, verdict_lines)
