"""
Regression tests for issue #92: cross-type families stuck at lifecycle=candidate.

Problem: `apply_verdict` unconditionally downgraded refine → keep for
family_type == "cross" (comment: "crossファミリーはrefine不可"), and the
consumed manifest body was never persisted on the family record, so there was
no material to re-propose a mutated manifest even if refine were allowed.

Fix (go-forward):
  1. autonomous_loop persists the full consumed manifest spec as
     families[key]["last_manifest_spec"] on every manifest trial.
  2. strategy_review.apply_verdict: cross + refine now builds a mutated
     manifest from last_manifest_spec (params merged at top level /
     evaluator.extras, code_b64/logic_spec untouched) and queues it as a
     "manifest" backlog entry with source {"kind": "review_refine"}.
     Families without last_manifest_spec (pre-fix data) are still downgraded
     to keep with an explicit reason.

Disk round-trip is exercised (known sandbox-alpha lesson: in-memory-only
tests miss persistence bugs — 2026-07-19 id bug, 2026-07-24 attempts bug).
"""
import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backlog import _new_entry

from loop_constants import FamilyLifecycle, REFINE_CAP

# Module under test
import strategy_review as sr


def _b64(code: str) -> str:
    return base64.b64encode(code.encode("utf-8")).decode("ascii")


SAMPLE_CODE = (
    "import numpy as np\n"
    "import pandas as pd\n\n"
    "def generate_cross_signal(data):\n"
    "    return pd.DataFrame(1, index=next(iter(data.values())).index, columns=list(data.keys()))\n"
)


def _make_manifest_spec(name="xs_mom", extras=None, code_b64=None,
                        execution_mode="structured"):
    spec = {
        "name": name,
        "code_b64": code_b64 if code_b64 is not None else _b64(SAMPLE_CODE),
        "data_sources": [
            {"type": "ohlcv", "universe": ["AAPL", "MSFT", "GOOGL"],
             "start": "2020-01-01"}
        ],
        "model_artifacts": [],
        "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
        "evaluator": {
            "type": "portfolio",
            "metrics": ["sharpe", "max_drawdown_pct", "total_return_pct"],
        },
        "execution_mode": execution_mode,
    }
    if extras:
        spec["evaluator"]["extras"] = extras
    return spec


def _make_cross_family(key, last_manifest_spec=None, refine_count=0,
                       n_trials=2, lifecycle=FamilyLifecycle.CANDIDATE):
    return {
        "n_trials": n_trials,
        "best_val_sharpe": 0.5,
        "best_params": {"universe_size": 3, "execution_mode": "structured",
                        "primary_metric": "sharpe"},
        "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                          "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": "2026-08-01T00:00:00",
        "family_type": "cross",
        "lifecycle": lifecycle,
        "refine_count": refine_count,
        "kill_reason": "",
        "param_clusters": [],
        "distinct_clusters": 1,
        "last_manifest_spec": last_manifest_spec,
    }


def _make_knowledge(families=None):
    # Mirror the real knowledge.json top-level keys (see repo knowledge.json)
    # so run_loop's `knowledge["iterations"] += 1` etc. work in E2E tests.
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
        "iterations": 0,
        "tested": [],
        "superseded": [],
    }


def _refine_verdict(params, change_summary="param change"):
    return {
        "verdict": "refine",
        "rationale": "addressable flaw",
        "refine_proposal": {"params": params, "change_summary": change_summary},
    }


def _make_entry(spec=None, priority=0.95, source=None):
    if spec is None:
        spec = _make_manifest_spec()
    return _new_entry(
        "manifest", priority,
        source or {"kind": "paper", "ref": "test_paper.md"},
        spec, {"extra_criteria": []},
    )


# ============================================================================
# 1. Core regression: cross + refine with persisted spec → real refine
# ============================================================================

class TestCrossRefineWithSpec:
    def test_cross_refine_queues_mutated_manifest(self, tmp_path, monkeypatch):
        """Cross family WITH last_manifest_spec: refine is applied, not downgraded."""
        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        family_key = "manifest:xs_mom|universe:abc123"
        spec = _make_manifest_spec(extras={"cost_bps": 10})
        knowledge = _make_knowledge(families={
            family_key: _make_cross_family(family_key, last_manifest_spec=spec),
        })
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = _refine_verdict({"execution_mode": "expert", "cost_bps": 5})

        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)
        save_knowledge(knowledge)

        # Refine must actually be applied
        assert result == "refine"

        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["lifecycle"] == FamilyLifecycle.REFINING
        assert fam["refine_count"] == 1

        # Backlog must contain a manifest entry sourced from review_refine
        data = backlog.load()
        entries = [e for e in data["entries"]
                   if e["source"].get("kind") == "review_refine"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["type"] == "manifest"
        assert entry["source"]["ref"] == family_key

        # Mutated spec: params merged, identity + code untouched
        m = entry["spec"]
        assert m["name"] == "xs_mom"
        assert m["code_b64"] == spec["code_b64"]
        assert m["execution_mode"] == "expert"
        assert m["evaluator"]["extras"]["cost_bps"] == 5
        # original extras preserved, only overridden key changed
        assert "universe" in m["data_sources"][0]

    def test_cross_refine_disk_roundtrip(self, tmp_path, monkeypatch):
        """Disk round-trip: REFINING + refine_count survive reload."""
        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        family_key = "manifest:xs_mom|universe:abc123"
        knowledge = _make_knowledge(families={
            family_key: _make_cross_family(
                family_key, last_manifest_spec=_make_manifest_spec()),
        })
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        result = sr.apply_verdict(
            family_key, _refine_verdict({"execution_mode": "expert"}),
            knowledge, backlog)
        assert result == "refine"
        save_knowledge(knowledge)

        # Fresh load from disk
        reloaded = load_knowledge()
        fam = reloaded["families"][family_key]
        assert fam["lifecycle"] == FamilyLifecycle.REFINING
        assert fam["refine_count"] == 1

        # Backlog entry persisted on disk
        data = backlog.load()
        assert any(e["type"] == "manifest"
                   and e["source"].get("kind") == "review_refine"
                   for e in data["entries"])

    def test_cross_refine_cap_kills(self, tmp_path, monkeypatch):
        """Cross refine at REFINE_CAP → auto-kill (same as single families)."""
        from autonomous_loop import save_knowledge, load_knowledge
        from backlog import Backlog

        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        family_key = "manifest:xs_mom|universe:abc123"
        knowledge = _make_knowledge(families={
            family_key: _make_cross_family(
                family_key, last_manifest_spec=_make_manifest_spec(),
                refine_count=REFINE_CAP),
        })
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        result = sr.apply_verdict(
            family_key, _refine_verdict({"execution_mode": "expert"}),
            knowledge, backlog)
        save_knowledge(knowledge)

        assert result == "kill"
        fam = load_knowledge()["families"][family_key]
        assert fam["lifecycle"] == FamilyLifecycle.KILLED
        # No refine entry should be queued
        assert all(e.get("source", {}).get("kind") != "review_refine"
                   for e in backlog.load()["entries"])

    def test_cross_refine_rejects_protected_fields(self, tmp_path, monkeypatch, capsys):
        """Refine params must not override identity/code fields."""
        from autonomous_loop import save_knowledge
        from backlog import Backlog

        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        family_key = "manifest:xs_mom|universe:abc123"
        knowledge = _make_knowledge(families={
            family_key: _make_cross_family(
                family_key, last_manifest_spec=_make_manifest_spec()),
        })
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        verdict = _refine_verdict({"name": "evil_override"})
        result = sr.apply_verdict(family_key, verdict, knowledge, backlog)

        assert result == "keep"
        assert knowledge["families"][family_key]["lifecycle"] == FamilyLifecycle.CANDIDATE
        assert knowledge["families"][family_key]["refine_count"] == 0
        assert all(e.get("source", {}).get("kind") != "review_refine"
                   for e in backlog.load()["entries"])
        out = capsys.readouterr().out
        assert "REVIEW_REFINE_INVALID_PARAMS" in out

    def test_cross_refine_empty_params_downgraded(self, tmp_path, monkeypatch):
        """Refine with empty params → downgraded to keep (no queued entry)."""
        from autonomous_loop import save_knowledge
        from backlog import Backlog

        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        family_key = "manifest:xs_mom|universe:abc123"
        knowledge = _make_knowledge(families={
            family_key: _make_cross_family(
                family_key, last_manifest_spec=_make_manifest_spec()),
        })
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        result = sr.apply_verdict(
            family_key, _refine_verdict({}), knowledge, backlog)

        assert result == "keep"
        assert all(e.get("source", {}).get("kind") != "review_refine"
                   for e in backlog.load()["entries"])


# ============================================================================
# 2. Legacy path: no persisted spec → still downgraded, with explicit reason
# ============================================================================

class TestCrossRefineLegacyNoSpec:
    def test_cross_refine_without_spec_downgraded_to_keep(self, tmp_path, monkeypatch):
        """Pre-fix families (no last_manifest_spec) stay on the keep path."""
        from autonomous_loop import save_knowledge
        from backlog import Backlog

        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        family_key = "manifest:xs_mom|universe:abc123"
        knowledge = _make_knowledge(families={
            family_key: _make_cross_family(family_key, last_manifest_spec=None),
        })
        save_knowledge(knowledge)

        backlog = Backlog(str(tmp_path / "backlog.json"))
        result = sr.apply_verdict(
            family_key, _refine_verdict({"execution_mode": "expert"}),
            knowledge, backlog)

        assert result == "keep"
        assert knowledge["families"][family_key]["lifecycle"] == FamilyLifecycle.CANDIDATE
        assert knowledge["families"][family_key]["refine_count"] == 0
        assert all(e.get("source", {}).get("kind") != "review_refine"
                   for e in backlog.load()["entries"])


# ============================================================================
# 3. Judge prompt: cross families get a manifest-params note
# ============================================================================

class TestJudgePromptCross:
    def test_judge_prompt_cross_notes_manifest_params(self):
        """Cross families: prompt must tell the judge params are manifest fields."""
        family = _make_cross_family("manifest:xs_mom|universe:abc123")
        report = {
            "family_key": "manifest:xs_mom|universe:abc123",
            "family_type": "cross",
            "flags": {"cost_bound": True},
            "baseline": {"val_sharpe": 0.5, "val_turnover": 10.0},
            "cost_free": {"val_sharpe": 0.7},
        }
        messages = sr._build_judge_prompt(report, family, _make_knowledge())
        user_content = messages[1]["content"]
        assert "CROSS-SECTIONAL" in user_content or "manifest" in user_content.lower()
        assert "universe_size" in user_content
        assert "evaluator.extras" in user_content

    def test_judge_prompt_single_has_no_manifest_note(self):
        """Single families keep the old prompt (no manifest note)."""
        family = {"family_type": "single", "n_trials": 5, "best_val_sharpe": 0.5,
                  "refine_count": 0, "gate_failures": {}}
        report = {
            "family_key": "sma_crossover|AAPL",
            "family_type": "single",
            "flags": {},
            "baseline": {"val_sharpe": 0.5, "val_turnover": 10.0},
            "cost_free": {"val_sharpe": 0.7},
        }
        messages = sr._build_judge_prompt(report, family, _make_knowledge())
        user_content = messages[1]["content"]
        assert "evaluator.extras" not in user_content


# ============================================================================
# 4. Loop side: manifest consumption persists last_manifest_spec
# ============================================================================

class TestManifestSpecPersistence:
    def test_manifest_trial_persists_last_manifest_spec(self, tmp_path, monkeypatch):
        """Consuming a manifest entry stores the full spec on the family record."""
        from autonomous_loop import run_loop
        from backlog import Backlog

        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        spec = _make_manifest_spec()
        bl = Backlog(str(backlog_path))
        bl.add_entry(_make_entry(spec))

        mock_response = {
            "status": "ok",
            "manifest_name": "manifest:xs_mom",
            "universe_size": 3,
            "n_days": 252,
            "execution_mode": "structured",
            "metrics": {
                "val_sharpe": 1.2,
                "val_max_drawdown_pct": -10.0,
                "val_total_return_pct": 15.0,
                "holdout_sharpe": 0.8,
                "holdout_max_drawdown_pct": -8.0,
                "holdout_total_return_pct": 10.0,
            },
            "config": {"weighting": "equal_active_signals", "benchmark": None},
            "train_end": "2024-01-01",
            "val_end": "2024-06-01",
        }

        def mock_urlopen(req, timeout=300):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        kf = tmp_path / "knowledge.json"
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
        monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
        (tmp_path / "results").mkdir(exist_ok=True)

        from unittest.mock import patch
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            run_loop(1)

        knowledge = json.loads(kf.read_text())
        families = knowledge.get("families", {})
        assert families, "expected a family record after manifest trial"
        fam = list(families.values())[0]
        assert fam.get("last_manifest_spec") == spec, (
            "family record must persist the consumed manifest body (issue #92)"
        )
        assert fam["family_type"] == "cross"


# ============================================================================
# 5. End-to-end smoke: refine-created manifest entry is consumable by the loop
# ============================================================================

class TestCrossRefineEndToEnd:
    def test_refine_entry_consumed_by_loop(self, tmp_path, monkeypatch):
        """The manifest entry queued by cross-refine is consumed by run_loop
        (/run_manifest 200) and the family keeps its REFINING state (issue #92
        smoke requirement: 実runnerでのsmoke)."""
        from autonomous_loop import run_loop, save_knowledge, load_knowledge
        from backlog import Backlog

        monkeypatch.setattr(sr, "KNOWLEDGE_FILE", tmp_path / "knowledge.json")
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_path / "knowledge.json")

        backlog_path = tmp_path / "backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        family_key = "manifest:xs_mom|universe:abc123"
        knowledge = _make_knowledge(families={
            family_key: _make_cross_family(
                family_key, last_manifest_spec=_make_manifest_spec()),
        })
        save_knowledge(knowledge)

        # Step 1: review refines the cross family → manifest entry queued
        backlog = Backlog(str(backlog_path))
        result = sr.apply_verdict(
            family_key, _refine_verdict({"execution_mode": "expert", "cost_bps": 5}),
            knowledge, backlog)
        assert result == "refine"
        save_knowledge(knowledge)

        entries = [e for e in backlog.load()["entries"]
                   if e.get("source", {}).get("kind") == "review_refine"]
        assert len(entries) == 1
        mutated_spec = entries[0]["spec"]
        assert mutated_spec["execution_mode"] == "expert"
        assert mutated_spec["evaluator"]["extras"]["cost_bps"] == 5

        # The loop derives the family key from the manifest spec
        # (strategy=f"manifest:{name}", symbol=f"universe:{sha256(universe)[:8]}")
        # — seed the family under that derived key so the consumed trial lands
        # on the family we refined (not a parallel new family).
        from autonomous_loop import _extract_universe_meta, _family_key
        _universe, _uh, _usize = _extract_universe_meta(mutated_spec)
        derived_key = _family_key(
            f"manifest:{mutated_spec['name']}", f"universe:{_uh}", "cross")
        assert derived_key != "manifest:xs_mom|universe:abc123"  # sanity: real hash
        # Re-seed under the correct key in the post-refine state (step 1's
        # verdict set REFINING + refine_count=1) so Step 2's consumed trial
        # joins the same family.
        knowledge = _make_knowledge(families={
            derived_key: _make_cross_family(
                derived_key, last_manifest_spec=_make_manifest_spec(),
                refine_count=1, lifecycle=FamilyLifecycle.REFINING),
        })
        save_knowledge(knowledge)
        family_key = derived_key

        # Step 2: loop consumes it — mock /run_manifest returning 200 ok
        mock_response = {
            "status": "ok",
            "manifest_name": "manifest:xs_mom",
            "universe_size": 3,
            "n_days": 252,
            "execution_mode": "expert",
            "metrics": {
                "val_sharpe": 1.2,
                "val_max_drawdown_pct": -10.0,
                "val_total_return_pct": 15.0,
                "holdout_sharpe": 0.8,
                "holdout_max_drawdown_pct": -8.0,
                "holdout_total_return_pct": 10.0,
            },
            "config": {"weighting": "equal_active_signals", "benchmark": None},
            "train_end": "2024-01-01",
            "val_end": "2024-06-01",
        }

        def mock_urlopen(req, timeout=300):
            body = json.loads(req.data)
            # the mutated manifest is what gets POSTed to /run_manifest
            assert body.get("execution_mode") == "expert"
            assert body.get("evaluator", {}).get("extras", {}).get("cost_bps") == 5
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
        (tmp_path / "results").mkdir(exist_ok=True)

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            run_loop(1)

        # Step 3: entry consumed, family state intact (REFINING + refine_count)
        data = backlog.load()
        entry = data["entries"][0]
        assert entry["status"] in ("done_adopted", "done_rejected")

        fam = load_knowledge()["families"][family_key]
        assert fam["lifecycle"] == FamilyLifecycle.REFINING
        assert fam["refine_count"] == 1
        # the new trial also refreshed last_manifest_spec
        assert fam["last_manifest_spec"]["execution_mode"] == "expert"
