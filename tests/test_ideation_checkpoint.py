"""
Tests for incremental checkpointing in the ideation pipeline (Issue #68).

The ideation pipeline must save partial progress after each stage so that
a killed run leaves a checkpoint file showing how far it got, rather than
writing nothing at all.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_ideation import (
    _run_ideation_v3,
    _save_ideation_log_v3,
    _save_checkpoint,
    _checkpoint_path,
    _IDEATION_LOG_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checkpoint_env(monkeypatch):
    """Set up temp dirs for checkpoint testing."""
    monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "DUMMY_KEY")
    monkeypatch.setenv("DUMMY_KEY", "noop")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        log_dir = td_path / "ideation_logs"
        log_dir.mkdir()
        monkeypatch.setattr("strategy_ideation._IDEATION_LOG_DIR", log_dir)
        monkeypatch.setattr("strategy_ideation._checkpoint_path", lambda: log_dir / "checkpoint.json")
        yield td_path, log_dir


# ---------------------------------------------------------------------------
# _save_checkpoint unit tests
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_writes_checkpoint_file(self, checkpoint_env):
        """_save_checkpoint writes a JSON file at the checkpoint path."""
        _, log_dir = checkpoint_env
        from strategy_ideation import _save_checkpoint

        _save_checkpoint(stage="brainstorm", data={"n_ideas": 5, "ideas": ["a", "b"]})

        cp_path = log_dir / "checkpoint.json"
        assert cp_path.exists()
        content = json.loads(cp_path.read_text())
        assert content["stage"] == "brainstorm"
        assert content["completed_stages"] == ["brainstorm"]
        assert "timestamp_utc" in content

    def test_accumulates_stages(self, checkpoint_env):
        """Each call appends to completed_stages."""
        _, log_dir = checkpoint_env
        from strategy_ideation import _save_checkpoint

        _save_checkpoint(stage="brainstorm", data={"n_ideas": 5})
        _save_checkpoint(stage="debate", data={"n_survived": 3})
        _save_checkpoint(stage="select", data={"n_proposals": 2})

        cp_path = log_dir / "checkpoint.json"
        content = json.loads(cp_path.read_text())
        assert content["completed_stages"] == ["brainstorm", "debate", "select"]
        assert content["stage"] == "select"

    def test_stores_stage_data(self, checkpoint_env):
        """Stage data is stored under the stage name key."""
        _, log_dir = checkpoint_env
        from strategy_ideation import _save_checkpoint

        _save_checkpoint(stage="brainstorm", data={"ideas": [1, 2, 3]})

        cp_path = log_dir / "checkpoint.json"
        content = json.loads(cp_path.read_text())
        assert content["brainstorm"]["ideas"] == [1, 2, 3]

    def test_overwrites_on_restart(self, checkpoint_env):
        """If called with stage='brainstorm' again (new run), checkpoint resets."""
        _, log_dir = checkpoint_env
        from strategy_ideation import _save_checkpoint

        _save_checkpoint(stage="brainstorm", data={"run": 1})
        _save_checkpoint(stage="debate", data={"run": 1})
        # New run starts
        _save_checkpoint(stage="brainstorm", data={"run": 2})

        cp_path = log_dir / "checkpoint.json"
        content = json.loads(cp_path.read_text())
        assert content["completed_stages"] == ["brainstorm"]
        assert content["brainstorm"]["run"] == 2


# ---------------------------------------------------------------------------
# Integration: checkpoint survives mid-pipeline kill
# ---------------------------------------------------------------------------


class TestPipelineCheckpoint:
    """Test that _run_ideation_v3 writes checkpoints as it progresses."""

    def test_checkpoint_after_brainstorm(self, checkpoint_env):
        """If debate raises, checkpoint still has brainstorm data."""
        _, log_dir = checkpoint_env
        import strategy_ideation as si

        ideas = [{"name": "test", "type": "code", "family": "x|Y", "one_line_rationale": "r"}]

        def mock_brainstorm(*a, **kw):
            si._save_checkpoint(stage="brainstorm", data={"n_ideas": len(ideas), "ideas": ideas})
            return ideas

        def mock_debate(*a, **kw):
            raise RuntimeError("debate crashed (simulated kill)")

        with patch.object(si, "_stage_brainstorm", side_effect=mock_brainstorm), \
             patch.object(si, "_stage_debate", side_effect=mock_debate):
            result = _run_ideation_v3(
                {"families": {}}, {}, [], MagicMock(), 5, False
            )

        # Pipeline falls back (returns None)
        assert result is None

        # But checkpoint exists with brainstorm data
        cp_path = log_dir / "checkpoint.json"
        assert cp_path.exists()
        content = json.loads(cp_path.read_text())
        assert "brainstorm" in content["completed_stages"]
        assert content["brainstorm"]["n_ideas"] == 1

    def test_checkpoint_after_debate(self, checkpoint_env):
        """If select raises, checkpoint has brainstorm + debate data."""
        _, log_dir = checkpoint_env
        import strategy_ideation as si

        ideas = [{"name": "test", "type": "code", "family": "x|Y", "one_line_rationale": "r"}]
        debate_results = [{"index": 0, "survive": True, "attack": "a", "rebuttal": "r", "variation": "v", "reason": "ok"}]
        judge_report = []

        def mock_brainstorm(*a, **kw):
            si._save_checkpoint(stage="brainstorm", data={"n_ideas": 1})
            return ideas

        def mock_debate(ideas_list):
            si._save_checkpoint(stage="debate", data={"n_survived": 1})
            return debate_results, judge_report

        def mock_select(*a, **kw):
            raise RuntimeError("select crashed (simulated kill)")

        with patch.object(si, "_stage_brainstorm", side_effect=mock_brainstorm), \
             patch.object(si, "_stage_debate", side_effect=mock_debate), \
             patch.object(si, "_stage_select_v3", side_effect=mock_select):
            result = _run_ideation_v3(
                {"families": {}}, {}, [], MagicMock(), 5, False
            )

        assert result is None
        cp_path = log_dir / "checkpoint.json"
        assert cp_path.exists()
        content = json.loads(cp_path.read_text())
        assert content["completed_stages"] == ["brainstorm", "debate"]

    def test_full_pipeline_writes_final_log(self, checkpoint_env):
        """Successful pipeline writes final log AND updates checkpoint to 'complete'."""
        _, log_dir = checkpoint_env
        import strategy_ideation as si

        ideas = [{"name": "test", "type": "code", "family": "x|Y", "one_line_rationale": "r"}]
        debate_results = [{"index": 0, "survive": True, "attack": "a", "rebuttal": "r", "variation": "v", "reason": "ok"}]
        judge_report = []

        mock_manifest = MagicMock()
        mock_manifest.execution_mode = "expert"
        mock_manifest.name = "test_m"
        mock_manifest.to_dict.return_value = {
            "name": "test_m",
            "code_b64": "ZGVmIGdlbmVyYXRlX3dlaWdodHMoZGYpOgogICAgcmV0dXJuIG5wLm9uZXMobGVuKGRmKSk=",
            "data_sources": [],
            "model_artifacts": [],
            "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
            "evaluator": {"type": "custom", "metrics": ["sharpe"]},
            "execution_mode": "expert",
        }
        mock_manifest.data_sources = []

        def mock_brainstorm(*a, **kw):
            si._save_checkpoint(stage="brainstorm", data={"n_ideas": 1})
            return ideas

        def mock_debate(ideas_list):
            si._save_checkpoint(stage="debate", data={"n_survived": 1})
            return debate_results, judge_report

        def mock_select(*a, **kw):
            si._save_checkpoint(stage="select", data={"n_manifests": 1})
            return [mock_manifest], False

        mock_backlog = MagicMock()
        mock_backlog.add_entry.return_value = (True, "entry_001")

        with patch.object(si, "_stage_brainstorm", side_effect=mock_brainstorm), \
             patch.object(si, "_stage_debate", side_effect=mock_debate), \
             patch.object(si, "_stage_select_v3", side_effect=mock_select), \
             patch.object(si, "_syntax_preflight_with_fix", return_value=(True, "", 0)), \
             patch.object(si, "_runtime_preflight_with_fix", return_value=(True, "", 0)), \
             patch.object(si, "_get_killed_families", return_value={}):
            result = _run_ideation_v3(
                {"families": {}}, {}, [], mock_backlog, 5, False
            )

        assert result == ["entry_001"]

        # Checkpoint should be marked complete
        cp_path = log_dir / "checkpoint.json"
        assert cp_path.exists()
        content = json.loads(cp_path.read_text())
        assert content["stage"] == "complete"
