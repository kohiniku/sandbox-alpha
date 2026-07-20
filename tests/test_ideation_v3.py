"""
Tests for strategy_ideation.py V3 — manifest-emitting 3-stage pipeline.
All LLM calls mocked, no network.
"""

import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_ideation import (
    _stage_select_v3,
    _run_ideation_v3,
    _build_brainstorm_prompt,
    _MANIFEST_JSON_SCHEMA,
    _EXPERT_MODE_CATALOG,
    run,
)
from autonomous_loop import STRATEGY_TEMPLATES
from manifest import StrategyManifest, ManifestValidationError

# ---------------------------------------------------------------------------
# Helper: build a valid structured manifest dict
# ---------------------------------------------------------------------------

def _structured_manifest_dict(name="test_strategy"):
    """Build a minimal valid structured manifest dict."""
    code = b"def generate_signals(df):\n    import pandas as pd\n    return pd.Series(0, index=df.index)\n"
    return {
        "name": name,
        "code_b64": base64.b64encode(code).decode("ascii"),
        "data_sources": [
            {"type": "ohlcv", "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "start": "2020-01-01"}
        ],
        "model_artifacts": [],
        "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
        "evaluator": {
            "type": "portfolio",
            "metrics": ["sharpe", "ir", "turnover"],
            "benchmark": "SPY",
        },
        "execution_mode": "structured",
        "priority": 0.85,
        "source": {"kind": "paper", "ref": "test-paper.md"},
    }


def _expert_manifest_dict(name="rl_portfolio_v1"):
    """Build a minimal valid expert manifest dict."""
    code = b"import torch\nimport numpy as np\ndef generate_weights(df):\n    return np.ones(len(df))\n"
    return {
        "name": name,
        "code_b64": base64.b64encode(code).decode("ascii"),
        "data_sources": [
            {"type": "ohlcv", "universe": ["SPY", "TLT", "GLD", "USO", "IWM", "QQQ"], "start": "2018-01-01"}
        ],
        "model_artifacts": [{"name": "timesfm-base", "revision": "v1.0"}],
        "compute": {"mode": "inference", "budget_seconds": 120, "gpu": True},
        "evaluator": {
            "type": "custom",
            "metrics": ["sharpe", "ir", "cvar_95", "factor_exposure"],
            "benchmark": "SPY",
            "extras": {"custom_evaluator": "rl_episode_sharpe"},
        },
        "execution_mode": "expert",
        "priority": 0.90,
        "source": {"kind": "paper", "ref": "rl-portfolio-paper.md"},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def v2_knowledge():
    return {
        "rejected": [],
        "adopted": [],
        "families": {
            "momentum|AAPL": {"n_trials": 5, "best_val_sharpe": 0.8},
            "mean_reversion|MSFT": {"n_trials": 1, "best_val_sharpe": -0.2},
            "rsi|SPY": {"n_trials": 0, "best_val_sharpe": -999.0},
        },
        "near_misses": [
            {"strategy": "momentum", "symbol": "NVDA", "params": {"lookback": 20, "hold_period": 5},
             "val_sharpe": 1.2, "deflated_threshold": 1.0, "failed_gate": "holdout"},
            {"strategy": "mean_reversion", "symbol": "GOOGL", "params": {"window": 30, "threshold": 2.0},
             "val_sharpe": 0.9, "deflated_threshold": 0.8, "failed_gate": "holdout"},
        ],
        "errors": [
            {"hypothesis": {"description": "bad strategy"}, "evaluation": {"error_type": "code", "error": "KeyError: 'close'"}},
        ],
        "iterations": 10,
    }


@pytest.fixture
def mock_env(monkeypatch):
    """Ensure no real API keys leak; set dummy paths."""
    monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "DUMMY_KEY")
    monkeypatch.setenv("DUMMY_KEY", "noop")
    monkeypatch.setenv("IDEATION_V3", "1")

    import strategy_ideation

    with tempfile.TemporaryDirectory() as td:
        bp = Path(td) / "research"
        bp.mkdir()
        monkeypatch.setenv("RESEARCH_DIRS", str(bp))
        monkeypatch.setenv("KNOWLEDGE_PATH", str(Path(td) / "knowledge.json"))
        monkeypatch.setenv("BACKLOG_PATH", str(Path(td) / "backlog.json"))
        yield td


# ---------------------------------------------------------------------------
# Stage 3 V3 Tests
# ---------------------------------------------------------------------------

class TestStageSelectV3:
    """Tests for _stage_select_v3 manifest emission."""

    def test_two_valid_manifests(self):
        """Stage 3 returns 2 valid manifests (1 structured, 1 expert) → both accepted."""
        import strategy_ideation as si

        surviving_ideas = [
            {"name": "vol_adaptive", "type": "code", "family": "mean_reversion|SPY",
             "one_line_rationale": "volatility scaling"},
            {"name": "rl_agent", "type": "code", "family": "strategy|QQQ",
             "one_line_rationale": "RL portfolio"},
        ]

        debate_results = [
            {"index": 0, "survive": True, "attack": "ok", "rebuttal": "ok", "variation": "", "reason": "ok"},
            {"index": 1, "survive": True, "attack": "risky", "rebuttal": "testable", "variation": "", "reason": "ok"},
        ]

        mock_manifests_raw = {
            "manifests": [
                _structured_manifest_dict("vol_adaptive_mr"),
                _expert_manifest_dict("rl_portfolio_v1"),
            ]
        }

        with patch.object(si, "_call_llm", return_value=mock_manifests_raw):
            manifests, fallback = _stage_select_v3(
                surviving_ideas, debate_results, {"families": {}}, STRATEGY_TEMPLATES, [], 5
            )

        assert len(manifests) == 2
        assert isinstance(manifests[0], StrategyManifest)
        assert isinstance(manifests[1], StrategyManifest)
        assert manifests[0].execution_mode == "structured"
        assert manifests[1].execution_mode == "expert"
        assert fallback is False

    def test_one_valid_one_invalid(self):
        """Stage 3 returns 1 valid + 1 invalid (missing required field) → valid accepted, invalid dropped."""
        import strategy_ideation as si

        surviving_ideas = [
            {"name": "good", "type": "code", "family": "rsi|SPY", "one_line_rationale": "test"},
            {"name": "bad", "type": "code", "family": "momentum|AAPL", "one_line_rationale": "test2"},
        ]

        debate_results = [
            {"index": 0, "survive": True, "attack": "ok", "rebuttal": "ok", "variation": "", "reason": "ok"},
            {"index": 1, "survive": True, "attack": "ok", "rebuttal": "ok", "variation": "", "reason": "ok"},
        ]

        # One valid, one missing 'name' field
        invalid = _structured_manifest_dict("good_idea")
        del invalid["name"]
        mock_manifests_raw = {
            "manifests": [
                _structured_manifest_dict("good_idea"),
                invalid,
            ]
        }

        with patch.object(si, "_call_llm", return_value=mock_manifests_raw):
            manifests, fallback = _stage_select_v3(
                surviving_ideas, debate_results, {"families": {}}, STRATEGY_TEMPLATES, [], 5
            )

        assert len(manifests) == 1
        assert manifests[0].name == "good_idea"

    def test_all_invalid(self):
        """Stage 3 returns all invalid manifests → raises with message about all failed validation."""
        import strategy_ideation as si

        surviving_ideas = [
            {"name": "bad1", "type": "code", "family": "rsi|SPY", "one_line_rationale": "test"},
            {"name": "bad2", "type": "code", "family": "momentum|AAPL", "one_line_rationale": "test2"},
        ]

        debate_results = [
            {"index": 0, "survive": True, "attack": "ok", "rebuttal": "ok", "variation": "", "reason": "ok"},
            {"index": 1, "survive": True, "attack": "ok", "rebuttal": "ok", "variation": "", "reason": "ok"},
        ]

        # Both invalid: missing code_b64
        invalid1 = _structured_manifest_dict("bad1")
        del invalid1["code_b64"]
        invalid2 = _expert_manifest_dict("bad2")
        del invalid2["code_b64"]

        mock_manifests_raw = {"manifests": [invalid1, invalid2]}

        with patch.object(si, "_call_llm", return_value=mock_manifests_raw):
            with pytest.raises(RuntimeError, match="select v3 failed"):
                _stage_select_v3(
                    surviving_ideas, debate_results, {"families": {}}, STRATEGY_TEMPLATES, [], 5
                )

    def test_retry_on_empty(self):
        """Empty manifests list → retry once, then fail."""
        import strategy_ideation as si

        surviving_ideas = [
            {"name": "test", "type": "code", "family": "rsi|SPY", "one_line_rationale": "test"},
        ]
        debate_results = [
            {"index": 0, "survive": True, "attack": "ok", "rebuttal": "ok", "variation": "", "reason": "ok"},
        ]

        call_count = [0]

        def mock_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"manifests": []}  # empty on first try
            else:
                # valid on retry
                return {"manifests": [_structured_manifest_dict("test_ok")]}

        with patch.object(si, "_call_llm", side_effect=mock_call):
            manifests, fallback = _stage_select_v3(
                surviving_ideas, debate_results, {"families": {}}, STRATEGY_TEMPLATES, [], 5
            )

        assert len(manifests) == 1
        assert manifests[0].name == "test_ok"
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Brainstorm prompt V3 mandates
# ---------------------------------------------------------------------------

class TestBrainstormV3Mandates:
    """Verify brainstorm prompt contains universe and expert-mode mandates."""

    def test_prompt_has_universe_mandate(self, v2_knowledge):
        """Brainstorm prompt contains 'UNIVERSE of 5+' mandate."""
        messages = _build_brainstorm_prompt(v2_knowledge, STRATEGY_TEMPLATES, [])
        prompt_text = messages[1]["content"]
        assert "UNIVERSE of 5+" in prompt_text

    def test_prompt_has_expert_mode_mandate(self, v2_knowledge):
        """Brainstorm prompt contains expert-mode mandate."""
        messages = _build_brainstorm_prompt(v2_knowledge, STRATEGY_TEMPLATES, [])
        prompt_text = messages[1]["content"]
        assert "EXPERT-MODE" in prompt_text

    def test_prompt_has_expert_catalog(self, v2_knowledge):
        """Brainstorm prompt includes the expert mode catalog."""
        messages = _build_brainstorm_prompt(v2_knowledge, STRATEGY_TEMPLATES, [])
        prompt_text = messages[1]["content"]
        assert "RL/deep learning" in prompt_text
        assert "Cross-sectional" in prompt_text
        assert "Regime detection" in prompt_text

    def test_prompt_has_lookahead_warning(self):
        """Debate risk prompt includes expert lookahead warning."""
        from strategy_ideation import _stage_debate
        import strategy_ideation as si

        ideas = [
            {"name": "test", "type": "code", "family": "rsi|SPY", "one_line_rationale": "test"},
        ]

        # We test indirectly: monkey-patch _call_llm to capture the risk prompt
        captured_prompts = []

        def capture_call(messages, **kwargs):
            captured_prompts.append(messages[1]["content"])
            if len(captured_prompts) == 1:
                return {"risk_report": [{"index": 0, "attack": "test"}]}
            elif len(captured_prompts) == 2:
                return {"quant_report": [{"index": 0, "rebuttal": "ok", "variation": ""}]}
            else:
                return {"judge_report": [{"index": 0, "survive": True, "reason": "ok"}]}

        with patch.object(si, "_call_llm", side_effect=capture_call):
            _stage_debate(ideas)

        risk_prompt = captured_prompts[0]
        assert "lookahead" in risk_prompt
        assert "train/val/holdout" in risk_prompt


# ---------------------------------------------------------------------------
# IDEATION_V3 toggle + fallback
# ---------------------------------------------------------------------------

class TestIdeationV3Toggle:
    """IDEATION_V3=0 falls back to v2 pipeline."""

    def test_v3_disabled_uses_v2(self, mock_env, v2_knowledge, monkeypatch):
        """IDEATION_V3=0 → bypasses v3 pipeline, uses v2."""
        monkeypatch.setenv("IDEATION_V3", "0")
        monkeypatch.setenv("IDEATION_V2", "1")
        import strategy_ideation as si

        # Mock both v2 and v3 to verify v2 is called
        v3_called = [False]
        v2_called = [False]

        def mock_v3(*args, **kwargs):
            v3_called[0] = True
            return ["v3_result"]

        def mock_v2(*args, **kwargs):
            v2_called[0] = True
            return ["v2_result"]

        def mock_save_log(*args, **kwargs):
            pass

        with patch.object(si, "_run_ideation_v3", side_effect=mock_v3):
            with patch.object(si, "_run_ideation_v2", side_effect=mock_v2):
                result = run(max_proposals=3, dry_run=True)

        assert not v3_called[0], "v3 should NOT be called when IDEATION_V3=0"
        assert v2_called[0], "v2 SHOULD be called when IDEATION_V3=0"
        assert result == ["v2_result"]

    def test_v3_enabled_runs_v3(self, mock_env, v2_knowledge, monkeypatch):
        """IDEATION_V3=1 (default) → runs v3 pipeline."""
        monkeypatch.setenv("IDEATION_V3", "1")
        import strategy_ideation as si

        v3_called = [False]
        v2_called = [False]

        def mock_v3(*args, **kwargs):
            v3_called[0] = True
            return ["v3_result"]

        def mock_v2(*args, **kwargs):
            v2_called[0] = True
            return ["v2_result"]

        with patch.object(si, "_run_ideation_v3", side_effect=mock_v3):
            with patch.object(si, "_run_ideation_v2", side_effect=mock_v2):
                result = run(max_proposals=3, dry_run=True)

        assert v3_called[0], "v3 SHOULD be called when IDEATION_V3=1"
        assert not v2_called[0], "v2 should NOT be called when v3 succeeds"
        assert result == ["v3_result"]


# ---------------------------------------------------------------------------
# Happy path: full V3 pipeline
# ---------------------------------------------------------------------------

class TestHappyPathV3:
    """Full V3 pipeline with mocked LLM — manifests accepted to backlog."""

    def test_happy_path_writes_manifest_entries(self, mock_env, v2_knowledge):
        """End-to-end: 2 valid manifests → both added to backlog with type='manifest'."""
        import strategy_ideation as si

        brainstorm_ideas = [
            {"name": "vol_adaptive", "type": "code", "family": "mean_reversion|SPY",
             "one_line_rationale": "vol scaling"},
            {"name": "rl_agent", "type": "code", "family": "strategy|QQQ",
             "one_line_rationale": "RL portfolio"},
        ]

        debate_results = [
            {"index": 0, "survive": True, "attack": "ok", "rebuttal": "ok", "variation": "", "reason": "ok"},
            {"index": 1, "survive": True, "attack": "risky", "rebuttal": "testable", "variation": "", "reason": "ok"},
        ]

        manifests_raw = {"manifests": [_structured_manifest_dict("test_structured"), _expert_manifest_dict("test_expert")]}

        def mock_select_v3(surviving, debate, knowledge, templates, research_docs, max_p):
            return [StrategyManifest.from_dict(m) for m in manifests_raw["manifests"]], False

        def mock_save_log(*args, **kwargs):
            pass

        def mock_preflight(*args, **kwargs):
            return None, "skipped"  # runner unavailable

        # Use a real backlog, not a MagicMock
        from backlog import Backlog
        backlog = Backlog(Path(os.environ["BACKLOG_PATH"]))

        with patch.object(si, "_stage_brainstorm", return_value=brainstorm_ideas):
            with patch.object(si, "_stage_debate", return_value=(debate_results, [{"index": 0, "survive": True}, {"index": 1, "survive": True}])):
                with patch.object(si, "_stage_select_v3", side_effect=mock_select_v3):
                    with patch.object(si, "_save_ideation_log_v3", side_effect=mock_save_log):
                        with patch.object(si, "_preflight_manifest", side_effect=mock_preflight):
                            result = _run_ideation_v3(
                                v2_knowledge, STRATEGY_TEMPLATES, [], backlog, 5, dry_run=False
                            )

        assert result is not None
        assert len(result) == 2

        # Check backlog
        bp = Path(os.environ["BACKLOG_PATH"])
        data = json.loads(bp.read_text())
        manifest_entries = [e for e in data["entries"] if e["type"] == "manifest"]
        assert len(manifest_entries) == 2
        modes = {e["spec"]["execution_mode"] for e in manifest_entries}
        assert modes == {"structured", "expert"}
        for entry in manifest_entries:
            assert entry["status"] == "pending"
            assert "code_b64" in entry["spec"]
            assert "data_sources" in entry["spec"]
            assert len(entry["spec"]["data_sources"]) >= 1

    def test_v3_fallback_to_v2_on_all_invalid(self, mock_env, v2_knowledge, monkeypatch):
        """When all manifests fail validation, v3 returns None → falls back to v2."""
        monkeypatch.setenv("IDEATION_V3", "1")
        monkeypatch.setenv("IDEATION_V2", "1")
        import strategy_ideation as si

        # v3 returns None (all manifests failed)
        def mock_v3(*args, **kwargs):
            return None

        # v2 should then be called and succeed
        v2_called = [False]

        def mock_v2(*args, **kwargs):
            v2_called[0] = True
            return ["v2_fallback_result"]

        with patch.object(si, "_run_ideation_v3", side_effect=mock_v3):
            with patch.object(si, "_run_ideation_v2", side_effect=mock_v2):
                result = run(max_proposals=3, dry_run=True)

        assert v2_called[0]
        assert result == ["v2_fallback_result"]


# ---------------------------------------------------------------------------
# Backlog manifest entry shape
# ---------------------------------------------------------------------------

class TestManifestBacklogEntry:
    """Verify the shape of manifest backlog entries."""

    def test_manifest_entry_has_required_fields(self, mock_env, v2_knowledge):
        """Manifest backlog entry has type='manifest' and spec is a valid manifest dict."""
        import strategy_ideation as si

        brainstorm_ideas = [
            {"name": "test", "type": "code", "family": "rsi|SPY", "one_line_rationale": "test"},
        ]
        debate_results = [
            {"index": 0, "survive": True, "attack": "ok", "rebuttal": "ok", "variation": "", "reason": "ok"},
        ]
        manifest_dict = _structured_manifest_dict("test_entry")

        def mock_select_v3(*args, **kwargs):
            return [StrategyManifest.from_dict(manifest_dict)], False

        def mock_save_log(*args, **kwargs):
            pass

        def mock_preflight(*args, **kwargs):
            return None, "skipped"

        from backlog import Backlog
        from pathlib import Path
        backlog = Backlog(Path(os.environ["BACKLOG_PATH"]))

        with patch.object(si, "_stage_brainstorm", return_value=brainstorm_ideas):
            with patch.object(si, "_stage_debate", return_value=(debate_results, [{"index": 0, "survive": True}])):
                with patch.object(si, "_stage_select_v3", side_effect=mock_select_v3):
                    with patch.object(si, "_save_ideation_log_v3", side_effect=mock_save_log):
                        with patch.object(si, "_preflight_manifest", side_effect=mock_preflight):
                            result = _run_ideation_v3(
                                v2_knowledge, STRATEGY_TEMPLATES, [], backlog, 5, dry_run=False
                            )

        bp = Path(os.environ["BACKLOG_PATH"])
        data = json.loads(bp.read_text())
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["type"] == "manifest"
        assert entry["status"] == "pending"
        assert isinstance(entry["spec"], dict)
        assert entry["spec"]["name"] == "test_entry"
        assert entry["spec"]["execution_mode"] == "structured"
        assert "code_b64" in entry["spec"]
        assert "data_sources" in entry["spec"]


# ---------------------------------------------------------------------------
# IDEATION_V3=0 uses v2 (non-manifest) code path
# ---------------------------------------------------------------------------

class TestV3DisabledUsesV2:
    """IDEATION_V3=0 uses the old v2 code path (non-manifest proposals)."""

    def test_v3_disabled_goes_to_v2_directly(self, mock_env, v2_knowledge, monkeypatch):
        """IDEATION_V3=0 → v3 not called, v2 is called directly (no v2 fallback from v3)."""
        monkeypatch.setenv("IDEATION_V3", "0")
        monkeypatch.setenv("IDEATION_V2", "1")
        import strategy_ideation as si

        v3_called = [False]

        def mock_v3(*args, **kwargs):
            v3_called[0] = True
            return ["v3"]

        v2_proposals = [
            {
                "type": "param",
                "priority": 0.8,
                "source": {"kind": "paper", "ref": "test.md"},
                "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
                "eval_plan": {"extra_criteria": []},
            }
        ]

        with patch.object(si, "_run_ideation_v3", side_effect=mock_v3):
            with patch.object(si, "_call_llm", return_value={"proposals": v2_proposals}):
                result = run(max_proposals=3, dry_run=True)

        assert not v3_called[0]
        assert len(result) == 1  # v1 single-call path picks it up
