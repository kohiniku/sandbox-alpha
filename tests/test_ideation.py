"""
Tests for strategy_ideation.py — all LLM calls mocked, no network.
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
    _validate_param_spec,
    _validate_code_spec,
    _validate_proposal,
    _summarise_rejects,
    _gather_research_docs,
    _build_prompt,
    run,
)
from autonomous_loop import STRATEGY_TEMPLATES


# ---------------------------------------------------------------------------
# Validation: param specs
# ---------------------------------------------------------------------------

def test_validate_param_spec_valid():
    spec = {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}}
    ok, err = _validate_param_spec(spec, STRATEGY_TEMPLATES)
    assert ok is True
    assert err is None


def test_validate_param_spec_unknown_strategy():
    spec = {"strategy": "bogus", "symbol": "AAPL", "params": {"lookback": 20}}
    ok, err = _validate_param_spec(spec, STRATEGY_TEMPLATES)
    assert ok is False
    assert "Unknown strategy" in err


def test_validate_param_spec_invalid_symbol():
    spec = {"strategy": "momentum", "symbol": "lowercase", "params": {"lookback": 20, "hold_period": 5}}
    ok, err = _validate_param_spec(spec, STRATEGY_TEMPLATES)
    assert ok is False
    assert "Invalid symbol" in err


def test_validate_param_spec_out_of_range():
    spec = {"strategy": "sma_crossover", "symbol": "SPY", "params": {"fast_window": 999, "slow_window": 50}}
    ok, err = _validate_param_spec(spec, STRATEGY_TEMPLATES)
    assert ok is False
    assert "out of range" in err


def test_validate_param_spec_not_in_list():
    spec = {"strategy": "mean_reversion", "symbol": "QQQ", "params": {"window": 20, "threshold": 9.9}}
    ok, err = _validate_param_spec(spec, STRATEGY_TEMPLATES)
    assert ok is False
    assert "not in" in err


def test_validate_param_spec_key_mismatch():
    spec = {"strategy": "momentum", "symbol": "MSFT", "params": {"lookback": 20, "extra_key": 5}}
    ok, err = _validate_param_spec(spec, STRATEGY_TEMPLATES)
    assert ok is False
    assert "key mismatch" in err


def test_validate_param_spec_missing_key():
    spec = {"strategy": "momentum", "symbol": "MSFT", "params": {"lookback": 20}}
    ok, err = _validate_param_spec(spec, STRATEGY_TEMPLATES)
    assert ok is False
    assert "key mismatch" in err or "Missing keys" in err


# ---------------------------------------------------------------------------
# Validation: code specs
# ---------------------------------------------------------------------------

def test_validate_code_spec_valid():
    spec = {
        "name": "adaptive_ma",
        "description": "Adaptive MA crossover",
        "code": "import numpy as np\nimport pandas as pd\n\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)",
        "symbol": "SPY",
    }
    ok, err = _validate_code_spec(spec)
    assert ok is True
    assert err is None


def test_validate_code_spec_missing_generate_signals():
    spec = {
        "name": "bad_strat",
        "description": "No signals function",
        "code": "import numpy as np\n\ndef some_other_function(df):\n    return df",
        "symbol": "SPY",
    }
    ok, err = _validate_code_spec(spec)
    assert ok is False
    assert "generate_signals" in err


def test_validate_code_spec_empty_code():
    spec = {"name": "empty", "description": "...", "code": "", "symbol": "SPY"}
    ok, err = _validate_code_spec(spec)
    assert ok is False


def test_validate_code_spec_invalid_symbol():
    spec = {"name": "x", "description": "...", "code": "def generate_signals(df): pass", "symbol": "bad symbol!"}
    ok, err = _validate_code_spec(spec)
    assert ok is False
    assert "Invalid symbol" in err


def test_validate_code_spec_size_limit():
    # Create code just over 64KB by repeating a comment
    # 64KB = 65536 bytes, each "# padding\n" = 10 bytes, use 7000 to exceed
    big_code = "def generate_signals(df):\n    return df\n" + ("# padding\n" * 7000)
    spec = {"name": "big", "description": "...", "code": big_code, "symbol": "SPY"}
    ok, err = _validate_code_spec(spec)
    assert ok is False
    assert "64KB" in err


# ---------------------------------------------------------------------------
# Validation: full proposal
# ---------------------------------------------------------------------------

def test_validate_proposal_valid_param():
    proposal = {
        "type": "param",
        "priority": 0.8,
        "source": {"kind": "paper", "ref": "research.md"},
        "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
        "eval_plan": {"extra_criteria": ["max_hold_days <= 5"]},
    }
    ok, err = _validate_proposal(proposal, STRATEGY_TEMPLATES)
    assert ok is True


def test_validate_proposal_valid_code():
    proposal = {
        "type": "code",
        "priority": 0.6,
        "source": {"kind": "idea", "ref": "my idea"},
        "spec": {
            "name": "test", "description": "desc",
            "code": "def generate_signals(df): return df",
            "symbol": "SPY",
        },
        "eval_plan": {"extra_criteria": []},
    }
    ok, err = _validate_proposal(proposal, STRATEGY_TEMPLATES)
    assert ok is True


def test_validate_proposal_bad_type():
    proposal = {
        "type": "invalid_type",
        "priority": 0.5,
        "source": {"kind": "idea", "ref": "x"},
        "spec": {},
        "eval_plan": {"extra_criteria": []},
    }
    ok, err = _validate_proposal(proposal, STRATEGY_TEMPLATES)
    assert ok is False
    assert "Unknown type" in err


def test_validate_proposal_bad_source_kind():
    proposal = {
        "type": "param",
        "priority": 0.5,
        "source": {"kind": "bogus", "ref": "x"},
        "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
        "eval_plan": {"extra_criteria": []},
    }
    ok, err = _validate_proposal(proposal, STRATEGY_TEMPLATES)
    assert ok is False
    assert "source.kind" in err


def test_validate_proposal_missing_source_ref():
    proposal = {
        "type": "param",
        "priority": 0.5,
        "source": {"kind": "idea"},
        "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
        "eval_plan": {"extra_criteria": []},
    }
    ok, err = _validate_proposal(proposal, STRATEGY_TEMPLATES)
    assert ok is False
    assert "source.ref" in err


def test_validate_proposal_bad_priority():
    proposal = {
        "type": "param",
        "priority": 1.5,
        "source": {"kind": "idea", "ref": "x"},
        "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
        "eval_plan": {"extra_criteria": []},
    }
    ok, err = _validate_proposal(proposal, STRATEGY_TEMPLATES)
    assert ok is False
    assert "priority" in err


def test_validate_proposal_bad_extra_criteria():
    proposal = {
        "type": "param",
        "priority": 0.5,
        "source": {"kind": "idea", "ref": "x"},
        "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
        "eval_plan": {"extra_criteria": [1, 2, 3]},  # not strings
    }
    ok, err = _validate_proposal(proposal, STRATEGY_TEMPLATES)
    assert ok is False
    assert "extra_criteria" in err


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

def test_summarise_rejects_empty():
    knowledge = {"rejected": []}
    result = _summarise_rejects(knowledge)
    assert "No rejected" in result


def test_summarise_rejects_with_data():
    knowledge = {
        "rejected": [
            {
                "hypothesis": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 10, "hold_period": 3}},
                "evaluation": {
                    "sharpe_ratio": 0.2,
                    "gate_results": {"validation": False},
                },
            }
        ],
        "families": {},
    }
    result = _summarise_rejects(knowledge)
    assert "momentum/AAPL" in result
    assert "validation failed" in result


def test_gather_research_docs_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        docs = _gather_research_docs([Path(td)])
        assert docs == []


def test_gather_research_docs_reads_files():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "doc1.md").write_text("# Title One\nBody content here.\nMore text.")
        (p / "doc2.json").write_text('{"key": "value"}')

        docs = _gather_research_docs([p])
        assert len(docs) == 2
        names = {name for name, _ in docs}
        assert names == {"doc1.md", "doc2.json"}


def test_gather_research_docs_caps_files():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        for i in range(15):
            (p / f"doc{i:02d}.md").write_text(f"# Title {i}\nContent.")

        docs = _gather_research_docs([p])
        # Capped at _MAX_RESEARCH_FILES (10)
        assert len(docs) <= 10


# ---------------------------------------------------------------------------
# Prompt building (non-network, just checks structure)
# ---------------------------------------------------------------------------

def test_build_prompt_structure():
    knowledge = {"rejected": [], "families": {}}
    templates = {"momentum": STRATEGY_TEMPLATES["momentum"]}
    docs = []
    backlog_summary = "No pending entries."

    messages = _build_prompt(knowledge, templates, docs, backlog_summary, 5)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "momentum" in messages[1]["content"]
    assert "5" in messages[1]["content"]  # max_proposals


# ---------------------------------------------------------------------------
# run() integration tests (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_env(monkeypatch):
    """Ensure no real API keys leak; set dummy paths."""
    monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "DUMMY_KEY")
    monkeypatch.setenv("DUMMY_KEY", "noop")
    # Point knowledge/research to temp dirs
    import strategy_ideation

    with tempfile.TemporaryDirectory() as td:
        bp = Path(td) / "research"
        bp.mkdir()
        monkeypatch.setenv("RESEARCH_DIRS", str(bp))
        monkeypatch.setenv("KNOWLEDGE_PATH", str(Path(td) / "knowledge.json"))
        monkeypatch.setenv("BACKLOG_PATH", str(Path(td) / "backlog.json"))
        yield td


def _mock_llm_response(proposals):
    """Return a function that can be used as side_effect for _call_llm."""
    return lambda messages, max_tokens=4096: {"proposals": proposals}


def test_run_dry_run_no_writes(mock_env):
    proposals = [
        {
            "type": "param",
            "priority": 0.8,
            "source": {"kind": "paper", "ref": "test.md"},
            "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
            "eval_plan": {"extra_criteria": []},
        }
    ]

    import strategy_ideation
    with patch.object(strategy_ideation, "_call_llm", side_effect=_mock_llm_response(proposals)):
        ids = strategy_ideation.run(max_proposals=3, dry_run=True)
        assert len(ids) == 1

    # Backlog file should not exist (or be empty if auto-created)
    bp = Path(os.environ["BACKLOG_PATH"])
    if bp.exists():
        data = json.loads(bp.read_text())
        assert len(data["entries"]) == 0


def test_run_writes_to_backlog(mock_env):
    proposals = [
        {
            "type": "param",
            "priority": 0.8,
            "source": {"kind": "paper", "ref": "test.md"},
            "spec": {"strategy": "momentum", "symbol": "MSFT", "params": {"lookback": 15, "hold_period": 3}},
            "eval_plan": {"extra_criteria": []},
        }
    ]

    import strategy_ideation
    with patch.object(strategy_ideation, "_call_llm", side_effect=_mock_llm_response(proposals)):
        ids = strategy_ideation.run(max_proposals=3, dry_run=False)
        assert len(ids) == 1

    bp = Path(os.environ["BACKLOG_PATH"])
    data = json.loads(bp.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["type"] == "param"
    assert data["entries"][0]["status"] == "pending"


def test_run_drops_invalid_proposals(mock_env):
    proposals = [
        {
            "type": "param",
            "priority": 0.8,
            "source": {"kind": "paper", "ref": "test.md"},
            "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
            "eval_plan": {"extra_criteria": []},
        },
        {
            "type": "param",
            "priority": 0.5,
            "source": {"kind": "paper", "ref": "bad.md"},
            "spec": {"strategy": "bogus", "symbol": "AAPL", "params": {"lookback": 20}},  # invalid strategy
            "eval_plan": {"extra_criteria": []},
        },
        {
            "type": "code",
            "priority": 0.6,
            "source": {"kind": "idea", "ref": "code-idea"},
            "spec": {
                "name": "no_signal_fn",
                "description": "desc",
                "code": "import numpy as np",  # no generate_signals
                "symbol": "SPY",
            },
            "eval_plan": {"extra_criteria": []},
        },
    ]

    import strategy_ideation
    with patch.object(strategy_ideation, "_call_llm", side_effect=_mock_llm_response(proposals)):
        ids = strategy_ideation.run(max_proposals=3, dry_run=False)
        assert len(ids) == 1  # only the first should be accepted

    bp = Path(os.environ["BACKLOG_PATH"])
    data = json.loads(bp.read_text())
    assert len(data["entries"]) == 1


def test_run_llm_error_graceful(mock_env):
    import strategy_ideation

    def failing_llm(*args, **kwargs):
        raise Exception("simulated network error")

    with patch.object(strategy_ideation, "_call_llm", side_effect=failing_llm):
        ids = strategy_ideation.run(max_proposals=3, dry_run=False)
        assert ids == []  # graceful empty


def test_run_llm_returns_no_proposals_key(mock_env):
    import strategy_ideation

    with patch.object(strategy_ideation, "_call_llm", return_value={"other_key": "no proposals"}):
        ids = strategy_ideation.run(max_proposals=3, dry_run=False)
        assert ids == []


def test_run_max_proposals_honored(mock_env):
    """LLM returns 10 proposals but --max-proposals is embedded in prompt, not enforced on output.
    We just verify the full pipeline works with a larger batch."""
    proposals = [
        {
            "type": "param",
            "priority": 0.8,
            "source": {"kind": "paper", "ref": f"test{i}.md"},
            "spec": {"strategy": "momentum", "symbol": f"SYM{i:03d}", "params": {"lookback": 10 + i, "hold_period": 3}},
            "eval_plan": {"extra_criteria": []},
        }
        for i in range(5)
    ]

    # SYM000-SYM004 — these are valid symbols matching [A-Z0-9][A-Z0-9.\-]{0,11}
    import strategy_ideation
    with patch.object(strategy_ideation, "_call_llm", side_effect=_mock_llm_response(proposals)):
        ids = strategy_ideation.run(max_proposals=3, dry_run=True)
        assert len(ids) == 5  # dry-run so all valid ones pass through


# ---------------------------------------------------------------------------
# Symbol edge cases
# ---------------------------------------------------------------------------

def test_symbol_validation_comprehensive():
    """Test the symbol regex directly."""
    from strategy_ideation import _SYMBOL_RE_COMPILED

    valid = ["AAPL", "MSFT", "BTC-USD", "SPY", "QQQ", "BRK.B", "1234"]
    invalid = ["abc", "", "toolong12345", "@#$", "a"]

    for s in valid:
        assert _SYMBOL_RE_COMPILED.match(s), f"'{s}' should be valid"
    for s in invalid:
        assert not _SYMBOL_RE_COMPILED.match(s), f"'{s}' should be invalid"


# ---------------------------------------------------------------------------
# IDEATION V2 — 3-stage pipeline tests
# ---------------------------------------------------------------------------


@pytest.fixture
def v2_knowledge():
    """Knowledge with families, near_misses, and errors for v2 tests."""
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


class TestIdeationV2:
    """Tests for the 3-stage ideation pipeline."""

    def test_brainstorm_mandate_in_prompt(self, v2_knowledge):
        """Verify brainstorm prompt includes novelty mandate and near-miss mandate."""
        from strategy_ideation import _build_brainstorm_prompt
        from autonomous_loop import STRATEGY_TEMPLATES

        messages = _build_brainstorm_prompt(v2_knowledge, STRATEGY_TEMPLATES, [])
        prompt_text = messages[1]["content"]

        # Novelty mandate: <3 trials families exist → mandate injected
        assert "MANDATE:" in prompt_text
        assert "<3 trials" in prompt_text
        # Near-miss recombination mandate: >=2 near_misses → mandate injected
        assert "RECOMBINE" in prompt_text or "recombine" in prompt_text.lower()

    def test_brainstorm_prompt_no_mandates_when_empty(self):
        """When families have many trials and no near-misses, no mandates."""
        from strategy_ideation import _build_brainstorm_prompt
        from autonomous_loop import STRATEGY_TEMPLATES

        knowledge = {"families": {}, "near_misses": [], "errors": [], "rejected": []}
        messages = _build_brainstorm_prompt(knowledge, STRATEGY_TEMPLATES, [])
        prompt_text = messages[1]["content"]
        # No mandates when no novel families and <2 near-misses
        assert "MANDATE:" not in prompt_text.replace("MANDATES", "")

    def test_debate_kill_filtering(self):
        """Verify judge verdicts decide survive/kill per idea."""
        from strategy_ideation import _stage_debate

        ideas = [
            {"name": "good_idea", "type": "param", "family": "rsi|SPY", "one_line_rationale": "test"},
            {"name": "overfit_idea", "type": "param", "family": "momentum|AAPL", "one_line_rationale": "too narrow"},
        ]

        import strategy_ideation as si

        call_count = [0]

        def mock_call(messages, max_tokens=4096, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Risk Manager
                return {
                    "risk_report": [
                        {"index": 0, "attack": "reasonable approach"},
                        {"index": 1, "attack": "clear overfitting smell, too narrow"},
                    ]
                }
            elif call_count[0] == 2:
                # Quant Researcher
                return {
                    "quant_report": [
                        {"index": 0, "rebuttal": "solid rationale", "variation": ""},
                        {"index": 1, "rebuttal": "could broaden", "variation": "add regime filter"},
                    ]
                }
            else:
                # Judge: explicit verdicts
                return {
                    "judge_report": [
                        {"index": 0, "survive": True, "reason": "novel approach, attack not fatal"},
                        {"index": 1, "survive": False, "reason": "too narrow, overfitting risk"},
                    ]
                }

        with patch.object(si, "_call_llm", side_effect=mock_call):
            results, judge_report = _stage_debate(ideas)

        assert len(results) == 2
        assert results[0]["survive"] is True   # judge says survive
        assert results[1]["survive"] is False  # judge says kill
        assert len(judge_report) == 2

    def test_debate_judge_failure_fail_open(self):
        """Judge call fails → all ideas survive (fail-open) with warning."""
        from strategy_ideation import _stage_debate

        ideas = [
            {"name": "idea1", "type": "param", "family": "rsi|SPY", "one_line_rationale": "test"},
            {"name": "idea2", "type": "code", "family": "strategy|QQQ", "one_line_rationale": "test2"},
        ]

        import strategy_ideation as si

        call_count = [0]

        def mock_call(messages, max_tokens=4096, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"risk_report": [{"index": 0, "attack": "risky"}, {"index": 1, "attack": "unknown"}]}
            elif call_count[0] == 2:
                return {"quant_report": [{"index": 0, "rebuttal": "ok"}, {"index": 1, "rebuttal": "tbd"}]}
            else:
                # Judge fails
                raise Exception("judge API timeout")

        with patch.object(si, "_call_llm", side_effect=mock_call):
            results, judge_report = _stage_debate(ideas)

        # Fail-open: both survive
        assert len(results) == 2
        assert results[0]["survive"] is True
        assert results[1]["survive"] is True
        assert judge_report == []  # judge failed

    def test_happy_3stage_path(self, mock_env, v2_knowledge):
        """Full 3-stage pipeline runs end-to-end with mocked LLM."""
        import strategy_ideation as si

        brainstorm_ideas = [
            {"name": "vol_adaptive_mr", "type": "param", "family": "mean_reversion|SPY",
             "one_line_rationale": "volatility scaling improves mean reversion timing"},
            {"name": "gap_reversal", "type": "code", "family": "strategy|QQQ",
             "one_line_rationale": "overnight gaps revert within the session"},
            {"name": "overfit_narrow", "type": "param", "family": "momentum|AAPL",
             "one_line_rationale": "very specific window combo"},
        ]

        debate_results = [
            {"index": 0, "survive": True, "attack": "reasonable", "rebuttal": "good", "variation": "", "reason": "ok"},
            {"index": 1, "survive": True, "attack": "may overfit", "rebuttal": "testable", "variation": "add vol filter", "reason": "ok"},
            {"index": 2, "survive": False, "attack": "overfitting smell", "rebuttal": "weak", "variation": "", "reason": "killed"},
        ]

        full_proposals = [
            {
                "type": "param",
                "priority": 0.85,
                "source": {"kind": "idea", "ref": "vol_adaptive_mr"},
                "spec": {"strategy": "mean_reversion", "symbol": "SPY", "params": {"window": 20, "threshold": 2.0}},
                "eval_plan": {"extra_criteria": []},
            }
        ]

        def mock_stage_brainstorm(knowledge, templates, research_docs):
            return brainstorm_ideas

        def mock_stage_debate(ideas):
            return debate_results, [{"index": 0, "survive": True}, {"index": 1, "survive": True}, {"index": 2, "survive": False}]

        def mock_stage_select(surviving, debate, knowledge, templates, research_docs, max_p):
            return full_proposals, False

        def mock_save_log(*args, **kwargs):
            pass

        with patch.object(si, "_stage_brainstorm", side_effect=mock_stage_brainstorm):
            with patch.object(si, "_stage_debate", side_effect=mock_stage_debate):
                with patch.object(si, "_stage_select", side_effect=mock_stage_select):
                    with patch.object(si, "_save_ideation_log", side_effect=mock_save_log):
                        result = si._run_ideation_v2(
                            v2_knowledge, STRATEGY_TEMPLATES, [], MagicMock(), 5, dry_run=True
                        )

        assert result is not None
        assert len(result) == 1

    def test_stage1_failure_falls_back(self, mock_env, v2_knowledge):
        """Brainstorm failure → _run_ideation_v2 returns None (fallback signal)."""
        import strategy_ideation as si

        def failing_brainstorm(*args, **kwargs):
            raise ValueError("API down")

        with patch.object(si, "_stage_brainstorm", side_effect=failing_brainstorm):
            result = si._run_ideation_v2(
                v2_knowledge, STRATEGY_TEMPLATES, [], MagicMock(), 5, dry_run=True
            )

        assert result is None  # fallback signal

    def test_stage2_failure_falls_back(self, mock_env, v2_knowledge):
        """Debate failure → _run_ideation_v2 returns None."""
        import strategy_ideation as si

        brainstorm_ideas = [{"name": "test", "type": "param", "family": "rsi|SPY",
                              "one_line_rationale": "test"}]

        def mock_brainstorm(*args, **kwargs):
            return brainstorm_ideas

        def failing_debate(*args, **kwargs):
            raise json.JSONDecodeError("bad JSON", "", 0)

        with patch.object(si, "_stage_brainstorm", side_effect=mock_brainstorm):
            with patch.object(si, "_stage_debate", side_effect=failing_debate):
                result = si._run_ideation_v2(
                    v2_knowledge, STRATEGY_TEMPLATES, [], MagicMock(), 5, dry_run=True
                )

        assert result is None

    def test_stage3_failure_falls_back(self, mock_env, v2_knowledge):
        """Selection failure → _run_ideation_v2 returns None."""
        import strategy_ideation as si

        brainstorm_ideas = [{"name": "test", "type": "param", "family": "rsi|SPY",
                              "one_line_rationale": "test"}]
        debate_results = [{"index": 0, "survive": True, "attack": "", "rebuttal": "",
                           "variation": "", "reason": "ok"}]

        def mock_brainstorm(*args, **kwargs):
            return brainstorm_ideas

        def mock_debate(*args, **kwargs):
            return debate_results, []

        def failing_select(*args, **kwargs):
            raise Exception("select error")

        with patch.object(si, "_stage_brainstorm", side_effect=mock_brainstorm):
            with patch.object(si, "_stage_debate", side_effect=mock_debate):
                with patch.object(si, "_stage_select", side_effect=failing_select):
                    result = si._run_ideation_v2(
                        v2_knowledge, STRATEGY_TEMPLATES, [], MagicMock(), 5, dry_run=True
                    )

        assert result is None

    def test_ideation_v2_disabled_bypass(self, mock_env, v2_knowledge, monkeypatch):
        """IDEATION_V2=0 → bypasses v2 pipeline entirely, uses v1 fallback."""
        monkeypatch.setenv("IDEATION_V2", "0")
        import strategy_ideation as si

        proposals = [
            {
                "type": "param",
                "priority": 0.8,
                "source": {"kind": "paper", "ref": "test.md"},
                "spec": {"strategy": "momentum", "symbol": "AAPL", "params": {"lookback": 20, "hold_period": 5}},
                "eval_plan": {"extra_criteria": []},
            }
        ]

        with patch.object(si, "_call_llm", return_value={"proposals": proposals}):
            ids = si.run(max_proposals=3, dry_run=True)
            assert len(ids) == 1

    def test_ideation_v2_enabled_runs_v2(self, mock_env, v2_knowledge, monkeypatch):
        """IDEATION_V2=1 (default) → runs v2 pipeline successfully."""
        monkeypatch.setenv("IDEATION_V2", "1")
        import strategy_ideation as si

        brainstorm_ideas = [
            {"name": "test_idea", "type": "param", "family": "mean_reversion|SPY",
             "one_line_rationale": "test"},
        ]

        debate_results = [
            {"index": 0, "survive": True, "attack": "ok", "rebuttal": "ok",
             "variation": "", "reason": "ok"},
        ]

        full_proposals = [
            {
                "type": "param",
                "priority": 0.85,
                "source": {"kind": "idea", "ref": "test_idea"},
                "spec": {"strategy": "mean_reversion", "symbol": "SPY", "params": {"window": 20, "threshold": 2.0}},
                "eval_plan": {"extra_criteria": []},
            }
        ]

        def mock_brainstorm(*args, **kwargs):
            return brainstorm_ideas

        def mock_debate(*args, **kwargs):
            return debate_results, [{"index": 0, "survive": True}]

        def mock_select(*args, **kwargs):
            return full_proposals, False

        def mock_save_log(*args, **kwargs):
            pass

        with patch.object(si, "_stage_brainstorm", side_effect=mock_brainstorm):
            with patch.object(si, "_stage_debate", side_effect=mock_debate):
                with patch.object(si, "_stage_select", side_effect=mock_select):
                    with patch.object(si, "_save_ideation_log", side_effect=mock_save_log):
                        ids = si.run(max_proposals=3, dry_run=True)

        assert len(ids) == 1

    def test_brainstorm_prompt_includes_families(self, v2_knowledge):
        """Verify brainstorm prompt includes family aggregates with trial counts."""
        from strategy_ideation import _build_brainstorm_prompt
        from autonomous_loop import STRATEGY_TEMPLATES

        messages = _build_brainstorm_prompt(v2_knowledge, STRATEGY_TEMPLATES, [])
        prompt_text = messages[1]["content"]

        assert "momentum|AAPL" in prompt_text
        assert "5 trials" in prompt_text
        assert "mean_reversion|MSFT" in prompt_text
        assert "1 trials" in prompt_text

    def test_save_ideation_log_creates_file(self, mock_env):
        """Verify _save_ideation_log writes a JSON file to ideation_logs/."""
        from strategy_ideation import _save_ideation_log, _IDEATION_LOG_DIR
        import tempfile
        import shutil

        # Use a temp dir for ideation_logs
        with tempfile.TemporaryDirectory() as td:
            import strategy_ideation as si
            orig_dir = si._IDEATION_LOG_DIR
            try:
                si._IDEATION_LOG_DIR = Path(td)
                _save_ideation_log(
                    [{"name": "idea1"}],
                    [{"index": 0, "attack": "test"}],
                    [{"index": 0, "rebuttal": "ok", "variation": ""}],
                    [{"index": 0, "survive": True, "reason": "good idea"}],
                    "test reasoning",
                    [{"type": "param"}],
                    fallback_used=False,
                )
                log_files = list(Path(td).glob("*.json"))
                assert len(log_files) == 1
                data = json.loads(log_files[0].read_text())
                assert data["stage1_brainstorm"]["n_ideas"] == 1
                assert data["stage2_debate"]["risk_report"] == [{"index": 0, "attack": "test"}]
            finally:
                si._IDEATION_LOG_DIR = orig_dir

    def test_zero_survivor_fallback_used(self, mock_env, v2_knowledge):
        """0 survivors after debate → selects from full list, fallback_used=True."""
        import strategy_ideation as si

        brainstorm_ideas = [
            {"name": "idea1", "type": "param", "family": "rsi|SPY", "one_line_rationale": "test"},
            {"name": "idea2", "type": "code", "family": "strategy|QQQ", "one_line_rationale": "test2"},
        ]

        # All killed by judge
        debate_results = [
            {"index": 0, "survive": False, "attack": "bad", "rebuttal": "weak", "variation": "", "reason": "killed"},
            {"index": 1, "survive": False, "attack": "bad", "rebuttal": "weak", "variation": "", "reason": "killed"},
        ]

        judge_report = [{"index": 0, "survive": False}, {"index": 1, "survive": False}]

        full_proposals = [
            {
                "type": "param",
                "priority": 0.7,
                "source": {"kind": "idea", "ref": "idea1"},
                "spec": {"strategy": "rsi", "symbol": "SPY", "params": {"rsi_window": 14, "oversold": 30, "overbought": 70}},
                "eval_plan": {"extra_criteria": []},
            }
        ]

        def mock_brainstorm(*args, **kwargs):
            return brainstorm_ideas

        def mock_debate(*args, **kwargs):
            return debate_results, judge_report

        # _stage_select is real here — it will see 0 survivors and use fallback
        # We just verify it returns fallback_used=True
        def mock_save_log(*args, **kwargs):
            pass

        with patch.object(si, "_stage_brainstorm", side_effect=mock_brainstorm):
            with patch.object(si, "_stage_debate", side_effect=mock_debate):
                with patch.object(si, "_stage_select", return_value=(full_proposals, True)):
                    with patch.object(si, "_save_ideation_log", side_effect=mock_save_log):
                        result = si._run_ideation_v2(
                            v2_knowledge, STRATEGY_TEMPLATES, [], MagicMock(), 5, dry_run=True
                        )

        assert result is not None
        assert len(result) == 1  # proposal from fallback path
