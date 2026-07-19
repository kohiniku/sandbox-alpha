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
