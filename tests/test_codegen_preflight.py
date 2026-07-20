"""
Tests for codegen preflight: synthetic df determinism, contract checks, preflight retry/drop logic.
"""
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.strategy_harness import (
    build_synthetic_df,
    run_preflight,
    _call_signals,
    exec_and_extract,
)


# ---------------------------------------------------------------------------
# Synthetic DataFrame determinism + contract
# ---------------------------------------------------------------------------

class TestSyntheticDF:
    def test_shape_and_columns(self):
        df = build_synthetic_df()
        assert len(df) == 250
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_datetime_index(self):
        df = build_synthetic_df()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_deterministic_seed(self):
        df1 = build_synthetic_df()
        df2 = build_synthetic_df()
        assert df1.equals(df2)

    def test_different_seed(self):
        df1 = build_synthetic_df(seed=42)
        df2 = build_synthetic_df(seed=99)
        assert not df1["Close"].equals(df2["Close"])

    def test_positive_prices(self):
        df = build_synthetic_df()
        assert (df["Close"] > 0).all()
        assert (df["Open"] > 0).all()
        assert (df["High"] >= df["Close"]).all()
        assert (df["Low"] <= df["Close"]).all()

    def test_no_date_column(self):
        df = build_synthetic_df()
        assert "Date" not in df.columns
        assert "date" not in df.columns


# ---------------------------------------------------------------------------
# Contract checks via _call_signals
# ---------------------------------------------------------------------------

class TestContractChecks:
    def _make_code(self, body):
        return f"import numpy as np\nimport pandas as pd\n\ndef generate_signals(df):\n{body}\n"

    def test_valid_signals_pass(self):
        df = build_synthetic_df()
        code = self._make_code("    return pd.Series(0, index=df.index)")
        gen = exec_and_extract(code, df)
        signals = _call_signals(gen, df)
        assert isinstance(signals, pd.Series)
        assert signals.index.equals(df.index)

    def test_lowercase_close_crashes(self):
        """Code using df['close'] (lowercase) must fail — contract violation."""
        df = build_synthetic_df()
        code = self._make_code("    return pd.Series(0, index=df.index) if df['close'].sum() > 0 else pd.Series(1, index=df.index)")
        gen = exec_and_extract(code, df)
        with pytest.raises(Exception):
            _call_signals(gen, df)

    def test_date_column_reference_crashes(self):
        """Code referencing df['Date'] must fail — no Date column in synthetic df."""
        df = build_synthetic_df()
        code = self._make_code("    dates = df['Date']\n    return pd.Series(0, index=df.index)")
        gen = exec_and_extract(code, df)
        with pytest.raises(Exception):
            _call_signals(gen, df)

    def test_invalid_signal_values_rejected(self):
        df = build_synthetic_df()
        code = self._make_code("    return pd.Series(2, index=df.index)")
        gen = exec_and_extract(code, df)
        with pytest.raises(RuntimeError, match="invalid"):
            _call_signals(gen, df)

    def test_wrong_index_rejected(self):
        df = build_synthetic_df()
        code = self._make_code("    return pd.Series(0, index=range(len(df)))")
        gen = exec_and_extract(code, df)
        with pytest.raises(RuntimeError, match="index"):
            _call_signals(gen, df)


# ---------------------------------------------------------------------------
# run_preflight integration
# ---------------------------------------------------------------------------

class TestRunPreflight:
    def test_valid_code(self):
        code = """
import numpy as np
import pandas as pd

def generate_signals(df):
    fast = df["Close"].rolling(10).mean()
    slow = df["Close"].rolling(30).mean()
    signals = pd.Series(0, index=df.index)
    signals[fast > slow] = 1
    signals[fast < slow] = -1
    return signals
"""
        b64 = base64.b64encode(code.encode()).decode()
        result = run_preflight(b64)
        assert result["valid"] is True
        assert "n_signals" in result

    def test_lowercase_close_fails(self):
        code = """
import pandas as pd

def generate_signals(df):
    return pd.Series(0, index=df.index) if df["close"].sum() > 0 else pd.Series(1, index=df.index)
"""
        b64 = base64.b64encode(code.encode()).decode()
        result = run_preflight(b64)
        assert result["valid"] is False
        assert "error" in result

    def test_date_column_fails(self):
        code = """
import pandas as pd

def generate_signals(df):
    dates = df["Date"]
    return pd.Series(0, index=df.index)
"""
        b64 = base64.b64encode(code.encode()).decode()
        result = run_preflight(b64)
        assert result["valid"] is False

    def test_forbidden_import_fails(self):
        code = """
import os

def generate_signals(df):
    return pd.Series(0, index=df.index)
"""
        b64 = base64.b64encode(code.encode()).decode()
        result = run_preflight(b64)
        assert result["valid"] is False
        assert "os" in result["error"]


# ---------------------------------------------------------------------------
# Preflight retry/drop logic in strategy_ideation (mock HTTP)
# ---------------------------------------------------------------------------

class TestPreflightRetryLogic:
    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "DUMMY_KEY")
        monkeypatch.setenv("DUMMY_KEY", "noop")
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9999")
        import strategy_ideation
        with tempfile.TemporaryDirectory() as td:
            bp = Path(td) / "research"
            bp.mkdir()
            monkeypatch.setenv("RESEARCH_DIRS", str(bp))
            monkeypatch.setenv("KNOWLEDGE_PATH", str(Path(td) / "knowledge.json"))
            monkeypatch.setenv("BACKLOG_PATH", str(Path(td) / "backlog.json"))
            yield td

    def test_preflight_pass_no_retry(self, mock_env):
        """Code passes preflight on first try → accepted, no retry."""
        import strategy_ideation

        proposals = [
            {
                "type": "code",
                "priority": 0.7,
                "source": {"kind": "idea", "ref": "test"},
                "spec": {
                    "name": "good_strat",
                    "description": "desc",
                    "code": 'import pandas as pd\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)',
                    "symbol": "SPY",
                },
                "eval_plan": {"extra_criteria": []},
            }
        ]

        def mock_llm(messages, max_tokens=4096):
            return {"proposals": proposals}

        def mock_validate(code_str):
            return True, "", ""

        with patch.object(strategy_ideation, "_call_llm", side_effect=mock_llm):
            with patch.object(strategy_ideation, "_preflight_validate", side_effect=mock_validate):
                ids = strategy_ideation.run(max_proposals=3, dry_run=True)
                assert len(ids) == 1

    def test_preflight_fail_then_fix(self, mock_env):
        """Code fails first try, LLM fixes it, second try passes → accepted."""
        import strategy_ideation

        proposals = [
            {
                "type": "code",
                "priority": 0.7,
                "source": {"kind": "idea", "ref": "test"},
                "spec": {
                    "name": "bad_strat",
                    "description": "desc",
                    "code": 'import pandas as pd\ndef generate_signals(df):\n    return pd.Series(0, index=df["Date"])',
                    "symbol": "SPY",
                },
                "eval_plan": {"extra_criteria": []},
            }
        ]

        def mock_llm(messages, max_tokens=4096):
            if "proposals" in str(messages):
                return {"proposals": proposals}
            else:
                # Fix attempt
                return 'import pandas as pd\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)'

        call_count = [0]
        def mock_validate(code_str):
            call_count[0] += 1
            if call_count[0] == 1:
                return False, "wrong index", "traceback..."
            else:
                return True, "", ""

        with patch.object(strategy_ideation, "_call_llm", side_effect=mock_llm):
            with patch.object(strategy_ideation, "_preflight_validate", side_effect=mock_validate):
                ids = strategy_ideation.run(max_proposals=3, dry_run=True)
                assert len(ids) == 1

    def test_preflight_fail_drop_after_retries(self, mock_env):
        """Code fails, LLM fix attempts all fail → dropped, not added."""
        import strategy_ideation

        proposals = [
            {
                "type": "code",
                "priority": 0.7,
                "source": {"kind": "idea", "ref": "test"},
                "spec": {
                    "name": "broken_strat",
                    "description": "desc",
                    "code": 'import pandas as pd\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)',
                    "symbol": "SPY",
                },
                "eval_plan": {"extra_criteria": []},
            }
        ]

        def mock_llm(messages, max_tokens=4096):
            if "proposals" in str(messages):
                return {"proposals": proposals}
            else:
                return 'import pandas as pd\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)'

        def mock_validate(code_str):
            return False, "persistent error", "traceback..."

        with patch.object(strategy_ideation, "_call_llm", side_effect=mock_llm):
            with patch.object(strategy_ideation, "_preflight_validate", side_effect=mock_validate):
                ids = strategy_ideation.run(max_proposals=3, dry_run=True)
                assert len(ids) == 0  # dropped

    def test_preflight_skip_when_runner_unset(self, mock_env, monkeypatch):
        """SANDBOX_RUNNER_URL unset → skip preflight, don't block."""
        import strategy_ideation
        monkeypatch.delenv("SANDBOX_RUNNER_URL", raising=False)

        proposals = [
            {
                "type": "code",
                "priority": 0.7,
                "source": {"kind": "idea", "ref": "test"},
                "spec": {
                    "name": "strat",
                    "description": "desc",
                    "code": 'import pandas as pd\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)',
                    "symbol": "SPY",
                },
                "eval_plan": {"extra_criteria": []},
            }
        ]

        def mock_llm(messages, max_tokens=4096):
            return {"proposals": proposals}

        with patch.object(strategy_ideation, "_call_llm", side_effect=mock_llm):
            ids = strategy_ideation.run(max_proposals=3, dry_run=True)
            assert len(ids) == 1  # accepted despite no preflight


def test_build_prompt_contains_interface_contract():
    """Verify the LLM prompt includes the INTERFACE CONTRACT section."""
    from strategy_ideation import _build_prompt
    from autonomous_loop import STRATEGY_TEMPLATES

    knowledge = {"rejected": [], "families": {}}
    templates = {"momentum": STRATEGY_TEMPLATES["momentum"]}
    docs = []
    backlog_summary = "No pending entries."

    messages = _build_prompt(knowledge, templates, docs, backlog_summary, 5)
    prompt_text = messages[1]["content"]

    assert "INTERFACE CONTRACT" in prompt_text
    assert "DatetimeIndex" in prompt_text
    assert "NO 'Date' column" in prompt_text
    assert "Open, High, Low, Close, Volume" in prompt_text
    assert "GOLDEN EXAMPLE" in prompt_text
