"""
Tests for llm_hypothesis.py — all network calls are mocked.
No real HTTP requests; no API key required.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hypothesis import generate, _validate_response, _SYMBOL_RE

# Minimal templates matching autonomous_loop.py STRATEGY_TEMPLATES
TEMPLATES = {
    "sma_crossover": {
        "description": "移動平均クロスオーバー",
        "param_space": {"fast_window": range(5, 30), "slow_window": range(20, 100)},
    },
    "mean_reversion": {
        "description": "平均回帰",
        "param_space": {"window": range(10, 60), "threshold": [1.0, 1.5, 2.0, 2.5, 3.0]},
    },
    "momentum": {
        "description": "モメンタム",
        "param_space": {"lookback": range(5, 60), "hold_period": range(1, 20)},
    },
}

# Minimal knowledge dict
KNOWLEDGE = {
    "tested": [],
    "tested_combinations": [],
    "adopted": [],
    "rejected": [],
    "iterations": 0,
}


# ---------------------------------------------------------------------------
# Symbol validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("AAPL", True),
        ("MSFT", True),
        ("BTC-USD", True),
        ("SPY", True),
        ("QQQ", True),
        ("BRK.B", True),
        ("12345", True),         # digits are valid in [A-Z0-9]
        ("abc", False),           # lowercase
        ("", False),
        ("A" * 13, False),        # too long (max 12)
        ("@#$", False),
    ],
)
def test_symbol_regex(symbol, expected):
    assert bool(_SYMBOL_RE.match(symbol)) == expected


# ---------------------------------------------------------------------------
# Validation tests (no network)
# ---------------------------------------------------------------------------

def test_valid_response_accepted():
    """Valid LLM response passes validation and returns proper hypothesis dict."""
    parsed = {
        "strategy": "momentum",
        "symbol": "AAPL",
        "params": {"lookback": 20, "hold_period": 5},
        "rationale": "Momentum looks strong on AAPL recently.",
    }
    validated = _validate_response(parsed, TEMPLATES)
    assert validated["strategy"] == "momentum"
    assert validated["params"] == {"lookback": 20, "hold_period": 5}


def test_unknown_strategy_rejected():
    """Strategy not in templates raises ValueError."""
    parsed = {
        "strategy": "bogus_strategy",
        "symbol": "AAPL",
        "params": {"lookback": 20, "hold_period": 5},
        "rationale": "test",
    }
    with pytest.raises(ValueError, match="Unknown strategy"):
        _validate_response(parsed, TEMPLATES)


def test_out_of_range_param_rejected():
    """Param value outside range raises ValueError."""
    parsed = {
        "strategy": "sma_crossover",
        "symbol": "SPY",
        "params": {"fast_window": 50, "slow_window": 80},  # fast_window max is 29
        "rationale": "test",
    }
    with pytest.raises(ValueError, match="out of range"):
        _validate_response(parsed, TEMPLATES)


def test_param_not_in_list_rejected():
    """Param value not in allowed list raises ValueError."""
    parsed = {
        "strategy": "mean_reversion",
        "symbol": "QQQ",
        "params": {"window": 20, "threshold": 9.9},  # 9.9 not in [1.0, 1.5, 2.0, 2.5, 3.0]
        "rationale": "test",
    }
    with pytest.raises(ValueError, match="not in allowed"):
        _validate_response(parsed, TEMPLATES)


def test_params_key_mismatch_rejected():
    """Wrong param key names raise ValueError."""
    parsed = {
        "strategy": "momentum",
        "symbol": "MSFT",
        "params": {"lookback": 20, "wrong_key": 5},
        "rationale": "test",
    }
    with pytest.raises(ValueError, match="key mismatch|Unexpected keys|Missing keys"):
        _validate_response(parsed, TEMPLATES)


def test_params_missing_key_rejected():
    """Missing required param key raises ValueError."""
    parsed = {
        "strategy": "momentum",
        "symbol": "MSFT",
        "params": {"lookback": 20},  # missing hold_period
        "rationale": "test",
    }
    with pytest.raises(ValueError, match="key mismatch|Missing keys"):
        _validate_response(parsed, TEMPLATES)


def test_invalid_symbol_rejected():
    """Symbol failing regex raises ValueError."""
    parsed = {
        "strategy": "momentum",
        "symbol": "lowercase",
        "params": {"lookback": 20, "hold_period": 5},
        "rationale": "test",
    }
    with pytest.raises(ValueError, match="Invalid symbol"):
        _validate_response(parsed, TEMPLATES)


# ---------------------------------------------------------------------------
# generate() tests (mocked HTTP)
# ---------------------------------------------------------------------------

@patch("llm_hypothesis._http_post_json")
def test_generate_valid_reply_accepted(mock_post):
    """Full generate() flow: valid LLM reply → proper hypothesis dict."""
    mock_post.return_value = (
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "strategy": "momentum",
                            "symbol": "NVDA",
                            "params": {"lookback": 15, "hold_period": 3},
                            "rationale": "NVDA shows strong trend signals.",
                        })
                    }
                }
            ]
        }
    )

    hyp = generate(KNOWLEDGE, TEMPLATES)
    assert hyp["strategy"] == "momentum"
    assert hyp["symbol"] == "NVDA"
    assert hyp["params"] == {"lookback": 15, "hold_period": 3}
    assert hyp["rationale"] == "NVDA shows strong trend signals."
    assert "id" in hyp
    assert hyp["id"].startswith("hyp_")


@patch("llm_hypothesis._http_post_json")
def test_generate_out_of_range_falls_back(mock_post):
    """Out-of-range param in LLM output → ValueError (caller catches)."""
    mock_post.return_value = (
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "strategy": "sma_crossover",
                            "symbol": "MSFT",
                            "params": {"fast_window": 999, "slow_window": 50},
                            "rationale": "test",
                        })
                    }
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="out of range"):
        generate(KNOWLEDGE, TEMPLATES)


@patch("llm_hypothesis._http_post_json")
def test_generate_unknown_strategy_rejected(mock_post):
    """Unknown strategy → ValueError."""
    mock_post.return_value = (
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "strategy": "pairs_trading",
                            "symbol": "AAPL",
                            "params": {"lookback": 20, "hold_period": 5},
                            "rationale": "test",
                        })
                    }
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="Unknown strategy"):
        generate(KNOWLEDGE, TEMPLATES)


@patch("llm_hypothesis._http_post_json")
def test_generate_malformed_json_raises(mock_post):
    """Non-JSON content → json.JSONDecodeError."""
    mock_post.return_value = (
        {
            "choices": [
                {
                    "message": {
                        "content": "not json at all, just some prose"
                    }
                }
            ]
        }
    )

    with pytest.raises(json.JSONDecodeError):
        generate(KNOWLEDGE, TEMPLATES)


@patch("llm_hypothesis._http_post_json")
def test_generate_params_key_mismatch_rejected(mock_post):
    """Wrong param keys → ValueError."""
    mock_post.return_value = (
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "strategy": "momentum",
                            "symbol": "SPY",
                            "params": {"lookback": 20, "extra_field": 5},
                            "rationale": "test",
                        })
                    }
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="key mismatch|Unexpected keys"):
        generate(KNOWLEDGE, TEMPLATES)
