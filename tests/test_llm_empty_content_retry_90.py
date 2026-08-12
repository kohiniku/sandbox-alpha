"""
Regression tests for issue #90: HTTP-200 empty/truncated LLM JSON from
reasoning-token exhaustion is invisible to the status-code retry policy.

DeepSeek v4-class reasoning models burn the max_tokens budget in hidden
``reasoning_content`` and return HTTP 200 with ``finish_reason="length"``
and EMPTY (or truncated) ``content``. The retry layer only covers
403/429/5xx, so this dominant failure mode (ideation failed ~50% of cycles
in 08-07/08-08) flowed straight into json.loads() and the parse-retry
cascade — which re-calls the same config and fails in lockstep.

Fix: _http_post_json detects the condition (empty content OR
finish_reason="length") and raises ContentLengthExhaustedError; the retry
loop catches it and retries with a DOUBLED max_tokens budget (the documented
in-repo mitigation, previously applied only by hand), capped at
_MAX_RETRY_MAX_TOKENS.
"""
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hypothesis import (
    _http_post_json,
    ContentLengthExhaustedError,
)

# ---------------------------------------------------------------------------
# Mock helpers (same pattern as tests/test_llm_hypothesis.py)
# ---------------------------------------------------------------------------


def _http_error(code):
    return urllib.error.HTTPError(
        url="https://example.test/chat/completions", code=code, msg="err",
        hdrs=None, fp=None,
    )


def _fake_ok_response(body: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode("utf-8")
    resp.__enter__ = lambda self: resp
    resp.__exit__ = lambda self, *a: None
    return resp


def _empty_content_response(finish_reason="length", content=""):
    """HTTP 200 envelope with empty/truncated content (the #90 signature)."""
    return _fake_ok_response({
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }]
    })


def _ok_content_response(content='{"ok": true}', finish_reason="stop"):
    return _fake_ok_response({
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }]
    })


def _request_payload(mock_urlopen, attempt_index):
    """Extract the JSON payload from the nth urlopen call's Request."""
    req = mock_urlopen.call_args_list[attempt_index].args[0]
    return json.loads(req.data.decode("utf-8"))


# ---------------------------------------------------------------------------
# 1. Empty content (the dominant failure) → retry with bigger max_tokens
# ---------------------------------------------------------------------------

@patch("llm_hypothesis.time.sleep")
@patch("llm_hypothesis.urllib.request.urlopen")
def test_http_post_json_retries_empty_content_with_doubled_budget(mock_urlopen, mock_sleep):
    """HTTP 200 with EMPTY content must be retried with max_tokens doubled,
    not returned as a success envelope."""
    mock_urlopen.side_effect = [
        _empty_content_response(content=""),
        _ok_content_response(),
    ]
    result = _http_post_json(
        "https://example.test/chat/completions",
        headers={},
        payload={"model": "deepseek-v4-flash", "max_tokens": 8192},
        timeout=5,
    )
    assert result["choices"][0]["message"]["content"] == '{"ok": true}'
    assert mock_urlopen.call_count == 2, "empty content must trigger a retry"
    # Second attempt carries a doubled output budget
    assert _request_payload(mock_urlopen, 1)["max_tokens"] == 16384
    # First attempt budget untouched
    assert _request_payload(mock_urlopen, 0)["max_tokens"] == 8192


@patch("llm_hypothesis.time.sleep")
@patch("llm_hypothesis.urllib.request.urlopen")
def test_http_post_json_exhausts_empty_content_retries_then_raises(mock_urlopen, mock_sleep):
    """Persistent empty content must raise ContentLengthExhaustedError (a
    DISTINCT failure type), with escalating budgets on each attempt."""
    mock_urlopen.side_effect = [
        _empty_content_response(),
        _empty_content_response(),
        _empty_content_response(),
    ]
    with pytest.raises(ContentLengthExhaustedError) as exc_info:
        _http_post_json(
            "https://example.test/chat/completions",
            headers={},
            payload={"model": "deepseek-v4-flash", "max_tokens": 8192},
            timeout=5, retries=2,
        )
    assert "finish_reason" in str(exc_info.value)
    assert mock_urlopen.call_count == 3
    budgets = [_request_payload(mock_urlopen, i)["max_tokens"] for i in range(3)]
    assert budgets == [8192, 16384, 32768], f"budgets must escalate, got {budgets}"


# ---------------------------------------------------------------------------
# 2. Truncated content (finish_reason=length, non-empty) → same treatment
# ---------------------------------------------------------------------------

@patch("llm_hypothesis.time.sleep")
@patch("llm_hypothesis.urllib.request.urlopen")
def test_http_post_json_retries_truncated_content_finish_reason_length(mock_urlopen, mock_sleep):
    """finish_reason='length' with non-empty (truncated) content must also
    retry — this is the 'Unterminated string at line 43' failure mode."""
    mock_urlopen.side_effect = [
        _empty_content_response(content='{"verdict": "refine", "rational', finish_reason="length"),
        _ok_content_response(),
    ]
    result = _http_post_json(
        "https://example.test/chat/completions",
        headers={},
        payload={"model": "deepseek-v4-pro", "max_tokens": 8192},
        timeout=5,
    )
    assert result["choices"][0]["message"]["content"] == '{"ok": true}'
    assert mock_urlopen.call_count == 2
    assert _request_payload(mock_urlopen, 1)["max_tokens"] == 16384


# ---------------------------------------------------------------------------
# 3. Invariant guards — the check must not over-trigger or change passthrough
# ---------------------------------------------------------------------------

@patch("llm_hypothesis.time.sleep")
@patch("llm_hypothesis.urllib.request.urlopen")
def test_http_post_json_complete_response_no_retry(mock_urlopen, mock_sleep):
    """finish_reason='stop' with valid content returns immediately."""
    mock_urlopen.side_effect = [_ok_content_response()]
    result = _http_post_json(
        "https://example.test/chat/completions",
        headers={},
        payload={"model": "deepseek-v4-flash", "max_tokens": 8192},
        timeout=5,
    )
    assert result["choices"][0]["message"]["content"] == '{"ok": true}'
    assert mock_urlopen.call_count == 1
    mock_sleep.assert_not_called()


@patch("llm_hypothesis.time.sleep")
@patch("llm_hypothesis.urllib.request.urlopen")
def test_http_post_json_passthrough_envelope_without_choices(mock_urlopen, mock_sleep):
    """Envelopes without a usable choices[0] (e.g. {"ok": true} from the
    existing throttle-retry tests) pass through unchanged — the content check
    only fires on the specific empty-content/length signature."""
    mock_urlopen.side_effect = [_fake_ok_response({"ok": True})]
    result = _http_post_json("https://example.test", {}, {}, timeout=5)
    assert result == {"ok": True}
    assert mock_urlopen.call_count == 1
    mock_sleep.assert_not_called()


@patch("llm_hypothesis.time.sleep")
@patch("llm_hypothesis.urllib.request.urlopen")
def test_http_post_json_status_retries_still_work(mock_urlopen, mock_sleep):
    """Invariant guard: the existing 403 throttle retry path is untouched."""
    mock_urlopen.side_effect = [_http_error(403), _ok_content_response()]
    result = _http_post_json(
        "https://example.test/chat/completions",
        headers={},
        payload={"model": "deepseek-v4-flash", "max_tokens": 8192},
        timeout=5,
    )
    assert result["choices"][0]["message"]["content"] == '{"ok": true}'
    assert mock_urlopen.call_count == 2
    # No max_tokens bump on a status-code retry — budget escalation is
    # specific to content-exhaustion retries.
    assert _request_payload(mock_urlopen, 1)["max_tokens"] == 8192


@patch("llm_hypothesis.time.sleep")
@patch("llm_hypothesis.urllib.request.urlopen")
def test_http_post_json_missing_content_key_treated_as_empty(mock_urlopen, mock_sleep):
    """Some providers omit 'content' entirely when reasoning ate the budget —
    that must be detected too (not a KeyError passthrough)."""
    mock_urlopen.side_effect = [
        _fake_ok_response({"choices": [{"message": {}, "finish_reason": "length"}]}),
        _ok_content_response(),
    ]
    result = _http_post_json(
        "https://example.test/chat/completions",
        headers={},
        payload={"model": "deepseek-v4-flash", "max_tokens": 4096},
        timeout=5,
    )
    assert result["choices"][0]["message"]["content"] == '{"ok": true}'
    assert mock_urlopen.call_count == 2
    assert _request_payload(mock_urlopen, 1)["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# 4. Ideation surface: the failure type is distinct, not a JSONDecodeError
# ---------------------------------------------------------------------------

def test_ideation_call_llm_surfaces_content_exhaustion(monkeypatch):
    """strategy_ideation._call_llm propagates ContentLengthExhaustedError as
    its own type (after internal budget-escalation retries are exhausted) —
    so the V3→V2→V1 parse-retry cascade sees a distinct, diagnosable failure
    instead of a generic JSONDecodeError from empty content."""
    import strategy_ideation as si

    def _always_empty(url, headers, payload, timeout, retries=2):
        raise ContentLengthExhaustedError("empty/truncated content (finish_reason='length')")

    monkeypatch.setattr(si, "_post_json", _always_empty)
    with pytest.raises(ContentLengthExhaustedError):
        si._call_llm([{"role": "user", "content": "hi"}], max_tokens=8192)
