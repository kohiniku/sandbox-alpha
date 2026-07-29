"""
Regression tests for issue #63: HTTP 400 param-validation errors
misclassified as retryable 'infra' — deterministic client errors
retried 3× with no feedback to ideation.

The fix: 4xx status codes → error_type "code" (not retryable),
5xx / network / timeout → error_type "infra" (retryable).
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_loop import (
    _classify_runner_response,
    _run_backtest_sandbox,
    _http_post_for_result,
    evaluate_result,
)


# ──────────────────────────────────────────────
# 1. _classify_runner_response: 4xx → code, 5xx → infra
# ──────────────────────────────────────────────

class TestClassifyRunnerResponseStatusCodes:
    """HTTP status class determines error_type: 4xx=code, 5xx=infra."""

    def test_http_400_is_code_error(self):
        """HTTP 400 (client validation) → error_type 'code', not 'infra'."""
        result, is_ok = _classify_runner_response(
            400, '{"error": "params for momentum must be exactly [\'lookback\', \'hold_period\']"}'
        )
        assert is_ok is False
        assert result["error_type"] == "code"
        assert "400" in result["error"]

    def test_http_422_is_code_error(self):
        """HTTP 422 (unprocessable entity) → error_type 'code'."""
        result, is_ok = _classify_runner_response(422, '{"error": "validation failed"}')
        assert is_ok is False
        assert result["error_type"] == "code"

    def test_http_404_is_code_error(self):
        """HTTP 404 → error_type 'code'."""
        result, is_ok = _classify_runner_response(404, '{"error": "not found"}')
        assert is_ok is False
        assert result["error_type"] == "code"

    def test_http_500_is_infra_error(self):
        """HTTP 500 → error_type 'infra' (retryable)."""
        result, is_ok = _classify_runner_response(500, '{"error": "internal error"}')
        assert is_ok is False
        assert result["error_type"] == "infra"

    def test_http_502_is_infra_error(self):
        """HTTP 502 → error_type 'infra' (retryable)."""
        result, is_ok = _classify_runner_response(502, '{"error": "bad gateway"}')
        assert is_ok is False
        assert result["error_type"] == "infra"

    def test_http_503_is_infra_error(self):
        """HTTP 503 → error_type 'infra' (retryable)."""
        result, is_ok = _classify_runner_response(503, '{"error": "service unavailable"}')
        assert is_ok is False
        assert result["error_type"] == "infra"

    def test_http_200_ok_still_works(self):
        """HTTP 200 with status=ok → parsed payload, is_ok=True."""
        body = json.dumps({"status": "ok", "sharpe_ratio": 1.5})
        result, is_ok = _classify_runner_response(200, body)
        assert is_ok is True
        assert result["sharpe_ratio"] == 1.5

    def test_http_200_error_body_still_uses_body_error_type(self):
        """HTTP 200 with status=error → uses body's error_type (unchanged)."""
        body = json.dumps({"status": "error", "error": "bad code", "error_type": "code"})
        result, is_ok = _classify_runner_response(200, body)
        assert is_ok is False
        assert result["error_type"] == "code"

    def test_http_200_error_body_infra_still_infra(self):
        """HTTP 200 with status=error and error_type=infra → infra (unchanged)."""
        body = json.dumps({"status": "error", "error": "oom", "error_type": "infra"})
        result, is_ok = _classify_runner_response(200, body)
        assert is_ok is False
        assert result["error_type"] == "infra"


# ──────────────────────────────────────────────
# 2. _run_backtest_sandbox: 4xx → code, 5xx → infra
# ──────────────────────────────────────────────

class TestRunBacktestSandboxStatusCodes:
    """_run_backtest_sandbox classifies HTTP status correctly."""

    def _mock_urlopen(self, status, body):
        """Create a mock for urllib.request.urlopen returning given status/body."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.read.return_value = body.encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_sandbox_400_is_code_error(self):
        """_run_backtest_sandbox: HTTP 400 → error_type 'code'."""
        mock_resp = self._mock_urlopen(400, '{"error": "params must be exactly [...]"}')
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _run_backtest_sandbox(
                "http://localhost:9999", "momentum", "AAPL",
                {"lookback_period": 60}  # wrong key
            )
        assert result["error_type"] == "code"
        assert "400" in result["error"]

    def test_sandbox_500_is_infra_error(self):
        """_run_backtest_sandbox: HTTP 500 → error_type 'infra'."""
        mock_resp = self._mock_urlopen(500, '{"error": "internal error"}')
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _run_backtest_sandbox(
                "http://localhost:9999", "momentum", "AAPL",
                {"lookback": 20, "hold_period": 5}
            )
        assert result["error_type"] == "infra"

    def test_sandbox_http_error_exception_400_is_code(self):
        """_run_backtest_sandbox: HTTPError(400) → error_type 'code'."""
        err = urllib.error.HTTPError(
            "http://localhost:9999/run", 400, "Bad Request", {},
            __import__("io").BytesIO(b'{"error": "params must be exactly [...]}')
        )
        with patch("urllib.request.urlopen", side_effect=err):
            result = _run_backtest_sandbox(
                "http://localhost:9999", "momentum", "AAPL", {"bad": 1}
            )
        assert result["error_type"] == "code"

    def test_sandbox_http_error_exception_500_is_infra(self):
        """_run_backtest_sandbox: HTTPError(500) → error_type 'infra'."""
        err = urllib.error.HTTPError(
            "http://localhost:9999/run", 500, "Internal Server Error", {},
            __import__("io").BytesIO(b'{"error": "boom"}')
        )
        with patch("urllib.request.urlopen", side_effect=err):
            result = _run_backtest_sandbox(
                "http://localhost:9999", "momentum", "AAPL", {"lookback": 20}
            )
        assert result["error_type"] == "infra"


# ──────────────────────────────────────────────
# 3. _http_post_for_result: 4xx → code, 5xx → infra
# ──────────────────────────────────────────────

class TestHttpPostForResultStatusCodes:
    """_http_post_for_result classifies HTTP status correctly."""

    def test_http_error_400_is_code(self):
        """_http_post_for_result: HTTPError(400) → error_type 'code'."""
        err = urllib.error.HTTPError(
            "http://localhost:9999/run", 400, "Bad Request", {},
            __import__("io").BytesIO(b'{"error": "validation failed"}')
        )
        with patch("urllib.request.urlopen", side_effect=err):
            result, is_ok = _http_post_for_result(
                "http://localhost:9999/run", b'{}', timeout=30
            )
        assert is_ok is False
        assert result["error_type"] == "code"

    def test_http_error_500_is_infra(self):
        """_http_post_for_result: HTTPError(500) → error_type 'infra'."""
        err = urllib.error.HTTPError(
            "http://localhost:9999/run", 500, "ISE", {},
            __import__("io").BytesIO(b'{"error": "boom"}')
        )
        with patch("urllib.request.urlopen", side_effect=err):
            result, is_ok = _http_post_for_result(
                "http://localhost:9999/run", b'{}', timeout=30
            )
        assert is_ok is False
        assert result["error_type"] == "infra"

    def test_url_error_is_infra(self):
        """_http_post_for_result: URLError (connection refused) → 'infra'."""
        err = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            result, is_ok = _http_post_for_result(
                "http://localhost:9999/run", b'{}', timeout=30
            )
        assert is_ok is False
        assert result["error_type"] == "infra"


# ──────────────────────────────────────────────
# 4. End-to-end: 400 → code_error verdict → no retry
# ──────────────────────────────────────────────

class TestEndToEndNoRetryOn400:
    """A 400 response flows through evaluate_result as code_error (no retry)."""

    def test_400_becomes_code_error_verdict(self):
        """classify(400) → evaluate_result → verdict 'code_error'."""
        result, is_ok = _classify_runner_response(
            400, '{"error": "params for momentum must be exactly [\'lookback\', \'hold_period\']"}'
        )
        assert is_ok is False

        hyp = {
            "id": "hyp_test_400",
            "strategy": "momentum",
            "symbol": "AAPL",
            "params": {"lookback_period": 60},
            "description": "momentum on AAPL",
            "generated_at": "2026-07-29T00:00:00",
        }
        knowledge = {
            "tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
            "superseded": [], "families": {}, "iterations": 0, "errors": [],
        }
        verdict, evaluation = evaluate_result(hyp, result, knowledge)

        # Must be code_error (immediate DONE_ERROR), NOT error (retry 3×)
        assert verdict == "code_error", (
            f"Expected 'code_error' for HTTP 400 but got '{verdict}' — "
            f"400 is a deterministic client error and must not be retried"
        )
        assert evaluation["error_type"] == "code"

    def test_500_becomes_error_verdict_retryable(self):
        """classify(500) → evaluate_result → verdict 'error' (retryable)."""
        result, is_ok = _classify_runner_response(500, '{"error": "internal"}')
        assert is_ok is False

        hyp = {
            "id": "hyp_test_500",
            "strategy": "momentum",
            "symbol": "AAPL",
            "params": {"lookback": 20, "hold_period": 5},
            "description": "momentum on AAPL",
            "generated_at": "2026-07-29T00:00:00",
        }
        knowledge = {
            "tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
            "superseded": [], "families": {}, "iterations": 0, "errors": [],
        }
        verdict, evaluation = evaluate_result(hyp, result, knowledge)

        assert verdict == "error"
        assert evaluation["error_type"] == "infra"
