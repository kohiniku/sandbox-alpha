"""
Regression tests for issue #101: engine version handshake.

The deployed engine image can silently lag behind main, causing merged
fixes to be absent from the live pipeline with no detectable signal.
This test suite verifies:

1. manifest_runner.py injects `engine_git_rev` into every JSON output
   (success and error paths).
2. The Dockerfile bakes the git SHA into the image via /backtest/git_rev.
3. autonomous_loop.py detects drift between engine rev and local HEAD
   and emits a loud warning.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# 1. manifest_runner._engine_git_rev reads the baked file
# ---------------------------------------------------------------------------

class TestEngineGitRev:
    def test_reads_git_rev_file(self, tmp_path):
        """_engine_git_rev() returns the content of /backtest/git_rev."""
        from manifest_runner import _engine_git_rev

        fake_rev_file = tmp_path / "git_rev"
        fake_rev_file.write_text("abc123def456\n")

        with patch("manifest_runner._GIT_REV_PATH", str(fake_rev_file)):
            assert _engine_git_rev() == "abc123def456"

    def test_returns_unknown_when_file_missing(self, tmp_path):
        """_engine_git_rev() returns 'unknown' if the file doesn't exist."""
        from manifest_runner import _engine_git_rev

        missing_path = tmp_path / "nonexistent_git_rev"
        with patch("manifest_runner._GIT_REV_PATH", str(missing_path)):
            assert _engine_git_rev() == "unknown"

    def test_strips_whitespace(self, tmp_path):
        """_engine_git_rev() strips trailing newline/whitespace."""
        from manifest_runner import _engine_git_rev

        fake_rev_file = tmp_path / "git_rev"
        fake_rev_file.write_text("  abc123  \n\n")

        with patch("manifest_runner._GIT_REV_PATH", str(fake_rev_file)):
            assert _engine_git_rev() == "abc123"


# ---------------------------------------------------------------------------
# 2. _error_json includes engine_git_rev
# ---------------------------------------------------------------------------

class TestErrorJsonIncludesRev:
    def test_error_json_has_engine_git_rev(self, tmp_path):
        """_error_json always includes engine_git_rev in its output."""
        from manifest_runner import _error_json

        fake_rev_file = tmp_path / "git_rev"
        fake_rev_file.write_text("deadbeef1234")

        with patch("manifest_runner._GIT_REV_PATH", str(fake_rev_file)):
            output = _error_json("infra", "test error message")
            parsed = json.loads(output)
            assert "engine_git_rev" in parsed
            assert parsed["engine_git_rev"] == "deadbeef1234"

    def test_error_json_with_extra_also_has_rev(self, tmp_path):
        """_error_json with extra dict still includes engine_git_rev."""
        from manifest_runner import _error_json

        fake_rev_file = tmp_path / "git_rev"
        fake_rev_file.write_text("cafe0000")

        with patch("manifest_runner._GIT_REV_PATH", str(fake_rev_file)):
            output = _error_json("code", "some error",
                                extra={"coverage": {"AAPL": {"rows": 0}}})
            parsed = json.loads(output)
            assert parsed["engine_git_rev"] == "cafe0000"
            assert parsed["coverage"] == {"AAPL": {"rows": 0}}


# ---------------------------------------------------------------------------
# 3. Success-path JSON also includes engine_git_rev
# ---------------------------------------------------------------------------

class TestSuccessJsonIncludesRev:
    def test_inject_engine_rev_into_result_dict(self, tmp_path):
        """_inject_engine_rev adds engine_git_rev to a result dict."""
        from manifest_runner import _inject_engine_rev

        fake_rev_file = tmp_path / "git_rev"
        fake_rev_file.write_text("face1234")

        with patch("manifest_runner._GIT_REV_PATH", str(fake_rev_file)):
            result = {"status": "ok", "metrics": {"sharpe": 1.5}}
            _inject_engine_rev(result)
            assert result["engine_git_rev"] == "face1234"


# ---------------------------------------------------------------------------
# 4. Drift detection in autonomous_loop
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_detects_drift_when_revs_differ(self, tmp_path):
        """_check_engine_drift emits warning when engine rev != local HEAD."""
        from autonomous_loop import _check_engine_drift

        local_head = "aaaa1111bbbb2222cccc3333dddd4444eeee5555"
        engine_rev = "ffff0000eeee1111dddd2222cccc3333bbbb4444"

        result = {"status": "ok", "engine_git_rev": engine_rev}

        with patch("autonomous_loop._local_git_head", return_value=local_head):
            drift_msg = _check_engine_drift(result)
            assert drift_msg is not None
            assert "DRIFT" in drift_msg or "drift" in drift_msg.lower()
            assert engine_rev[:12] in drift_msg
            assert local_head[:12] in drift_msg

    def test_no_drift_when_revs_match(self):
        """_check_engine_drift returns None when engine rev == local HEAD."""
        from autonomous_loop import _check_engine_drift

        rev = "aaaa1111bbbb2222cccc3333dddd4444eeee5555"
        result = {"status": "ok", "engine_git_rev": rev}

        with patch("autonomous_loop._local_git_head", return_value=rev):
            assert _check_engine_drift(result) is None

    def test_no_drift_when_engine_unknown(self):
        """_check_engine_drift returns None when engine reports 'unknown'."""
        from autonomous_loop import _check_engine_drift

        result = {"status": "ok", "engine_git_rev": "unknown"}

        with patch("autonomous_loop._local_git_head", return_value="abc123"):
            assert _check_engine_drift(result) is None

    def test_no_drift_when_no_rev_in_result(self):
        """_check_engine_drift returns None when result lacks engine_git_rev."""
        from autonomous_loop import _check_engine_drift

        result = {"status": "ok", "metrics": {}}
        assert _check_engine_drift(result) is None

    def test_local_git_head_reads_git(self):
        """_local_git_head returns the actual HEAD SHA from git."""
        from autonomous_loop import _local_git_head

        rev = _local_git_head()
        # Should be a 40-char hex SHA or None (if not in a git repo)
        if rev is not None:
            assert len(rev) == 40
            assert all(c in "0123456789abcdef" for c in rev)


# ---------------------------------------------------------------------------
# 5. Dockerfile bakes the rev
# ---------------------------------------------------------------------------

class TestDockerfile:
    def test_dockerfile_has_git_rev_arg(self):
        """Dockerfile declares ARG GIT_REV and writes /backtest/git_rev."""
        dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "ARG GIT_REV" in content
        assert "git_rev" in content.lower()
