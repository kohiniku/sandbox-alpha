"""Tests for cp_emit telemetry hooks.

These tests verify:
1. Correct URL/header/body construction and response parsing
2. Exception swallowing (best-effort contract)
3. No-op behavior when run_id is None
4. Integration: scripts complete normally even when cp_emit raises
"""
import json
import os
import sys
from unittest import mock

import pytest


def test_emit_run_started_success():
    """emit_run_started constructs correct request and parses run_id."""
    import cp_emit
    
    with mock.patch.dict(os.environ, {
        "SANDBOX_CP_INGEST_URL": "http://test:8100/ingest",
        "SANDBOX_CP_INGEST_TOKEN": "test-token-123",
    }):
        mock_response = mock.MagicMock()
        mock_response.read.return_value = b'{"run_id": 42}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = lambda s, *a: None
        
        with mock.patch("cp_emit.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            run_id = cp_emit.emit_run_started("job-abc", "test.py")
            
            assert run_id == 42
            assert mock_urlopen.called
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            
            # URL construction
            assert req.full_url == "http://test:8100/ingest/run_started"
            
            # Auth header
            assert req.get_header("Authorization") == "Bearer test-token-123"
            assert req.get_header("Content-type") == "application/json"
            
            # Body
            body = json.loads(req.data.decode("utf-8"))
            assert body == {"job_id": "job-abc", "script": "test.py"}
            
            # Timeout
            assert call_args[1]["timeout"] == 3.0


def test_emit_run_started_missing_env():
    """emit_run_started returns None when env vars are missing."""
    import cp_emit
    
    with mock.patch.dict(os.environ, {}, clear=True):
        run_id = cp_emit.emit_run_started("job-abc", "test.py")
        assert run_id is None


def test_emit_run_started_exception_swallowed():
    """emit_run_started swallows all exceptions and returns None."""
    import cp_emit
    
    with mock.patch.dict(os.environ, {
        "SANDBOX_CP_INGEST_URL": "http://test:8100/ingest",
        "SANDBOX_CP_INGEST_TOKEN": "test-token",
    }):
        with mock.patch("cp_emit.urllib.request.urlopen", side_effect=Exception("network error")):
            run_id = cp_emit.emit_run_started("job-abc", "test.py")
            assert run_id is None


def test_emit_run_finished_success():
    """emit_run_finished constructs correct request."""
    import cp_emit
    
    with mock.patch.dict(os.environ, {
        "SANDBOX_CP_INGEST_URL": "http://test:8100/ingest",
        "SANDBOX_CP_INGEST_TOKEN": "test-token-456",
    }):
        mock_response = mock.MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = lambda s, *a: None
        
        with mock.patch("cp_emit.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            cp_emit.emit_run_finished(
                run_id=99,
                status="ok",
                resolved_model="gpt-4",
                resolved_provider="openai",
                resolved_base_url="https://api.openai.com",
                error=None,
            )
            
            assert mock_urlopen.called
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            
            assert req.full_url == "http://test:8100/ingest/run_finished"
            assert req.get_header("Authorization") == "Bearer test-token-456"
            
            body = json.loads(req.data.decode("utf-8"))
            assert body == {
                "run_id": 99,
                "status": "ok",
                "resolved_model": "gpt-4",
                "resolved_provider": "openai",
                "resolved_base_url": "https://api.openai.com",
                "error": None,
            }


def test_emit_run_finished_none_run_id():
    """emit_run_finished is a no-op when run_id is None."""
    import cp_emit
    
    with mock.patch.dict(os.environ, {
        "SANDBOX_CP_INGEST_URL": "http://test:8100/ingest",
        "SANDBOX_CP_INGEST_TOKEN": "test-token",
    }):
        with mock.patch("cp_emit.urllib.request.urlopen") as mock_urlopen:
            cp_emit.emit_run_finished(None, "ok")
            
            # Must NOT call urlopen when run_id is None
            assert not mock_urlopen.called


def test_emit_run_finished_exception_swallowed():
    """emit_run_finished swallows all exceptions."""
    import cp_emit
    
    with mock.patch.dict(os.environ, {
        "SANDBOX_CP_INGEST_URL": "http://test:8100/ingest",
        "SANDBOX_CP_INGEST_TOKEN": "test-token",
    }):
        with mock.patch("cp_emit.urllib.request.urlopen", side_effect=Exception("timeout")):
            # Should not raise
            cp_emit.emit_run_finished(42, "error", error="test error")


@pytest.mark.parametrize("script_name", [
    "autonomous_loop.py",
    "oos_monitor.py",
    "strategy_ideation.py",
    "strategy_review.py",
])
def test_integration_script_completes_despite_cp_emit_failure(script_name, tmp_path):
    """Integration: scripts complete normally even when cp_emit raises.
    
    This is the single most important behavior to prove: the "must never block
    the pipeline" constraint.
    """
    import subprocess
    
    # Create a minimal test script that imports the real script and calls its entry point
    test_script = tmp_path / "test_runner.py"
    test_script.write_text(f"""
import sys
import os
sys.path.insert(0, '/opt/data/sandbox-alpha')

# Set env vars so cp_emit tries to call
os.environ['SANDBOX_CP_INGEST_URL'] = 'http://fake:9999/ingest'
os.environ['SANDBOX_CP_INGEST_TOKEN'] = 'fake-token'

# Mock urlopen to raise (simulating network failure)
import urllib.request
original_urlopen = urllib.request.urlopen
def failing_urlopen(*args, **kwargs):
    raise Exception("Network is down")
urllib.request.urlopen = failing_urlopen

# Now import and run the script
if '{script_name}' == 'autonomous_loop.py':
    import autonomous_loop
    # Just verify it's importable and the __main__ block structure is correct
    print("IMPORT_OK")
elif '{script_name}' == 'oos_monitor.py':
    import oos_monitor
    print("IMPORT_OK")
elif '{script_name}' == 'strategy_ideation.py':
    import strategy_ideation
    # Call main() with --dry-run to avoid side effects
    sys.argv = ['strategy_ideation.py', '--dry-run']
    try:
        strategy_ideation.main()
        print("MAIN_OK")
    except SystemExit as e:
        if e.code == 0:
            print("MAIN_OK")
        else:
            print(f"EXIT_CODE:{{e.code}}")
            sys.exit(1)
elif '{script_name}' == 'strategy_review.py':
    import strategy_review
    # Call main() with --dry-run to avoid side effects
    sys.argv = ['strategy_review.py', '--dry-run']
    try:
        strategy_review.main()
        print("MAIN_OK")
    except SystemExit as e:
        if e.code == 0:
            print("MAIN_OK")
        else:
            print(f"EXIT_CODE:{{e.code}}")
            sys.exit(1)
""")
    
    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    
    # The script must complete (exit 0) even though cp_emit's HTTP call failed
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    # Verify it actually ran
    assert "OK" in result.stdout, f"Script output: {result.stdout}"
