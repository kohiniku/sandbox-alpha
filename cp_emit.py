"""Best-effort telemetry hooks for the sandbox-cp control-plane dashboard.

These two functions POST run_started/run_finished events so the dashboard can
answer "is a loop currently running right now?" without shell inspection.

ABSOLUTE CONTRACT: these calls are best-effort only. If the ingest endpoint
is down, slow, or the env vars are missing, the caller's actual work must
complete completely unaffected — same behavior, same exit code, same output.
Every exception is swallowed. No dependency on `requests` (stdlib only).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

_TIMEOUT_SECONDS = 3.0


def _ingest_url() -> Optional[str]:
    """Return base ingest URL or None if env is unset."""
    return os.environ.get("SANDBOX_CP_INGEST_URL")


def _auth_header() -> Optional[dict]:
    token = os.environ.get("SANDBOX_CP_INGEST_TOKEN")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _post_json(url: str, body: dict, headers: dict) -> bytes:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST"
    )
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        return resp.read()


def emit_run_started(job_id: str, script: str) -> Optional[int]:
    """POST run_started; return the dashboard-assigned run_id, or None on any failure."""
    try:
        base = _ingest_url()
        auth = _auth_header()
        if not base or not auth:
            return None
        url = base.rstrip("/") + "/run_started"
        payload = {"job_id": job_id, "script": script}
        raw = _post_json(url, payload, auth)
        obj = json.loads(raw)
        run_id = obj.get("run_id")
        if run_id is None:
            return None
        return int(run_id)
    except Exception:
        return None


def emit_run_finished(
    run_id: Optional[int],
    status: str,
    *,
    resolved_model: Optional[str] = None,
    resolved_provider: Optional[str] = None,
    resolved_base_url: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """POST run_finished. No-op if run_id is None. Swallows all exceptions."""
    if run_id is None:
        return
    try:
        base = _ingest_url()
        auth = _auth_header()
        if not base or not auth:
            return
        url = base.rstrip("/") + "/run_finished"
        payload = {
            "run_id": int(run_id),
            "status": status,
            "resolved_model": resolved_model,
            "resolved_provider": resolved_provider,
            "resolved_base_url": resolved_base_url,
            "error": error,
        }
        _post_json(url, payload, auth)
    except Exception:
        pass
