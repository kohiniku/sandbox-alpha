"""
Regression tests for issue #88: OOS monitor dispatch for manifest/codegen adoptions.

Before the fix:
  - Every adopted manifest:* / codegen entry was funnelled through
    run_backtest() -> POST /run {"strategy": "manifest:..."}, which the runner
    rejects with "unknown strategy" (verified live: OOS_STATUS ... error=runner_code x12).
  - checked=0 was reported as "OOS_SUMMARY checked=0", which the cron delivery
    policy maps to [SILENT] — a fully broken monitor is indistinguishable from
    "nothing to check".

After the fix:
  - manifest:* entries resolve their spec (stored on the record, else backlog
    history) and re-run via /run_manifest with data_sources[].start overridden
    to the adoption date — the runner ignores metrics_since on /run_manifest,
    so the window override is the only way to restrict to post-adoption data.
  - codegen entries resolve their code (code_hash match against backlog
    history) and re-run via /run_code, recording the holdout split
    (oos_kind=codegen_holdout — /run_code cannot restrict the window).
  - param entries keep the /run + metrics_since path.
  - a run that checks nothing for structural reasons prints OOS_BROKEN and
    exits non-zero, so the delivery policy can no longer mistake it for
    "no adopted strategies".

All tests are offline: /run_manifest and /run_code POSTs are mocked.
"""
import base64
import hashlib
import io
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backlog import Backlog
from autonomous_loop import _extract_universe_meta, _classify_http_status
from oos_monitor import (
    run_oos_check,
    run_oos_monitor,
    _resolve_manifest_spec,
    _resolve_code_spec,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SAMPLE_CODE = (
    "import numpy as np\n"
    "import pandas as pd\n"
    "def generate_weights(data_dict):\n"
    "    return {t: 1.0 / len(data_dict) for t in data_dict}\n"
)


def _b64(code: str) -> str:
    return base64.b64encode(code.encode("utf-8")).decode("ascii")


def _manifest_spec(name="regime_switching_gmm", universe=None, start="2020-01-01"):
    if universe is None:
        universe = ["SPY", "GLD"]
    return {
        "name": name,
        "code_b64": _b64(SAMPLE_CODE),
        "data_sources": [{"type": "ohlcv", "universe": universe, "start": start}],
        "evaluator": {"type": "portfolio", "metrics": ["sharpe"]},
    }


def _adopted_manifest_entry(name="regime_switching_gmm", universe=None,
                            tested_at="2026-06-20T10:00:00", **extra):
    """Build an adopted knowledge entry for a manifest strategy (issue #88:
    the record carries only strategy=manifest:<name> + symbol=universe:<hash>)."""
    spec = _manifest_spec(name, universe)
    _, uhash, usize = _extract_universe_meta(spec)
    entry = {
        "hypothesis": {
            "id": f"bl_test_{name[:8]}",
            "strategy": f"manifest:{name}",
            "symbol": f"universe:{uhash}",
            "params": {
                "universe_size": usize,
                "execution_mode": "structured",
                "primary_metric": "sharpe",
            },
        },
        "tested_at": tested_at,
    }
    entry.update(extra)
    return entry


def _adopted_codegen_entry(symbol="IWM", tested_at="2026-06-20T10:00:00", **extra):
    entry = {
        "hypothesis": {
            "id": f"bl_test_code_{symbol}",
            "strategy": "codegen",
            "symbol": symbol,
            "params": {},
        },
        "backtest_result": {"code_hash": hashlib.sha256(SAMPLE_CODE.encode()).hexdigest()},
        "tested_at": tested_at,
    }
    entry.update(extra)
    return entry


def _good_manifest_response():
    """/run_manifest response shape observed live on the sandbox runner."""
    return {
        "status": "ok",
        "n_days": 45,
        "metrics": {
            "val_sharpe": 1.3,
            "val_total_return_pct": 10.0,
            "val_max_drawdown_pct": -1.0,
            "holdout_sharpe": 1.1,
            "holdout_total_return_pct": 9.0,
            "holdout_max_drawdown_pct": -2.0,
        },
    }


def _good_code_response():
    """/run_code response with walkforward splits (as returned by the runner)."""
    return {
        "strategy": "codegen",
        "symbol": "IWM",
        "walkforward": {"enabled": True},
        "holdout": {"sharpe_ratio": 0.9, "total_return_pct": 5.0,
                    "max_drawdown_pct": -3.0, "num_days": 40},
    }


def _seed_backlog(path, entries):
    """Write a backlog file with the given entry dicts (status preserved)."""
    data = {"entries": entries}
    path.write_text(json.dumps(data, indent=2))
    return path


def _manifest_backlog_entry(name, universe=None, status="done_adopted",
                            start="2020-01-01"):
    return {
        "id": str(uuid.uuid4()),
        "type": "manifest",
        "status": status,
        "created_at": "2026-07-20T10:00:00+00:00",
        "source": {"kind": "idea", "ref": name},
        "spec": _manifest_spec(name, universe, start),
    }


def _code_backlog_entry(code=None, symbol="IWM", status="done_adopted"):
    return {
        "id": str(uuid.uuid4()),
        "type": "code",
        "status": status,
        "created_at": "2026-07-20T10:00:00+00:00",
        "source": {"kind": "idea", "ref": "test"},
        "spec": {"name": "volume_confirmed_sma_iwm", "symbol": symbol,
                 "code": code or SAMPLE_CODE, "description": "test"},
    }


def _capture_posts(mock_post, responses):
    """Configure oos_monitor._post_json mock to return responses in order,
    capturing (url, payload) pairs."""
    calls = []

    def fake_post(url, payload, timeout=300):
        calls.append((url, payload))
        return responses.pop(0) if len(responses) > 1 else responses[0]

    mock_post.side_effect = fake_post
    return calls


# ---------------------------------------------------------------------------
# manifest dispatch
# ---------------------------------------------------------------------------

class TestManifestDispatch:
    def test_manifest_routes_to_run_manifest_with_start_override(self, tmp_path, monkeypatch):
        """Adopted manifest entry -> spec resolved from backlog, re-run via
        /run_manifest with data_sources[].start = adoption date."""
        today = datetime(2026, 7, 20)
        adoption = "2026-06-20T10:00:00"  # 30 days before today
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        universe = ["SPY", "GLD"]
        _seed_backlog(tmp_path / "bl.json",
                      [_manifest_backlog_entry("regime_switching_gmm", universe)])

        entry = _adopted_manifest_entry("regime_switching_gmm", universe, tested_at=adoption)

        with patch("oos_monitor._post_json") as mock_post:
            calls = _capture_posts(mock_post, [_good_manifest_response()])
            oos_record, error = run_oos_check(entry, today=today)

        assert error is None, error
        assert len(calls) == 1
        url, payload = calls[0]
        assert url.endswith("/run_manifest")
        # window override: start == adoption date, end dropped
        ohlcv = [ds for ds in payload["data_sources"] if ds.get("type") == "ohlcv"]
        assert ohlcv[0]["start"] == "2026-06-20"
        assert "end" not in ohlcv[0]
        # spec identity preserved
        assert payload["name"] == "regime_switching_gmm"
        assert payload["code_b64"] == _b64(SAMPLE_CODE)
        # record
        assert oos_record["oos_kind"] == "manifest_since_adoption"
        assert oos_record["oos_sharpe"] == 1.1
        assert oos_record["oos_return_pct"] == 9.0
        assert oos_record["window_days"] == 29  # 6/20 10:00 → 7/20 00:00
        assert oos_record["n_days"] == 45

    def test_manifest_uses_record_spec_when_present(self, tmp_path, monkeypatch):
        """Go-forward: entries carrying manifest_spec on the record are re-run
        without any backlog resolution."""
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "empty_bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        universe = ["SPY", "GLD"]
        _seed_backlog(tmp_path / "empty_bl.json", [])  # nothing resolvable

        entry = _adopted_manifest_entry("regime_switching_gmm", universe)
        entry["manifest_spec"] = _manifest_spec("regime_switching_gmm", universe)

        with patch("oos_monitor._post_json") as mock_post:
            calls = _capture_posts(mock_post, [_good_manifest_response()])
            oos_record, error = run_oos_check(entry, today=today)

        assert error is None, error
        assert len(calls) == 1
        assert calls[0][1]["name"] == "regime_switching_gmm"
        assert oos_record["oos_kind"] == "manifest_since_adoption"

    def test_manifest_unresolved_spec_skipped(self, tmp_path, monkeypatch):
        """No backlog entry and no record spec -> manifest_spec_unresolved."""
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "empty_bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        _seed_backlog(tmp_path / "empty_bl.json", [])
        entry = _adopted_manifest_entry("regime_switching_gmm")

        with patch("oos_monitor._post_json") as mock_post:
            oos_record, error = run_oos_check(entry, today=today)
        assert error == "manifest_spec_unresolved"
        assert oos_record is None
        mock_post.assert_not_called()

    def test_manifest_universe_hash_mismatch_not_reused(self, tmp_path, monkeypatch):
        """Same name but different universe must NOT be reused (the hash in the
        adopted symbol is the join key)."""
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        # backlog holds same name on a DIFFERENT universe
        _seed_backlog(tmp_path / "bl.json",
                      [_manifest_backlog_entry("regime_switching_gmm", ["AAPL", "MSFT"])])
        entry = _adopted_manifest_entry("regime_switching_gmm", ["SPY", "GLD"])

        with patch("oos_monitor._post_json") as mock_post:
            oos_record, error = run_oos_check(entry, today=today)
        assert error == "manifest_spec_unresolved"
        mock_post.assert_not_called()

    def test_manifest_runner_error_skipped(self, tmp_path, monkeypatch):
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        universe = ["SPY", "GLD"]
        _seed_backlog(tmp_path / "bl.json",
                      [_manifest_backlog_entry("regime_switching_gmm", universe)])
        entry = _adopted_manifest_entry("regime_switching_gmm", universe)

        with patch("oos_monitor._post_json") as mock_post:
            calls = _capture_posts(mock_post, [
                {"status": "error", "error_type": "code", "error": "User code crashed"}])
            oos_record, error = run_oos_check(entry, today=today)
        assert error == "runner_code"
        assert oos_record is None

    def test_manifest_insufficient_data_skipped(self, tmp_path, monkeypatch):
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        universe = ["SPY", "GLD"]
        _seed_backlog(tmp_path / "bl.json",
                      [_manifest_backlog_entry("regime_switching_gmm", universe)])
        entry = _adopted_manifest_entry("regime_switching_gmm", universe)
        resp = _good_manifest_response()
        resp["n_days"] = 5

        with patch("oos_monitor._post_json") as mock_post:
            _capture_posts(mock_post, [resp])
            oos_record, error = run_oos_check(entry, today=today)
        assert error == "insufficient_data"
        assert oos_record is None

    def test_manifest_no_ohlcv_source_skipped(self, tmp_path, monkeypatch):
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        spec = _manifest_spec("regime_switching_gmm")
        spec["data_sources"] = [{"type": "news_sentiment", "universe": ["SPY", "GLD"]}]
        bl_entry = _manifest_backlog_entry("regime_switching_gmm")
        bl_entry["spec"] = spec
        _seed_backlog(tmp_path / "bl.json", [bl_entry])
        # entry must hash to the SAME (empty) ohlcv universe as the backlog spec
        _, uhash, _ = _extract_universe_meta(spec)
        entry = _adopted_manifest_entry("regime_switching_gmm")
        entry["hypothesis"]["symbol"] = f"universe:{uhash}"

        with patch("oos_monitor._post_json") as mock_post:
            oos_record, error = run_oos_check(entry, today=today)
        assert error == "manifest_no_ohlcv_source"
        mock_post.assert_not_called()

    def test_resolve_manifest_spec_prefers_universe_match(self, tmp_path, monkeypatch):
        """_resolve_manifest_spec picks the backlog entry whose universe hashes
        to the adopted symbol's hash even when several same-name entries exist."""
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        wanted = ["SPY", "GLD"]
        _seed_backlog(tmp_path / "bl.json", [
            _manifest_backlog_entry("regime_switching_gmm", ["AAPL", "MSFT"]),
            _manifest_backlog_entry("regime_switching_gmm", wanted),
        ])
        _, uhash, _ = _extract_universe_meta(_manifest_spec("regime_switching_gmm", wanted))
        spec = _resolve_manifest_spec("regime_switching_gmm", uhash)
        assert spec is not None
        assert spec["data_sources"][0]["universe"] == wanted


# ---------------------------------------------------------------------------
# codegen dispatch
# ---------------------------------------------------------------------------

class TestCodegenDispatch:
    def test_codegen_routes_to_run_code_with_code_b64(self, tmp_path, monkeypatch):
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        _seed_backlog(tmp_path / "bl.json", [_code_backlog_entry(symbol="IWM")])
        entry = _adopted_codegen_entry("IWM")

        with patch("oos_monitor._post_json") as mock_post:
            calls = _capture_posts(mock_post, [_good_code_response()])
            oos_record, error = run_oos_check(entry, today=today)

        assert error is None, error
        assert len(calls) == 1
        url, payload = calls[0]
        assert url.endswith("/run_code")
        assert payload["symbol"] == "IWM"
        decoded = base64.b64decode(payload["code_b64"]).decode("utf-8")
        assert SAMPLE_CODE in decoded
        assert oos_record["oos_kind"] == "codegen_holdout"
        assert oos_record["oos_sharpe"] == 0.9
        assert oos_record["oos_return_pct"] == 5.0

    def test_codegen_uses_record_code_spec(self, tmp_path, monkeypatch):
        """Go-forward: code_spec on the record is used without backlog lookup."""
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "empty_bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        _seed_backlog(tmp_path / "empty_bl.json", [])
        entry = _adopted_codegen_entry("IWM")
        entry["code_spec"] = {"name": "vol_conf_sma", "symbol": "IWM", "code": SAMPLE_CODE}

        with patch("oos_monitor._post_json") as mock_post:
            calls = _capture_posts(mock_post, [_good_code_response()])
            oos_record, error = run_oos_check(entry, today=today)
        assert error is None, error
        assert calls[0][1]["symbol"] == "IWM"

    def test_codegen_unresolved_skipped(self, tmp_path, monkeypatch):
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "empty_bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        _seed_backlog(tmp_path / "empty_bl.json", [])
        entry = _adopted_codegen_entry("IWM")

        with patch("oos_monitor._post_json") as mock_post:
            oos_record, error = run_oos_check(entry, today=today)
        assert error == "code_spec_unresolved"
        mock_post.assert_not_called()

    def test_codegen_code_hash_mismatch_unresolved(self, tmp_path, monkeypatch):
        """Backlog has code for the symbol but a different hash -> unresolved."""
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        _seed_backlog(tmp_path / "bl.json",
                      [_code_backlog_entry(code="def generate_weights(d):\n    return {}\n",
                                           symbol="IWM")])
        entry = _adopted_codegen_entry("IWM")

        with patch("oos_monitor._post_json") as mock_post:
            oos_record, error = run_oos_check(entry, today=today)
        assert error == "code_spec_unresolved"
        mock_post.assert_not_called()

    def test_codegen_runner_error_skipped(self, tmp_path, monkeypatch):
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        _seed_backlog(tmp_path / "bl.json", [_code_backlog_entry(symbol="IWM")])
        entry = _adopted_codegen_entry("IWM")

        with patch("oos_monitor._post_json") as mock_post:
            calls = _capture_posts(mock_post, [
                {"error": "Sandbox runner connection error: x", "error_type": "infra"}])
            oos_record, error = run_oos_check(entry, today=today)
        assert error == "runner_infra"
        assert oos_record is None


# ---------------------------------------------------------------------------
# param dispatch unchanged
# ---------------------------------------------------------------------------

class TestParamDispatchUnchanged:
    def test_param_entry_keeps_run_backtest_metrics_since(self, monkeypatch):
        today = datetime(2026, 7, 20)
        entry = {
            "hypothesis": {"strategy": "rsi", "symbol": "TLT",
                           "params": {"rsi_window": 14}},
            "tested_at": "2026-06-20T10:00:00",
        }
        with patch("oos_monitor.run_backtest") as mock_bt:
            mock_bt.return_value = {
                "since_metrics": {"sharpe_ratio": 0.5, "total_return_pct": 10.0,
                                  "max_drawdown_pct": -5.0, "n_days": 29},
            }
            oos_record, error = run_oos_check(entry, today=today)
        assert error is None
        assert oos_record["oos_kind"] == "since_adoption"
        assert oos_record["oos_sharpe"] == 0.5
        # metrics_since == adoption date (existing contract preserved)
        mock_bt.assert_called_once()
        _, kwargs = mock_bt.call_args
        assert kwargs["metrics_since"] == "2026-06-20"


# ---------------------------------------------------------------------------
# loudness: OOS_BROKEN / summary / history
# ---------------------------------------------------------------------------

class TestMonitorLoudness:
    def _run(self, knowledge, today=None):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            result = run_oos_monitor(knowledge, today=today)
        finally:
            sys.stdout = old
        return result, buf.getvalue()

    def test_oos_broken_marker_when_nothing_checkable(self, tmp_path, monkeypatch):
        """Every adopted entry skipped with a structural error -> OOS_BROKEN."""
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "empty_bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        _seed_backlog(tmp_path / "empty_bl.json", [])
        knowledge = {"adopted": [_adopted_manifest_entry("regime_switching_gmm")]}

        with patch("oos_monitor.save_knowledge"):
            (checked, negative, skipped), output = self._run(knowledge, today=today)

        assert (checked, negative, skipped) == (0, 0, 1)
        assert "OOS_BROKEN checked=0 skipped=1 error_skips=1 adopted=1" in output
        assert "OOS_SUMMARY checked=0 negative=0 skipped=1" in output

    def test_no_oos_broken_when_only_benign_skips(self, tmp_path, monkeypatch):
        """Fresh adoptions (adoption_in_future) are benign — no OOS_BROKEN."""
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "empty_bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        _seed_backlog(tmp_path / "empty_bl.json", [])
        # adopted "in the future" — nothing to check yet, legitimately
        entry = _adopted_manifest_entry("regime_switching_gmm", tested_at="2026-07-25T10:00:00")
        knowledge = {"adopted": [entry]}

        with patch("oos_monitor.save_knowledge"):
            (checked, negative, skipped), output = self._run(knowledge, today=today)

        assert (checked, negative, skipped) == (0, 0, 1)
        assert "OOS_BROKEN" not in output

    def test_manifest_oos_history_recorded(self, tmp_path, monkeypatch):
        """A successfully checked manifest adoption appends to oos_history and
        counts as checked (not skipped)."""
        today = datetime(2026, 7, 20)
        monkeypatch.setenv("BACKLOG_PATH", str(tmp_path / "bl.json"))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")
        universe = ["SPY", "GLD"]
        _seed_backlog(tmp_path / "bl.json",
                      [_manifest_backlog_entry("regime_switching_gmm", universe)])
        entry = _adopted_manifest_entry("regime_switching_gmm", universe)
        knowledge = {"adopted": [entry]}

        with patch("oos_monitor._post_json") as mock_post, \
             patch("oos_monitor.save_knowledge"):
            _capture_posts(mock_post, [_good_manifest_response()])
            (checked, negative, skipped), output = self._run(knowledge, today=today)

        assert (checked, negative, skipped) == (1, 0, 0)
        assert "OOS_STATUS manifest:regime_switching_gmm/universe:" in output
        assert "OOS_BROKEN" not in output
        assert len(entry["oos_history"]) == 1
        assert entry["oos_history"][0]["oos_sharpe"] == 1.1


# ---------------------------------------------------------------------------
# go-forward: spec persisted on adopted records (autonomous_loop)
# ---------------------------------------------------------------------------

class TestGoForwardRecordSpec:
    def test_adopted_manifest_record_keeps_manifest_spec(self, tmp_path, monkeypatch):
        """Disk round-trip: an adopted manifest entry persists its spec on the
        knowledge record so later OOS re-runs do not depend on backlog history."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        spec = _manifest_spec("cross_sectional_momentum")
        from backlog import _new_entry
        bl.add_entry(_new_entry("manifest", 0.95, {"kind": "paper", "ref": "t.md"}, spec))

        mock_response = {
            "status": "ok",
            "n_days": 500,
            "metrics": {
                "val_sharpe": 1.5, "val_max_drawdown_pct": -5.0,
                "val_total_return_pct": 15.0,
                "holdout_sharpe": 1.0, "holdout_max_drawdown_pct": -3.0,
                "holdout_total_return_pct": 10.0,
            },
        }

        def mock_urlopen(req, timeout=300):
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.status = 200
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)
            run_loop(1)

        knowledge = json.loads(kf.read_text())
        adopted = [r for r in knowledge.get("adopted", [])
                   if r.get("hypothesis", {}).get("strategy", "").startswith("manifest:")]
        assert len(adopted) == 1, f"expected 1 adopted manifest, got {len(adopted)}"
        record = adopted[0]
        assert record.get("manifest_spec") == spec, \
            "adopted manifest record must persist the consumed spec (issue #88)"

    def test_adopted_code_record_keeps_code_spec(self, tmp_path, monkeypatch):
        """Disk round-trip: an adopted code entry persists its code on the
        knowledge record (code_spec) for later OOS re-runs."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        from backlog import make_code_entry
        bl.add_entry(make_code_entry(
            name="vol_conf_sma", description="test", code=SAMPLE_CODE,
            symbol="IWM", priority=0.9, source={"kind": "idea", "ref": "t"},
            eval_plan={"extra_criteria": []},
        ))

        mock_response = {
            "strategy": "codegen", "symbol": "IWM",
            "walkforward": {"enabled": True, "train_ratio": 0.6, "val_ratio": 0.2,
                            "holdout_ratio": 0.2},
            "in_sample": {"sharpe_ratio": 1.6, "total_return_pct": 20.0,
                          "max_drawdown_pct": -8.0, "num_days": 378, "num_trades": 10},
            "out_of_sample": {"sharpe_ratio": 1.2, "total_return_pct": 15.0,
                              "max_drawdown_pct": -10.0, "num_days": 250, "num_trades": 5},
            "holdout": {"sharpe_ratio": 0.8, "total_return_pct": 10.0,
                        "max_drawdown_pct": -5.0, "num_days": 80, "num_trades": 3},
        }

        def mock_urlopen(req, timeout=240):
            from unittest.mock import MagicMock
            body = json.loads(req.data)
            assert body["symbol"] == "IWM"
            resp = MagicMock()
            resp.status = 200
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)
            run_loop(1)

        knowledge = json.loads(kf.read_text())
        adopted = [r for r in knowledge.get("adopted", [])
                   if r.get("hypothesis", {}).get("strategy") == "codegen"]
        assert len(adopted) == 1, f"expected 1 adopted codegen, got {len(adopted)}"
        record = adopted[0]
        code_spec = record.get("code_spec")
        assert code_spec is not None, "adopted code record must persist code_spec"
        assert code_spec["symbol"] == "IWM"
        assert code_spec["code"] == SAMPLE_CODE
