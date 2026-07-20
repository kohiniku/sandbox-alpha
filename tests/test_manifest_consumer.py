"""
Tests for manifest entry consumption in autonomous_loop.py (Phase 1 PR-G).
No network — /run_manifest POSTs are monkeypatched.
"""

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backlog import Backlog, _new_entry
from autonomous_loop import (
    load_knowledge,
    _evaluate_manifest_result,
    _classify_near_miss,
    compute_effective_min_sharpe,
    MAX_DRAWDOWN_LIMIT,
    MIN_SHARPE_BASE,
)


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def _b64(code: str) -> str:
    return base64.b64encode(code.encode("utf-8")).decode("ascii")


SAMPLE_CODE = 'import numpy as np\nimport pandas as pd\n\ndef generate_signals(data):\n    return pd.DataFrame(1, index=next(iter(data.values())).index, columns=list(data.keys()))\n'


def _make_manifest_spec(name="cross_sectional_momentum", universe=None,
                        execution_mode="structured", extras=None):
    """Build a minimal valid manifest dict as stored by ideation."""
    if universe is None:
        universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    spec = {
        "name": name,
        "code_b64": _b64(SAMPLE_CODE),
        "data_sources": [
            {"type": "ohlcv", "universe": universe, "start": "2020-01-01"}
        ],
        "model_artifacts": [],
        "compute": {"mode": "inference", "budget_seconds": 60, "gpu": False},
        "evaluator": {
            "type": "portfolio",
            "metrics": ["sharpe", "max_drawdown_pct", "total_return_pct"],
        },
        "execution_mode": execution_mode,
    }
    if extras:
        spec["evaluator"]["extras"] = extras
    return spec


def _make_manifest_entry(spec=None, priority=0.9, source=None,
                          eval_plan=None):
    """Create a backlog entry of type='manifest'."""
    if spec is None:
        spec = _make_manifest_spec()
    if source is None:
        source = {"kind": "paper", "ref": "test_paper.md"}
    return _new_entry("manifest", priority, source, spec, eval_plan)


def _ok_runner_response(metrics_overrides=None, manifest_name="test_strat",
                        universe_size=5, n_days=252, execution_mode="structured",
                        expert_extras=None):
    """Build a runner /run_manifest response with status='ok'."""
    metrics = {
        "val_sharpe": 1.2,
        "val_max_drawdown_pct": -10.0,
        "val_total_return_pct": 15.0,
        "holdout_sharpe": 0.8,
        "holdout_max_drawdown_pct": -8.0,
        "holdout_total_return_pct": 10.0,
    }
    if metrics_overrides:
        metrics.update(metrics_overrides)
    resp = {
        "status": "ok",
        "manifest_name": manifest_name,
        "universe_size": universe_size,
        "n_days": n_days,
        "execution_mode": execution_mode,
        "metrics": metrics,
        "config": {"weighting": "equal_active_signals", "benchmark": None},
        "train_end": "2024-01-01",
        "val_end": "2024-06-01",
    }
    if expert_extras:
        resp["expert_extras"] = expert_extras
    return resp


def _error_runner_response(error_type, error_msg):
    """Build a runner /run_manifest response with status='error'."""
    return {
        "status": "error",
        "error_type": error_type,
        "error": error_msg,
    }


# ──────────────────────────────────────────────
# Test (a): valid structured manifest passes gates → adopted
# ──────────────────────────────────────────────

class TestManifestAdopted:
    def test_manifest_passes_gates_adopted(self, tmp_path, monkeypatch):
        """A valid manifest that passes all gates → done_adopted."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = _make_manifest_entry(priority=0.95)
        bl.add_entry(entry)

        mock_response = _ok_runner_response()

        def mock_urlopen(req, timeout=300):
            body = json.loads(req.data)
            assert body.get("name") == "cross_sectional_momentum"
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"
        assert data["entries"][0]["result"]["verdict"] == "adopted"

        # Verify knowledge has source attribution
        knowledge = json.loads(kf.read_text())
        found = False
        for rec in knowledge.get("adopted", []):
            if rec.get("source"):
                found = True
                break
        assert found, "Knowledge record should have source attribution"

        # family_key should be manifest:... format
        families = knowledge.get("families", {})
        assert len(families) > 0
        family_key = list(families.keys())[0]
        assert "manifest:" in family_key
        assert "universe:" in family_key


# ──────────────────────────────────────────────
# Test (b): valid manifest fails holdout → rejected + near_miss
# ──────────────────────────────────────────────

class TestManifestHoldoutFail:
    def test_manifest_fails_holdout_rejected_near_miss(self, tmp_path, monkeypatch):
        """Manifest passes val but fails holdout → done_rejected, near_miss recorded."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = _make_manifest_entry(priority=0.8)
        bl.add_entry(entry)

        # Good val Sharpe, bad holdout
        mock_response = _ok_runner_response(metrics_overrides={
            "val_sharpe": 1.2,
            "val_max_drawdown_pct": -10.0,
            "val_total_return_pct": 15.0,
            "holdout_sharpe": -0.1,  # fails holdout gate
            "holdout_max_drawdown_pct": -25.0,
            "holdout_total_return_pct": -5.0,
        })

        def mock_urlopen(req, timeout=300):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_rejected"
        assert data["entries"][0]["result"]["verdict"] == "rejected"

        # near_miss should be recorded
        knowledge = json.loads(kf.read_text())
        near_misses = knowledge.get("near_misses", [])
        assert len(near_misses) == 1
        assert near_misses[0]["failed_gate"] == "holdout"


# ──────────────────────────────────────────────
# Test (c): valid manifest fails val_sharpe below deflated threshold
#            → rejected, no near_miss unless >=90% threshold
# ──────────────────────────────────────────────

class TestManifestValSharpeFail:
    def test_manifest_fails_val_sharpe_below_threshold(self, tmp_path, monkeypatch):
        """val_sharpe way below deflated threshold → rejected, no near_miss."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = _make_manifest_entry(priority=0.5)
        bl.add_entry(entry)

        # val_sharpe=0.2 is well below effective_min_sharpe (MIN_SHARPE_BASE=0.5)
        mock_response = _ok_runner_response(metrics_overrides={
            "val_sharpe": 0.2,
            "val_max_drawdown_pct": -10.0,
            "val_total_return_pct": 3.0,
            "holdout_sharpe": 0.1,
            "holdout_max_drawdown_pct": -15.0,
            "holdout_total_return_pct": 2.0,
        }, n_days=252)

        def mock_urlopen(req, timeout=300):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_rejected"

        # No near_miss because 0.2 is way below 90% of deflated threshold
        knowledge = json.loads(kf.read_text())
        near_misses = knowledge.get("near_misses", [])
        assert len(near_misses) == 0

    def test_manifest_near_miss_at_90pct_threshold(self, tmp_path, monkeypatch):
        """val_sharpe at 90% of deflated threshold → near_miss recorded."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = _make_manifest_entry(priority=0.5)
        bl.add_entry(entry)

        # Effective min sharpe for N=0, T=252: sqrt(2*ln(2))*sqrt(252/252) ≈ 1.18
        # 90% ≈ 1.06. Use val_sharpe=1.06 which is between 0.9*eff and eff.
        eff_min = compute_effective_min_sharpe(0, 252)  # N_family=0, but max(N,2)=2
        near_miss_val = eff_min * 0.95  # between 90% and 100%
        mock_response = _ok_runner_response(metrics_overrides={
            "val_sharpe": near_miss_val,
            "val_max_drawdown_pct": -40.0,  # fails DD gate even if sharpe is close
            "val_total_return_pct": 5.0,
            "holdout_sharpe": 0.5,
            "holdout_max_drawdown_pct": -20.0,
            "holdout_total_return_pct": 3.0,
        }, n_days=252)

        def mock_urlopen(req, timeout=300):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
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
        near_misses = knowledge.get("near_misses", [])
        assert len(near_misses) >= 1
        nm = near_misses[0]
        assert nm["failed_gate"] == "val_sharpe_90pct"


# ──────────────────────────────────────────────
# Test (d): runner returns status='error' error_type='manifest' → code_error
# ──────────────────────────────────────────────

class TestManifestRunnerCodeError:
    def test_runner_manifest_error_becomes_code_error(self, tmp_path, monkeypatch):
        """Runner reports manifest validation error → code_error, rejected count unchanged."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = _make_manifest_entry(priority=0.9)
        bl.add_entry(entry)

        error_resp = _error_runner_response("manifest", "code_b64 is not valid base64")

        def mock_urlopen(req, timeout=300):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(error_resp).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_error"
        assert data["entries"][0]["result"]["verdict"] == "code_error"

        # Rejected count should be zero (errors ≠ rejections)
        knowledge = json.loads(kf.read_text())
        assert len(knowledge.get("rejected", [])) == 0
        # But there should be an error record
        assert len(knowledge.get("errors", [])) == 1
        assert knowledge["errors"][0]["evaluation"]["error_type"] == "code"


# ──────────────────────────────────────────────
# Test (e): runner returns status='error' error_type='infra' → retried, then done_error
# ──────────────────────────────────────────────

class TestManifestRunnerInfraError:
    def test_runner_infra_error_retried_3_times_then_done_error(self, tmp_path, monkeypatch):
        """Infra errors are retried up to 3 times, then marked done_error."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = _make_manifest_entry(priority=0.9)
        bl.add_entry(entry)

        error_resp = _error_runner_response("infra", "container not available")

        call_count = [0]

        def mock_urlopen(req, timeout=300):
            call_count[0] += 1
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(error_resp).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            # Run 3 iterations to consume the same entry 3 times (retries)
            run_loop(3)

        # Should have tried 3 times (each iteration consumes the retried entry)
        # Actually: run_loop(3) does 3 iterations, each tries to consume next_pending.
        # After first attempt (infra error, attempt 1): marked pending, attempts=1
        # After second attempt (infra error, attempt 2): marked pending, attempts=2
        # After third attempt (infra error, attempt 3 >= 3): marked done_error
        # So 3 URL calls total, and final status is done_error
        assert call_count[0] == 3

        data = bl.load()
        assert data["entries"][0]["status"] == "done_error"
        assert data["entries"][0]["result"]["verdict"] == "error"


# ──────────────────────────────────────────────
# Test (f): expert-mode manifest with extras → extras recorded
# ──────────────────────────────────────────────

class TestManifestExpertExtras:
    def test_expert_mode_extras_recorded(self, tmp_path, monkeypatch):
        """Expert-mode manifest runner response includes expert_extras → recorded in evaluation."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        spec = _make_manifest_spec(execution_mode="expert", extras={"custom_flag": True})
        entry = _make_manifest_entry(spec=spec, priority=0.95)
        bl.add_entry(entry)

        expert_extras = {"my_extra_metric": 42.5, "factor_exposure": {"market": 0.3}}
        mock_response = _ok_runner_response(
            execution_mode="expert",
            expert_extras=expert_extras,
        )

        def mock_urlopen(req, timeout=300):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_response).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"

        knowledge = json.loads(kf.read_text())
        adopted = knowledge.get("adopted", [])
        assert len(adopted) == 1
        evaluation = adopted[0].get("evaluation", {})
        assert "expert_extras" in evaluation
        assert evaluation["expert_extras"] == expert_extras


# ──────────────────────────────────────────────
# Test: malformed runner response → infra error
# ──────────────────────────────────────────────

class TestMalformedRunnerResponse:
    def test_missing_status_is_infra_error(self, tmp_path, monkeypatch):
        """Runner response without 'status' field → infra error."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = _make_manifest_entry(priority=0.9)
        bl.add_entry(entry)

        # Malformed: no 'status' key
        malformed = {"metrics": {"val_sharpe": 1.0}}

        def mock_urlopen(req, timeout=300):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(malformed).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(3)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_error"

    def test_missing_metrics_is_ok_but_gate_fail(self, tmp_path, monkeypatch):
        """Runner response with status='ok' but no metrics → infra error."""
        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = _make_manifest_entry(priority=0.9)
        bl.add_entry(entry)

        # status='ok' but no metrics key
        malformed = {"status": "ok", "manifest_name": "test", "n_days": 252}

        def mock_urlopen(req, timeout=300):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(malformed).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(3)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_error"


# ──────────────────────────────────────────────
# Unit test: _evaluate_manifest_result standalone
# ──────────────────────────────────────────────

class TestEvaluateManifestResultUnit:
    def test_passes_all_gates(self):
        """Standalone _evaluate_manifest_result with good metrics."""
        knowledge = {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
                     "superseded": [], "families": {}, "iterations": 0}
        runner_result = _ok_runner_response(n_days=252)
        hypothesis = {
            "id": "test_id",
            "strategy": "manifest:test_strat",
            "symbol": "universe:abcdef01",
            "params": {"universe_size": 5, "execution_mode": "structured", "primary_metric": "sharpe"},
        }

        verdict, evaluation = _evaluate_manifest_result(runner_result, hypothesis, knowledge)
        assert verdict == "adopted"
        assert evaluation["gate_results"]["validation"] is True
        assert evaluation["gate_results"]["holdout"] is True

    def test_fails_holdout(self):
        """val passes but holdout fails."""
        knowledge = {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
                     "superseded": [], "families": {}, "iterations": 0}
        runner_result = _ok_runner_response(metrics_overrides={
            "holdout_sharpe": -0.1,
            "holdout_total_return_pct": -5.0,
        }, n_days=252)
        hypothesis = {
            "id": "test_id",
            "strategy": "manifest:test_strat",
            "symbol": "universe:abcdef01",
            "params": {"universe_size": 5, "execution_mode": "structured"},
        }

        verdict, evaluation = _evaluate_manifest_result(runner_result, hypothesis, knowledge)
        assert verdict == "rejected"
        assert evaluation["gate_results"]["validation"] is True
        assert evaluation["gate_results"]["holdout"] is False

    def test_fails_val_drawdown(self):
        """val_sharpe OK but drawdown too severe."""
        knowledge = {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
                     "superseded": [], "families": {}, "iterations": 0}
        runner_result = _ok_runner_response(metrics_overrides={
            "val_max_drawdown_pct": -35.0,
        }, n_days=252)
        hypothesis = {
            "id": "test_id",
            "strategy": "manifest:test_strat",
            "symbol": "universe:abcdef01",
            "params": {"universe_size": 5, "execution_mode": "structured"},
        }

        verdict, evaluation = _evaluate_manifest_result(runner_result, hypothesis, knowledge)
        assert verdict == "rejected"
        assert evaluation["gate_results"]["validation"] is False


# ──────────────────────────────────────────────
# Regression: param and code entries still work
# ──────────────────────────────────────────────

class TestRegressionExistingTypes:
    SAMPLE_CODE = "import numpy as np\nimport pandas as pd\n\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)\n"

    def test_param_entry_still_works(self, tmp_path, monkeypatch):
        """Param-type entries still route to /run and evaluate correctly."""
        from backlog import make_param_entry

        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = make_param_entry(
            strategy="sma_crossover", symbol="AAPL",
            params={"fast_window": 10, "slow_window": 30},
            priority=0.9, source={"kind": "paper", "ref": "test.md"},
            eval_plan={"extra_criteria": []},
        )
        bl.add_entry(entry)

        # Build a response that passes gates
        from tests.test_backlog_consumption import _known_good_result
        mock_result = _known_good_result()

        def mock_urlopen(req, timeout=180):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_result).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"

    def test_code_entry_still_works(self, tmp_path, monkeypatch):
        """Code-type entries still route to /run_code and evaluate correctly."""
        from backlog import make_code_entry

        backlog_path = tmp_path / "test_backlog.json"
        monkeypatch.setenv("BACKLOG_PATH", str(backlog_path))
        monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://mock-runner:9000")

        bl = Backlog(str(backlog_path))
        entry = make_code_entry(
            name="test_strat", description="test",
            code=self.SAMPLE_CODE, symbol="SPY",
            priority=0.88, source={"kind": "idea", "ref": "test"},
            eval_plan={"extra_criteria": []},
        )
        bl.add_entry(entry)

        from tests.test_backlog_consumption import _known_good_result
        mock_result = _known_good_result(symbol="SPY", params={})

        def mock_urlopen(req, timeout=240):
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = json.dumps(mock_result).encode("utf-8")
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from autonomous_loop import run_loop
            kf = tmp_path / "knowledge.json"
            monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
            monkeypatch.setattr("autonomous_loop.RESULTS_DIR", tmp_path / "results")
            (tmp_path / "results").mkdir(exist_ok=True)

            run_loop(1)

        data = bl.load()
        assert data["entries"][0]["status"] == "done_adopted"
