"""
Tests for scripts/characterize_gate_v2.py (PR #3c).
"""
import json
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CHARACTERIZE_SCRIPT = str(REPO_ROOT / "scripts" / "characterize_gate_v2.py")


def _make_knowledge_json(near_misses=None):
    """Construct a minimal knowledge.json with synthetic near_misses."""
    nm = near_misses or [
        {
            "id": "hyp_001",
            "strategy": "sma_crossover",
            "symbol": "AAPL",
            "params": {"fast_window": 10, "slow_window": 30},
            "val_sharpe": 0.45,
            "deflated_threshold": 0.50,
            "holdout_sharpe": 0.3,
            "failed_gate": "val_sharpe_90pct",
            "date": "2025-07-01T00:00:00Z",
        },
        {
            "id": "hyp_002",
            "strategy": "momentum",
            "symbol": "MSFT",
            "params": {"lookback": 20, "hold_period": 5},
            "val_sharpe": 0.48,
            "deflated_threshold": 0.50,
            "holdout_sharpe": 0.35,
            "failed_gate": "val_sharpe_90pct",
            "date": "2025-07-02T00:00:00Z",
        },
        {
            "id": "hyp_003",
            "strategy": "rsi",
            "symbol": "NVDA",
            "params": {"rsi_window": 14, "oversold": 30, "overbought": 70},
            "val_sharpe": 0.60,
            "deflated_threshold": 0.55,
            "holdout_sharpe": 0.20,
            "failed_gate": "holdout",
            "date": "2025-07-03T00:00:00Z",
        },
    ]
    return {
        "tested": [],
        "tested_combinations": [],
        "adopted": [],
        "rejected": [],
        "superseded": [],
        "families": {},
        "iterations": 10,
        "errors": [],
        "near_misses": nm,
    }


def test_characterize_produces_summary_line(tmp_path, monkeypatch):
    """Script with --dry-run produces SUMMARY line with expected counts."""
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(json.dumps(_make_knowledge_json()))

    monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://localhost:9999")

    result = subprocess.run(
        [sys.executable, CHARACTERIZE_SCRIPT,
         "--knowledge", str(knowledge_path),
         "--dry-run"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )

    stdout = result.stdout
    stderr = result.stderr

    assert "SUMMARY" in stdout, f"No SUMMARY line. stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "total=" in stdout
    assert "adopted_reversals=" in stdout
    assert "unchanged=" in stdout

    # With 3 near_misses, total should be 3
    assert "total=3" in stdout, f"Expected total=3, got: {stdout}"


def test_characterize_dry_run_output_has_table(monkeypatch, tmp_path):
    """--dry-run output contains markdown table rows."""
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(json.dumps(_make_knowledge_json()))

    monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://localhost:9999")

    result = subprocess.run(
        [sys.executable, CHARACTERIZE_SCRIPT,
         "--knowledge", str(knowledge_path),
         "--dry-run"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )

    stdout = result.stdout
    # Should have table header and data rows
    assert "| strategy/symbol |" in stdout
    assert "sma_crossover/AAPL" in stdout
    assert "momentum/MSFT" in stdout
    assert "rsi/NVDA" in stdout


def test_characterize_limit_respects(monkeypatch, tmp_path):
    """--limit N only processes N entries."""
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(json.dumps(_make_knowledge_json()))

    monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://localhost:9999")

    result = subprocess.run(
        [sys.executable, CHARACTERIZE_SCRIPT,
         "--knowledge", str(knowledge_path),
         "--dry-run", "--limit", "2"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )

    stdout = result.stdout
    assert "total=2" in stdout, f"Expected total=2 with --limit 2, got: {stdout}"


def test_characterize_no_near_misses(monkeypatch, tmp_path, capsys):
    """Script with empty near_misses prints message and exits cleanly."""
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(json.dumps(_make_knowledge_json(near_misses=[])))

    monkeypatch.setenv("SANDBOX_RUNNER_URL", "http://localhost:9999")

    result = subprocess.run(
        [sys.executable, CHARACTERIZE_SCRIPT,
         "--knowledge", str(knowledge_path),
         "--dry-run"],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )

    assert "No near_misses found" in result.stdout or result.returncode == 0


def test_characterize_missing_runner_url_exits(monkeypatch, tmp_path):
    """Without SANDBOX_RUNNER_URL and no --dry-run, script exits with error."""
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(json.dumps(_make_knowledge_json()))

    # Unset runner url
    monkeypatch.delenv("SANDBOX_RUNNER_URL", raising=False)

    result = subprocess.run(
        [sys.executable, CHARACTERIZE_SCRIPT,
         "--knowledge", str(knowledge_path)],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode != 0, "Should exit non-zero without runner URL"
