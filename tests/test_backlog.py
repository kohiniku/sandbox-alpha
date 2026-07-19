"""
Tests for backlog.py — no network, no external deps.
"""
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backlog import (
    Backlog,
    _new_entry,
    _spec_equal,
    make_param_entry,
    make_code_entry,
)


@pytest.fixture
def tmp_backlog():
    """Create a Backlog pointing at a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    bl = Backlog(path)
    yield bl
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def sample_param_entry():
    return make_param_entry(
        strategy="momentum",
        symbol="AAPL",
        params={"lookback": 20, "hold_period": 5},
        priority=0.8,
        source={"kind": "paper", "ref": "test-research.md"},
    )


@pytest.fixture
def sample_code_entry():
    return make_code_entry(
        name="adaptive_ma",
        description="Adaptive moving average crossover",
        code="import numpy as np\n\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)",
        symbol="SPY",
        priority=0.6,
        source={"kind": "idea", "ref": "adaptive-smoothing"},
    )


# ---------------------------------------------------------------------------
# Basic CRUD + load/save
# ---------------------------------------------------------------------------

def test_new_backlog_starts_empty(tmp_backlog):
    data = tmp_backlog.load()
    assert data == {"entries": []}


def test_add_and_load_roundtrip(tmp_backlog, sample_param_entry):
    ok, eid = tmp_backlog.add_entry(sample_param_entry)
    assert ok is True
    data = tmp_backlog.load()
    assert len(data["entries"]) == 1
    assert data["entries"][0]["id"] == eid
    assert data["entries"][0]["status"] == "pending"


def test_save_overwrites(tmp_backlog, sample_param_entry):
    tmp_backlog.add_entry(sample_param_entry)
    # Direct save
    tmp_backlog.save({"entries": []})
    data = tmp_backlog.load()
    assert data["entries"] == []


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def test_dedup_identical_spec_rejected(tmp_backlog, sample_param_entry):
    ok1, eid1 = tmp_backlog.add_entry(sample_param_entry)
    assert ok1 is True

    # Same spec
    dup = make_param_entry(
        strategy="momentum",
        symbol="AAPL",
        params={"lookback": 20, "hold_period": 5},
        priority=0.9,  # different priority — still duplicate
        source={"kind": "idea", "ref": "something-else"},
    )
    ok2, reason = tmp_backlog.add_entry(dup)
    assert ok2 is False
    assert reason == eid1  # points to original


def test_dedup_different_symbol_allowed(tmp_backlog, sample_param_entry):
    tmp_backlog.add_entry(sample_param_entry)

    e2 = make_param_entry(
        strategy="momentum",
        symbol="MSFT",  # different symbol
        params={"lookback": 20, "hold_period": 5},
        priority=0.7,
        source={"kind": "paper", "ref": "test.md"},
    )
    ok, _ = tmp_backlog.add_entry(e2)
    assert ok is True


def test_dedup_archived_entry_does_not_block(tmp_backlog, sample_param_entry):
    ok, eid = tmp_backlog.add_entry(sample_param_entry)
    assert ok is True
    tmp_backlog.mark(eid, "archived")

    # Same spec again — should be allowed since first is archived
    dup = make_param_entry(
        strategy="momentum",
        symbol="AAPL",
        params={"lookback": 20, "hold_period": 5},
        priority=0.5,
        source={"kind": "idea", "ref": "retry"},
    )
    ok2, _ = tmp_backlog.add_entry(dup)
    assert ok2 is True


def test_dedup_code_spec(tmp_backlog, sample_code_entry):
    ok, _ = tmp_backlog.add_entry(sample_code_entry)
    assert ok is True

    dup = make_code_entry(
        name="adaptive_ma",
        description="Adaptive moving average crossover",
        code="import numpy as np\n\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)",
        symbol="SPY",
        priority=0.3,
        source={"kind": "idea", "ref": "retry"},
    )
    ok2, _ = tmp_backlog.add_entry(dup)
    assert ok2 is False


# ---------------------------------------------------------------------------
# Priority ordering (next_pending)
# ---------------------------------------------------------------------------

def test_next_pending_highest_first(tmp_backlog):
    entries = []
    for pri, sym in [(0.3, "AAPL"), (0.9, "MSFT"), (0.5, "GOOGL")]:
        e = make_param_entry("momentum", sym, {"lookback": 10, "hold_period": 3}, pri, {"kind": "idea", "ref": "x"})
        ok, _ = tmp_backlog.add_entry(e)
        assert ok is True

    next_e = tmp_backlog.next_pending()
    assert next_e is not None
    assert next_e["priority"] == 0.9
    assert next_e["spec"]["symbol"] == "MSFT"


def test_next_pending_empty(tmp_backlog):
    assert tmp_backlog.next_pending() is None


def test_next_pending_skips_non_pending(tmp_backlog, sample_param_entry):
    ok, eid = tmp_backlog.add_entry(sample_param_entry)
    assert ok is True
    tmp_backlog.mark(eid, "testing")
    assert tmp_backlog.next_pending() is None


# ---------------------------------------------------------------------------
# Status transitions (mark)
# ---------------------------------------------------------------------------

def test_mark_status(tmp_backlog, sample_param_entry):
    ok, eid = tmp_backlog.add_entry(sample_param_entry)
    tmp_backlog.mark(eid, "testing")
    data = tmp_backlog.load()
    assert data["entries"][0]["status"] == "testing"


def test_mark_with_result(tmp_backlog, sample_param_entry):
    ok, eid = tmp_backlog.add_entry(sample_param_entry)
    result = {"verdict": "adopted", "summary": "Good results", "finished_at": "2026-07-19T00:00:00Z"}
    tmp_backlog.mark(eid, "done_adopted", result)
    data = tmp_backlog.load()
    assert data["entries"][0]["status"] == "done_adopted"
    assert data["entries"][0]["result"]["verdict"] == "adopted"


# ---------------------------------------------------------------------------
# Cap eviction (50 pending max)
# ---------------------------------------------------------------------------

def test_cap_eviction_lowest_priority(tmp_backlog):
    # Add 55 pending entries
    for i in range(55):
        e = make_param_entry(
            "momentum", f"SYM{i:03d}",
            {"lookback": 10, "hold_period": 3},
            priority=float(i) / 100.0,  # 0.00 to 0.54
            source={"kind": "idea", "ref": f"test-{i}"},
        )
        tmp_backlog.add_entry(e)

    data = tmp_backlog.load()
    pending = [e for e in data["entries"] if e["status"] == "pending"]
    archived = [e for e in data["entries"] if e["status"] == "archived"]

    assert len(pending) == 50
    assert len(archived) == 5

    # Archived should be the 5 lowest priorities
    archived_priorities = [e["priority"] for e in archived]
    assert all(p < 0.05 for p in archived_priorities)


def test_cap_eviction_stable_at_50(tmp_backlog):
    # Add exactly 50
    for i in range(50):
        e = make_param_entry("momentum", f"SYM{i:03d}", {"lookback": 10, "hold_period": 3}, 0.5, {"kind": "idea", "ref": f"t-{i}"})
        tmp_backlog.add_entry(e)

    data = tmp_backlog.load()
    assert len([e for e in data["entries"] if e["status"] == "pending"]) == 50
    assert len([e for e in data["entries"] if e["status"] == "archived"]) == 0


# ---------------------------------------------------------------------------
# Stale archiving
# ---------------------------------------------------------------------------

def test_archive_stale_old_entries(tmp_backlog):
    # Insert an entry with an old created_at via direct data manipulation
    old_entry = make_param_entry("momentum", "AAPL", {"lookback": 10, "hold_period": 3}, 0.5, {"kind": "idea", "ref": "old"})
    ok, eid = tmp_backlog.add_entry(old_entry)
    assert ok

    # Manually set created_at to 20 days ago
    old_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    data = tmp_backlog.load()
    for e in data["entries"]:
        if e["id"] == eid:
            e["created_at"] = old_date
    tmp_backlog.save(data)

    # Add a fresh entry
    fresh = make_param_entry("momentum", "MSFT", {"lookback": 10, "hold_period": 3}, 0.5, {"kind": "idea", "ref": "fresh"})
    tmp_backlog.add_entry(fresh)

    tmp_backlog.archive_stale(days=14)

    data = tmp_backlog.load()
    for e in data["entries"]:
        if e["id"] == eid:
            assert e["status"] == "archived"
        else:
            assert e["status"] == "pending"


def test_archive_stale_fresh_untouched(tmp_backlog, sample_param_entry):
    tmp_backlog.add_entry(sample_param_entry)
    tmp_backlog.archive_stale(days=14)
    data = tmp_backlog.load()
    assert data["entries"][0]["status"] == "pending"


# ---------------------------------------------------------------------------
# Lock roundtrip (concurrency stress test — verify no corruption)
# ---------------------------------------------------------------------------

def test_lock_roundtrip_many_adds(tmp_backlog):
    """Add 100 entries sequentially — verify data integrity."""
    for i in range(100):
        e = make_param_entry("momentum", f"SYM{i:03d}", {"lookback": 10, "hold_period": 3}, 0.5, {"kind": "idea", "ref": f"t-{i}"})
        tmp_backlog.add_entry(e)

    data = tmp_backlog.load()
    # Should have exactly 100 entries (50 pending + 50 archived due to cap)
    assert len(data["entries"]) == 100


# ---------------------------------------------------------------------------
# _spec_equal helper
# ---------------------------------------------------------------------------

def test_spec_equal_identical():
    s1 = {"strategy": "momentum", "symbol": "AAPL", "params": {"a": 1, "b": 2}}
    s2 = {"symbol": "AAPL", "params": {"b": 2, "a": 1}, "strategy": "momentum"}
    assert _spec_equal(s1, s2) is True


def test_spec_equal_different():
    s1 = {"strategy": "momentum", "symbol": "AAPL", "params": {"a": 1}}
    s2 = {"strategy": "momentum", "symbol": "AAPL", "params": {"a": 2}}
    assert _spec_equal(s1, s2) is False


# ---------------------------------------------------------------------------
# _new_entry / make_* helpers
# ---------------------------------------------------------------------------

def test_new_entry_has_required_keys():
    e = _new_entry("param", 0.5, {"kind": "idea", "ref": "x"}, {"strategy": "m", "symbol": "A", "params": {}})
    for key in ("id", "type", "status", "priority", "created_at", "source", "spec", "eval_plan", "result"):
        assert key in e
    assert e["type"] == "param"
    assert e["status"] == "pending"
    assert e["priority"] == 0.5


def test_make_param_entry_shortcut():
    e = make_param_entry("momentum", "SPY", {"lookback": 10, "hold_period": 3}, 0.7, {"kind": "paper", "ref": "r.md"})
    assert e["type"] == "param"
    assert e["spec"]["strategy"] == "momentum"
    assert e["spec"]["symbol"] == "SPY"
    assert e["spec"]["params"] == {"lookback": 10, "hold_period": 3}


def test_make_code_entry_shortcut():
    e = make_code_entry("test", "desc", "def generate_signals(df): pass", "SPY", 0.5, {"kind": "idea", "ref": "x"})
    assert e["type"] == "code"
    assert e["spec"]["name"] == "test"
    assert e["spec"]["code"] == "def generate_signals(df): pass"
