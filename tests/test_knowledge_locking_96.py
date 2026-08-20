"""
Regression tests for issue #96: knowledge.json written without locking.

The old implementation was a bare `KNOWLEDGE_FILE.write_text(...)` — no lock,
no atomic rename. With three strategy jobs sharing one checkout and running
concurrently during catch-up bursts, that allows:

1. Readers observing a truncated/half-written file (JSONDecodeError).
2. Interleaved writers corrupting the file (mixed content, invalid JSON).
3. A mid-write kill leaving truncated JSON permanently.

The fix (autonomous_loop.py): save_knowledge serializes under an exclusive
flock on a sidecar `knowledge.json.lock` and publishes via temp + os.replace
(atomic rename); load_knowledge reads under a shared lock. Mirrors the
discipline backlog.py already had for backlog.json.
"""
import json
import os
import sys
from multiprocessing import Process
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autonomous_loop


@pytest.fixture
def tmp_knowledge(tmp_path, monkeypatch):
    kf = tmp_path / "knowledge.json"
    monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", kf)
    return kf


def _big_payload(marker, n=4000):
    """A payload large enough that a bare write_text is not atomic in practice."""
    return {
        "marker": marker,
        "families": {f"fam_{i}": {"n": i, "data": "x" * 200} for i in range(n)},
        "tested": [i for i in range(n)],
        "errors": ["e"] * 100,
    }


def _writer_loop(knowledge_path, marker, iterations):
    """Subprocess entry: repeatedly save a distinct large payload."""
    import autonomous_loop as al

    al.KNOWLEDGE_FILE = Path(knowledge_path)
    for _ in range(iterations):
        al.save_knowledge(_big_payload(marker))


class TestKnowledgeLockedAtomicWrite:
    def test_save_publishes_via_temp_and_os_replace(self, tmp_knowledge, monkeypatch):
        """The write path must be temp + atomic rename, not in-place write_text."""
        replaced = []

        real_replace = os.replace

        def spy_replace(src, dst):
            replaced.append((Path(src).name, Path(dst).name))
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy_replace)
        autonomous_loop.save_knowledge(_big_payload("spy"))
        # Exactly one publish, from a .tmp staging file onto knowledge.json.
        assert replaced == [("knowledge.json.tmp", "knowledge.json")]
        # No stray temp files left behind.
        assert not list(tmp_knowledge.parent.glob("knowledge.json.tmp*"))
        # File is complete valid JSON.
        data = json.loads(tmp_knowledge.read_text())
        assert data["marker"] == "spy"
        assert len(data["families"]) == 4000

    def test_concurrent_writers_never_corrupt_file(self, tmp_knowledge):
        """Two processes hammering save_knowledge: every write is whole."""
        iters = 25
        procs = [
            Process(target=_writer_loop, args=(str(tmp_knowledge), "A", iters)),
            Process(target=_writer_loop, args=(str(tmp_knowledge), "B", iters)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f"writer process died: {p.exitcode}"

        # Final file must parse as valid JSON and be exactly one complete
        # payload — never a truncation or interleaving of the two writers.
        raw = tmp_knowledge.read_text()
        data = json.loads(raw)  # raises JSONDecodeError if corrupted
        assert data["marker"] in ("A", "B")
        assert len(data["families"]) == 4000
        assert len(data["tested"]) == 4000

    def test_reader_never_observes_partial_file(self, tmp_knowledge):
        """A concurrent reader must never hit JSONDecodeError mid-write."""
        import threading

        stop = threading.Event()
        errors = []
        results = []

        def reader():
            while not stop.is_set():
                try:
                    data = autonomous_loop.load_knowledge()
                    results.append(data.get("marker"))
                except Exception as exc:  # noqa: BLE001 - any read failure is the bug
                    errors.append(exc)
                    stop.set()

        t = threading.Thread(target=reader)
        t.start()
        try:
            for i in range(40):
                autonomous_loop.save_knowledge(_big_payload("A" if i % 2 else "B"))
        finally:
            stop.set()
            t.join(timeout=30)

        assert errors == [], f"reader saw a partial/corrupt file: {errors[:3]}"
        # Every observed read carried a complete payload marker; None only
        # occurs for reads racing ahead of the first write (fresh state).
        assert all(m in ("A", "B", None) for m in results)

    def test_lock_sidecar_does_not_break_fresh_state(self, tmp_knowledge, monkeypatch):
        """load_knowledge with no knowledge.json (only a stray lock file) is fine."""
        lock = tmp_knowledge.with_name(tmp_knowledge.name + ".lock")
        lock.write_text("")
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_knowledge)
        data = autonomous_loop.load_knowledge()
        assert data["families"] == {}
        assert data["tested"] == []

    def test_save_then_load_roundtrip_preserves_content(self, tmp_knowledge):
        payload = _big_payload("rt")
        autonomous_loop.save_knowledge(payload)
        loaded = autonomous_loop.load_knowledge()
        assert loaded["marker"] == "rt"
        # load_knowledge runs migrations (stamps family_type etc.), so compare
        # the substantive payload fields rather than exact dict equality.
        assert len(loaded["families"]) == len(payload["families"])
        assert loaded["families"]["fam_7"]["n"] == payload["families"]["fam_7"]["n"]
        assert loaded["tested"] == payload["tested"]
