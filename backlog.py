#!/usr/bin/env python3
"""
JSON-file-backed priority queue for strategy ideation proposals.
Thread/process-safe via flock-based file locking. Pure stdlib.
"""
import fcntl
import json
import os
import uuid
from datetime import datetime, timezone

from loop_constants import BacklogStatus

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _new_entry(entry_type, priority, source, spec, eval_plan=None):
    """Create a well-formed backlog entry dict."""
    return {
        "id": str(uuid.uuid4()),
        "type": entry_type,
        "status": BacklogStatus.PENDING,
        "priority": float(priority),
        "created_at": _now_iso(),
        "source": source,
        "spec": spec,
        "eval_plan": eval_plan or {"extra_criteria": []},
        "result": None,
    }


# ---------------------------------------------------------------------------
# Backlog class
# ---------------------------------------------------------------------------

class Backlog:
    """Thread/process-safe JSON-file-backed priority queue.

    Path: env BACKLOG_PATH, default ./backlog.json
    """

    def __init__(self, path=None):
        self.path = path or os.environ.get("BACKLOG_PATH", "./backlog.json")
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({"entries": []}, f)

    # -- low-level lock helpers --

    def _locked_read(self):
        """Acquire exclusive lock, read full data, return (data, fd)."""
        fd = open(self.path, "r+")
        fcntl.flock(fd, fcntl.LOCK_EX)
        fd.seek(0)
        raw = fd.read() or '{"entries": []}'
        return json.loads(raw), fd

    def _locked_write(self, data, fd):
        """Write data + unlock + close."""
        fd.seek(0)
        fd.truncate()
        fd.write(json.dumps(data, indent=2))
        fd.flush()
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()

    # -- public API --

    def load(self):
        """Atomically load the backlog."""
        data, fd = self._locked_read()
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        return data

    def save(self, data):
        """Atomically save (overwrite) the backlog."""
        fd = open(self.path, "r+")
        fcntl.flock(fd, fcntl.LOCK_EX)
        self._locked_write(data, fd)

    def add_entry(self, entry):
        """Add an entry. Rejects exact-spec duplicates across non-archived entries.

        Returns (accepted: bool, id_or_reason: str).
        Enforces max 50 pending entries; lowest-priority pending entries
        beyond the cap are moved to 'archived'.
        """
        data, fd = self._locked_read()

        # Callers may pass entries without an id (documented as "replaced on add")
        if not entry.get("id"):
            entry["id"] = str(uuid.uuid4())

        # Dedup: identical spec vs any non-archived entry
        spec = entry["spec"]
        for e in data["entries"]:
            if e["status"] != "archived" and _spec_equal(e["spec"], spec):
                self._locked_write(data, fd)
                return False, e["id"]  # duplicate-of id

        # Set created_at if the caller didn't supply one
        if "created_at" not in entry or entry["created_at"] is None:
            entry["created_at"] = _now_iso()

        data["entries"].append(entry)

        # Cap pending entries at 50
        pending = [e for e in data["entries"] if e["status"] == BacklogStatus.PENDING]
        if len(pending) > 50:
            pending.sort(key=lambda e: (e["priority"], e["created_at"]))
            evict = pending[: len(pending) - 50]
            for e in evict:
                e["status"] = "archived"

        self._locked_write(data, fd)
        return True, entry["id"]

    def next_pending(self):
        """Return highest-priority pending entry, or None."""
        data = self.load()
        pending = [e for e in data["entries"] if e["status"] == BacklogStatus.PENDING]
        if not pending:
            return None
        pending.sort(key=lambda e: (-e["priority"], e["created_at"]))
        return pending[0]

    def mark(self, entry_id, status, result=None):
        """Update an entry's status (and optionally result)."""
        data, fd = self._locked_read()
        for e in data["entries"]:
            if e["id"] == entry_id:
                e["status"] = status
                if result is not None:
                    e["result"] = result
                break
        self._locked_write(data, fd)

    def archive_stale(self, days=14):
        """Archive pending entries older than `days` (default 14)."""
        cutoff = datetime.now(timezone.utc)
        data, fd = self._locked_read()
        for e in data["entries"]:
            if e["status"] != BacklogStatus.PENDING:
                continue
            try:
                created = datetime.fromisoformat(e["created_at"])
            except (ValueError, KeyError):
                continue
            if (cutoff - created).days >= days:
                e["status"] = "archived"
        self._locked_write(data, fd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec_equal(s1, s2):
    """Two specs are identical if their JSON-sorted representation matches."""
    return json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_param_entry(strategy, symbol, params, priority, source, eval_plan=None):
    """Shortcut for type='param' entries."""
    spec = {"strategy": strategy, "symbol": symbol, "params": params}
    return _new_entry("param", priority, source, spec, eval_plan)


def make_code_entry(name, description, code, symbol, priority, source, eval_plan=None):
    """Shortcut for type='code' entries."""
    spec = {
        "name": name,
        "description": description,
        "code": code,
        "symbol": symbol,
    }
    return _new_entry("code", priority, source, spec, eval_plan)
