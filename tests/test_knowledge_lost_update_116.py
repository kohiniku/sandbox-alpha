"""
Regression test for issue #116: knowledge.json lost-update race.

Two writers load knowledge concurrently, each appends different items to
shared list keys, then each saves. Without delta-merge, the second save
clobbers the first writer's additions. This test verifies that both
writers' changes survive.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autonomous_loop


@pytest.fixture
def tmp_knowledge_file(tmp_path, monkeypatch):
    """Create a temp knowledge.json and patch KNOWLEDGE_FILE."""
    kf = tmp_path / "knowledge.json"
    monkeypatch.setattr(autonomous_loop, "KNOWLEDGE_FILE", kf)
    return kf


def _seed_knowledge(path, data):
    """Write initial knowledge.json to disk."""
    path.write_text(json.dumps(data, default=str))


def _empty_knowledge():
    return {
        "tested": [],
        "tested_combinations": [],
        "adopted": [],
        "rejected": [],
        "superseded": [],
        "families": {},
        "iterations": 0,
        "errors": [],
        "reviews": [],
        "review_state": {},
        "near_misses": [],
        "near_misses_cross": [],
    }


class TestLostUpdateReviews:
    """Two writers each append to reviews[] — both must survive."""

    def test_concurrent_reviews_both_survive(self, tmp_knowledge_file):
        seed = _empty_knowledge()
        _seed_knowledge(tmp_knowledge_file, seed)

        # Writer A (strategy_review): load and append review
        writer_a = autonomous_loop.load_knowledge()
        writer_a.setdefault("reviews", []).append({
            "family_key": "fam_A",
            "verdict": "kill",
            "reviewed_at": "2026-08-24T03:32:00Z",
        })
        writer_a.setdefault("review_state", {})["last_review_at"] = "2026-08-24T03:32:00Z"

        # Writer B (autonomous_loop): load and append its own review BEFORE A saves
        writer_b = autonomous_loop.load_knowledge()
        writer_b.setdefault("reviews", []).append({
            "family_key": "fam_B",
            "verdict": "keep",
            "reviewed_at": "2026-08-24T03:32:05Z",
        })
        writer_b.setdefault("review_state", {})["last_review_at"] = "2026-08-24T03:32:05Z"

        # A saves first
        autonomous_loop.save_knowledge(writer_a)

        # B saves second — must NOT clobber A's review
        autonomous_loop.save_knowledge(writer_b)

        # Verify both reviews survived
        final = json.loads(tmp_knowledge_file.read_text())
        review_keys = [r["family_key"] for r in final.get("reviews", [])]
        assert "fam_A" in review_keys, "Writer A's review was clobbered"
        assert "fam_B" in review_keys, "Writer B's review was clobbered"

    def test_concurrent_adopted_and_reviews(self, tmp_knowledge_file):
        """Writer A appends to adopted[]; writer B appends to reviews[].
        Both must survive."""
        seed = _empty_knowledge()
        _seed_knowledge(tmp_knowledge_file, seed)

        # Writer A (autonomous_loop): adds to adopted and rejected
        writer_a = autonomous_loop.load_knowledge()
        writer_a["adopted"].append({
            "id": "hyp_001",
            "hypothesis": {"strategy": "sma", "symbol": "AAPL"},
            "family_key": "sma|AAPL|single",
            "sharpe_ratio": 1.5,
        })
        writer_a["tested"].append("hyp_001")

        # Writer B (strategy_review): adds to reviews (loaded concurrently)
        writer_b = autonomous_loop.load_knowledge()
        writer_b.setdefault("reviews", []).append({
            "family_key": "rsi|MSFT|single",
            "verdict": "refine",
            "reviewed_at": "2026-08-24T12:03:20Z",
        })

        # A saves first
        autonomous_loop.save_knowledge(writer_a)
        # B saves second
        autonomous_loop.save_knowledge(writer_b)

        final = json.loads(tmp_knowledge_file.read_text())
        # A's adoption survived
        assert len(final["adopted"]) == 1
        assert final["adopted"][0]["id"] == "hyp_001"
        assert "hyp_001" in final["tested"]
        # B's review survived
        assert len(final.get("reviews", [])) == 1
        assert final["reviews"][0]["family_key"] == "rsi|MSFT|single"


class TestLostUpdateOosHistory:
    """oos_monitor appends to adopted[i].oos_history while another writer
    may add new adopted entries. Both must coexist."""

    def test_oos_history_append_survives_concurrent_adoption(self, tmp_knowledge_file):
        seed = _empty_knowledge()
        seed["adopted"] = [{
            "id": "hyp_existing",
            "hypothesis": {"strategy": "sma", "symbol": "AAPL"},
            "family_key": "sma|AAPL|single",
            "oos_history": [],
        }]
        _seed_knowledge(tmp_knowledge_file, seed)

        # Writer A (oos_monitor): appends to oos_history of existing entry
        writer_a = autonomous_loop.load_knowledge()
        writer_a["adopted"][0].setdefault("oos_history", []).append({
            "date": "2026-08-24",
            "oos_sharpe": 0.5,
            "window_days": 30,
        })

        # Writer B (autonomous_loop): adds a new adopted entry
        writer_b = autonomous_loop.load_knowledge()
        writer_b["adopted"].append({
            "id": "hyp_new",
            "hypothesis": {"strategy": "rsi", "symbol": "MSFT"},
            "family_key": "rsi|MSFT|single",
            "oos_history": [],
        })
        writer_b["tested"].append("hyp_new")

        # B saves first (adds new entry)
        autonomous_loop.save_knowledge(writer_b)
        # A saves second (updates oos_history of existing entry)
        autonomous_loop.save_knowledge(writer_a)

        final = json.loads(tmp_knowledge_file.read_text())
        # Both adopted entries present
        ids = [e["id"] for e in final["adopted"]]
        assert "hyp_existing" in ids
        assert "hyp_new" in ids
        # OOS history was appended to the right entry
        existing = next(e for e in final["adopted"] if e["id"] == "hyp_existing")
        assert len(existing.get("oos_history", [])) == 1
        assert existing["oos_history"][0]["date"] == "2026-08-24"


class TestLostUpdateFamilies:
    """Two writers update different families — both changes must survive."""

    def test_concurrent_family_updates(self, tmp_knowledge_file):
        seed = _empty_knowledge()
        seed["families"] = {
            "sma|AAPL|single": {
                "n_trials": 1, "best_val_sharpe": 1.0,
                "lifecycle": "active", "family_type": "single",
            },
        }
        _seed_knowledge(tmp_knowledge_file, seed)

        # Writer A updates family sma|AAPL|single
        writer_a = autonomous_loop.load_knowledge()
        writer_a["families"]["sma|AAPL|single"]["n_trials"] = 2
        writer_a["families"]["sma|AAPL|single"]["best_val_sharpe"] = 1.5

        # Writer B updates/adds a different family
        writer_b = autonomous_loop.load_knowledge()
        writer_b["families"]["rsi|MSFT|single"] = {
            "n_trials": 1, "best_val_sharpe": 0.8,
            "lifecycle": "candidate", "family_type": "single",
        }

        autonomous_loop.save_knowledge(writer_a)
        autonomous_loop.save_knowledge(writer_b)

        final = json.loads(tmp_knowledge_file.read_text())
        # A's update survived
        assert final["families"]["sma|AAPL|single"]["n_trials"] == 2
        assert final["families"]["sma|AAPL|single"]["best_val_sharpe"] == 1.5
        # B's new family survived
        assert "rsi|MSFT|single" in final["families"]
        assert final["families"]["rsi|MSFT|single"]["best_val_sharpe"] == 0.8


class TestIdempotentSave:
    """Calling save_knowledge twice with the same data should not duplicate."""

    def test_double_save_no_duplication(self, tmp_knowledge_file):
        seed = _empty_knowledge()
        _seed_knowledge(tmp_knowledge_file, seed)

        writer = autonomous_loop.load_knowledge()
        writer["reviews"].append({
            "family_key": "fam_X",
            "verdict": "keep",
            "reviewed_at": "2026-08-24T00:00:00Z",
        })

        autonomous_loop.save_knowledge(writer)
        # Second save with same in-memory state (no new changes)
        autonomous_loop.save_knowledge(writer)

        final = json.loads(tmp_knowledge_file.read_text())
        # Should have exactly one review, not two
        assert len(final.get("reviews", [])) == 1


class TestStrictJsonOutput:
    """Saved JSON must be parseable by strict parsers (no NaN, Inf)."""

    def test_save_produces_valid_json(self, tmp_knowledge_file):
        seed = _empty_knowledge()
        seed["rejected"].append({
            "id": "hyp_nan",
            "metrics": {"val_sharpe": float("nan"), "val_max_drawdown_pct": float("inf")},
        })
        _seed_knowledge(tmp_knowledge_file, {k: v for k, v in seed.items()
                                            if k != "rejected"})
        # Write seed without NaN first (json.dumps default doesn't error)
        tmp_knowledge_file.write_text(json.dumps(seed, default=str))

        writer = autonomous_loop.load_knowledge()
        autonomous_loop.save_knowledge(writer)

        # Must be re-parseable with strict=True (allow_nan=False equivalent)
        raw = tmp_knowledge_file.read_text()
        # Should not contain bare NaN or Infinity tokens
        assert "NaN" not in raw, "Saved JSON contains bare NaN token"
        assert "Infinity" not in raw, "Saved JSON contains bare Infinity token"
        # Must parse successfully
        parsed = json.loads(raw)
        assert "rejected" in parsed
