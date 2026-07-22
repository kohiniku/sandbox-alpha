"""
Migration tests for PR 4d knowledge schema changes:
- family_type stamp on legacy family entries
- near_misses_cross list introduction
- idempotency on re-load
"""
import json
import sys
import os
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_loop import load_knowledge


@pytest.fixture
def tmp_knowledge_file():
    """Create a temp knowledge.json file and patch KNOWLEDGE_FILE."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="test_knowledge_")
    os.close(fd)
    yield Path(path)
    try:
        os.unlink(path)
    except OSError:
        pass


class TestKnowledgeMigration:
    def test_family_type_stamp_on_legacy_families(self, tmp_knowledge_file, monkeypatch):
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_knowledge_file)
        loss = "MISSING_METRIC"  # MISSING_METRIC as a string
        legacy = {
            "families": {
                "sma_crossover|AAPL": {
                    "n_trials": 3,
                    "best_val_sharpe": 1.2,
                    "best_params": {},
                    "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                                      "duplicate_cluster": 0, "exhausted_cluster": 0},
                    "last_tried": "2026-01-01",
                },
                "mean_reversion|MSFT": {
                    "n_trials": 1,
                    "best_val_sharpe": 0.5,
                    "best_params": {"window": 20},
                    "gate_failures": {"validation": 1, "deflation": 0, "holdout": 0,
                                      "duplicate_cluster": 0, "exhausted_cluster": 0},
                    "last_tried": "2026-01-02",
                },
                "manifest:xs_momentum|universe:abc123": {
                    "n_trials": 2,
                    "best_val_sharpe": 0.8,
                    "best_params": {"lookback": 60},
                    "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                                      "duplicate_cluster": 0, "exhausted_cluster": 0},
                    "last_tried": "2026-01-03",
                },
            },
            "adopted": [],
            "rejected": [],
            "superseded": [],
            "tested_combinations": [],
        }
        # Legacy file: no family_type fields, no near_misses_cross
        tmp_knowledge_file.write_text(json.dumps(legacy))

        # First load — should stamp family_type and add near_misses_cross
        # Note: load_knowledge doesn't save-back automatically for the
        # family_type stamp and near_misses_cross migrations. We need to
        # save manually to test idempotency. But load_knowledge only
        # auto-saves when "families" key was missing. Our test file has
        # "families", so it won't auto-save. We use a helper.
        from autonomous_loop import save_knowledge
        data1 = load_knowledge()

        # Assert family_type stamped
        assert data1["families"]["sma_crossover|AAPL"]["family_type"] == "single"
        assert data1["families"]["mean_reversion|MSFT"]["family_type"] == "single"
        assert data1["families"]["manifest:xs_momentum|universe:abc123"]["family_type"] == "cross"

        # Assert near_misses_cross list exists
        assert "near_misses_cross" in data1
        assert data1["near_misses_cross"] == []

        # Save and load again to verify idempotency
        save_knowledge(data1)
        data2 = load_knowledge()

        # Assert families unchanged
        for key in data1["families"]:
            assert data1["families"][key] == data2["families"][key]

        # Assert near_misses_cross still empty
        assert data2["near_misses_cross"] == []

    def test_migration_idempotent_second_load_no_change(self, tmp_knowledge_file, monkeypatch):
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_knowledge_file)

        legacy = {
            "families": {
                "sma_crossover|AAPL": {
                    "n_trials": 1,
                    "best_val_sharpe": 0.3,
                    "best_params": {},
                    "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                                      "duplicate_cluster": 0, "exhausted_cluster": 0},
                    "last_tried": "2026-01-01",
                },
            },
            "adopted": [],
            "rejected": [],
            "superseded": [],
            "tested_combinations": [],
            "near_misses": [],
        }
        tmp_knowledge_file.write_text(json.dumps(legacy))

        from autonomous_loop import save_knowledge
        data1 = load_knowledge()
        save_knowledge(data1)
        data2 = load_knowledge()

        # Families should be identical
        assert data1["families"] == data2["families"]
        # near_misses_cross should be present and empty
        assert data2["near_misses_cross"] == []

    def test_migration_with_existing_family_type_no_overwrite(self, tmp_knowledge_file, monkeypatch):
        """Existing family_type must not be overwritten."""
        monkeypatch.setattr("autonomous_loop.KNOWLEDGE_FILE", tmp_knowledge_file)

        legacy = {
            "families": {
                "sma_crossover|AAPL": {
                    "n_trials": 2,
                    "best_val_sharpe": 1.0,
                    "best_params": {},
                    "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                                      "duplicate_cluster": 0, "exhausted_cluster": 0},
                    "last_tried": "2026-01-01",
                    "family_type": "cross",  # pre-existing, even though key suggests single
                },
            },
            "adopted": [],
            "rejected": [],
            "superseded": [],
            "tested_combinations": [],
            "near_misses": [],
            "near_misses_cross": [],
        }
        tmp_knowledge_file.write_text(json.dumps(legacy))

        data = load_knowledge()
        assert data["families"]["sma_crossover|AAPL"]["family_type"] == "cross"
