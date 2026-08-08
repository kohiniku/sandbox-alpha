"""
Regression tests for issue #83 — "Validation loop silently collapsed to ~0
tests/day for 7+ days: LLM endpoint env mismatch + 97.5% of single-name grid
killed, no alert".

Three legs, all structural:

Leg 1 — LLM endpoint env coherence: a deepseek model name sent to an
Alibaba/MaaS endpoint (or authenticated with an Alibaba key env var) fails
every call (403/429) and the script silently falls back to random. New
`llm_hypothesis.validate_llm_config()` must reject the incoherent combo, and
`run_loop` must fail loudly at startup when the combo is bad.

Leg 2 — killed-family wall: `generate_hypothesis` had no killed-family filter,
so with ~97.5% of the single-name grid killed the random fallback was a
40-slot lottery of KILLED_SKIP no-ops. It must (a) never pick a killed family,
and (b) re-seed the search space with extra symbols once grid coverage drops
below a threshold, so the space cannot be consumed to zero.

Leg 3 — silent no-op: KILLED_SKIP is stdout-only, runs that consist entirely of
skips deliver [SILENT] with last_status=ok. The loop must aggregate per-run
killed-skip/test counts into knowledge.json and print a machine-greppable
ZERO_TESTS_ALERT when a run produced no tests.
"""
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomous_loop import (
    STRATEGY_TEMPLATES,
    SYMBOL_POOL,
    generate_hypothesis,
    get_killed_families,
    run_loop,
)
from loop_constants import FamilyLifecycle


# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def _grid_keys():
    return [
        f"{s}|{sym}"
        for s in STRATEGY_TEMPLATES
        for sym in SYMBOL_POOL
    ]


def _minimal_knowledge(families=None):
    return {
        "tested": [],
        "tested_combinations": [],
        "adopted": [],
        "rejected": [],
        "superseded": [],
        "families": families or {},
        "iterations": 0,
        "errors": [],
    }


def _killed_family():
    return {
        "lifecycle": FamilyLifecycle.KILLED,
        "kill_reason": "test_kill",
        "n_trials": 5,
        "best_val_sharpe": -0.5,
        "family_type": "single",
        "refine_count": 0,
    }


def _validate_llm_config():
    """Lazy import: the function does not exist on main (fails loudly there)."""
    from llm_hypothesis import validate_llm_config
    return validate_llm_config()


def _extra_symbol_pool():
    """Lazy import: the constant does not exist on main."""
    from autonomous_loop import EXTRA_SYMBOL_POOL
    return EXTRA_SYMBOL_POOL


def _killed_knowledge():
    """All 40 grid families killed — the production state from issue #83."""
    return _minimal_knowledge({k: _killed_family() for k in _grid_keys()})


def _synthetic_backtest_result():
    """Backtest result that passes all gates (adopted)."""
    return {
        "strategy": "sma_crossover",
        "symbol": "MSFT",
        "params": {"fast_window": 10, "slow_window": 30},
        "in_sample": {
            "total_return_pct": 17.0, "sharpe_ratio": 1.3,
            "max_drawdown_pct": -8.0, "num_trades": 5,
            "avg_daily_return_pct": 0.05, "cost_bps": 5.0, "num_days": 151,
        },
        "out_of_sample": {
            "total_return_pct": 15.0, "sharpe_ratio": 1.2,
            "max_drawdown_pct": -10.0, "num_trades": 5,
            "avg_daily_return_pct": 0.04, "cost_bps": 5.0, "num_days": 252,
        },
        "holdout": {
            "total_return_pct": 10.0, "sharpe_ratio": 0.8,
            "max_drawdown_pct": -8.0, "num_trades": 3,
            "avg_daily_return_pct": 0.02, "cost_bps": 5.0, "num_days": 80,
        },
        "walkforward": {"enabled": True, "train_ratio": 0.6, "val_ratio": 0.2,
                        "holdout_ratio": 0.2},
    }


# ──────────────────────────────────────────────
# Leg 1 — LLM endpoint env coherence
# ──────────────────────────────────────────────

class TestValidateLlmConfig:
    def test_rejects_deepseek_model_on_aliyun_endpoint(self, monkeypatch):
        """The exact production drift: deepseek model + Alibaba/MaaS base URL."""
        monkeypatch.setenv(
            "HYPO_LLM_BASE_URL",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )
        monkeypatch.setenv("HYPO_LLM_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY")
        with pytest.raises(ValueError, match="HYPO_LLM_BASE_URL"):
            _validate_llm_config()

    def test_rejects_deepseek_model_with_alibaba_key_var(self, monkeypatch):
        """Deepseek model but the key env var names an Alibaba credential."""
        monkeypatch.setenv("HYPO_LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("HYPO_LLM_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "ALIBABA_TOKEN_PLAN_API_KEY")
        with pytest.raises(ValueError, match="HYPO_LLM_API_KEY_ENV"):
            _validate_llm_config()

    def test_accepts_coherent_deepseek_config(self, monkeypatch):
        monkeypatch.setenv("HYPO_LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("HYPO_LLM_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY")
        cfg = _validate_llm_config()
        assert cfg["model"] == "deepseek-v4-flash"

    def test_accepts_aliyun_endpoint_with_qwen_model(self, monkeypatch):
        """No false positive: an Alibaba endpoint serving a qwen model is coherent."""
        monkeypatch.setenv(
            "HYPO_LLM_BASE_URL",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )
        monkeypatch.setenv("HYPO_LLM_MODEL", "qwen3.6-flash")
        monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "ALIBABA_TOKEN_PLAN_API_KEY")
        assert _validate_llm_config()["model"] == "qwen3.6-flash"

    def test_run_loop_fails_loudly_on_incoherent_llm_env(self, monkeypatch, capsys):
        """run_loop with use_llm=True must abort at startup with LLM_CONFIG_ERROR
        instead of silently falling back to random for every iteration."""
        import autonomous_loop

        monkeypatch.setenv(
            "HYPO_LLM_BASE_URL",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )
        monkeypatch.setenv("HYPO_LLM_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("HYPO_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY")

        knowledge = _minimal_knowledge()
        monkeypatch.setattr(autonomous_loop, "load_knowledge", lambda: knowledge)
        monkeypatch.setattr(autonomous_loop, "save_knowledge", lambda k: None)
        monkeypatch.setattr(autonomous_loop, "_get_loop_config", lambda: {
            "runner_url": None, "backlog_path": "/dev/null",
            "use_llm": True, "gate_v2": "0",
        })
        # Even if generation somehow got reached, keep it offline.
        monkeypatch.setattr(
            autonomous_loop, "_generate_with_llm_fallback",
            lambda k: ({"strategy": "momentum", "symbol": "MSFT",
                        "params": {"lookback": 20, "hold_period": 5},
                        "description": "x", "id": "h1",
                        "generated_at": "2026-01-01T00:00:00"}, "(random)"),
        )

        with pytest.raises(ValueError, match="HYPO_LLM_BASE_URL"):
            autonomous_loop.run_loop(1)
        captured = capsys.readouterr()
        assert "LLM_CONFIG_ERROR" in captured.out


# ──────────────────────────────────────────────
# Leg 2 — killed-family filter + re-seed
# ──────────────────────────────────────────────

class TestGenerateHypothesisKilledFilter:
    def test_never_picks_killed_family_across_many_seeds(self):
        """Kill half the grid (coverage 0.5 < re-seed threshold). generate_hypothesis
        must NEVER return a killed family, no matter the RNG seed."""
        killed = {k: _killed_family() for k in _grid_keys()[:20]}  # 20 of 40 killed
        knowledge = _minimal_knowledge(killed)
        killed_keys = set(get_killed_families(knowledge))
        for seed in range(300):
            random.seed(seed)
            hyp = generate_hypothesis(knowledge)
            key = f"{hyp['strategy']}|{hyp['symbol']}"
            assert key not in killed_keys, f"seed {seed} picked killed family {key}"
            assert hyp["symbol"] in SYMBOL_POOL  # no re-seed below threshold

    def test_reseeds_when_grid_fully_killed(self):
        """All 40 grid families killed (97.5%+ coverage): the pool must expand to
        EXTRA_SYMBOL_POOL so the search space cannot be consumed to zero."""
        knowledge = _killed_knowledge()
        for seed in range(100):
            random.seed(seed)
            hyp = generate_hypothesis(knowledge)
            assert hyp["symbol"] in _extra_symbol_pool(), (
                f"seed {seed}: expected re-seeded symbol, got {hyp['symbol']}"
            )
            key = f"{hyp['strategy']}|{hyp['symbol']}"
            assert key not in get_killed_families(knowledge)

    def test_adopted_neighborhood_skips_killed_family(self):
        """30% adopted-neighborhood branch must also skip killed families."""
        killed = {k: _killed_family() for k in _grid_keys()[:20]}
        knowledge = _minimal_knowledge(killed)
        knowledge["adopted"] = [{
            "hypothesis": {
                "strategy": "momentum", "symbol": SYMBOL_POOL[0],
                "params": {"lookback": 20, "hold_period": 5},
            },
        }]
        killed_keys = set(get_killed_families(knowledge))
        for seed in range(200):
            random.seed(seed)
            hyp = generate_hypothesis(knowledge)
            key = f"{hyp['strategy']}|{hyp['symbol']}"
            assert key not in killed_keys, f"seed {seed} picked killed family {key}"


# ──────────────────────────────────────────────
# Leg 3 — KILLED_SKIP aggregation + ZERO_TESTS alert
# ──────────────────────────────────────────────

def _run_loop_with_mocks(knowledge, tmp_path, monkeypatch, iterations=3,
                         generate_hyp=None):
    import autonomous_loop

    monkeypatch.setattr(autonomous_loop, "load_knowledge", lambda: knowledge)
    monkeypatch.setattr(autonomous_loop, "save_knowledge", lambda k: None)
    monkeypatch.setattr(autonomous_loop, "_get_loop_config", lambda: {
        "runner_url": None, "backlog_path": str(tmp_path / "backlog.json"),
        "use_llm": False, "gate_v2": "0",
    })
    if generate_hyp is not None:
        monkeypatch.setattr(autonomous_loop, "generate_hypothesis", generate_hyp)
    return autonomous_loop


class TestRunStats:
    def test_zero_tests_alert_when_everything_killed_skipped(self, tmp_path, monkeypatch, capsys):
        """A run where every iteration is a KILLED_SKIP must print RUN_SUMMARY +
        ZERO_TESTS_ALERT and record the stats in knowledge — never a silent no-op."""
        knowledge = _killed_knowledge()
        killed_hyp = {
            "id": "h_killed", "strategy": "momentum", "symbol": SYMBOL_POOL[0],
            "params": {"lookback": 20, "hold_period": 5},
            "description": "x", "generated_at": "2026-01-01T00:00:00",
        }

        def _gen(k):
            return killed_hyp

        autonomous_loop = _run_loop_with_mocks(
            knowledge, tmp_path, monkeypatch, iterations=3, generate_hyp=_gen,
        )
        autonomous_loop.run_loop(3)

        captured = capsys.readouterr()
        assert "KILLED_SKIP momentum|AAPL" in captured.out
        assert "RUN_SUMMARY" in captured.out
        assert "killed_skips=3" in captured.out
        assert "ZERO_TESTS_ALERT" in captured.out
        stats = knowledge.get("last_run_stats", {})
        assert stats.get("zero_tests") is True
        assert stats.get("killed_skips") == 3
        assert stats.get("tested") == 0
        assert len(knowledge.get("run_stats", [])) == 1

    def test_no_alert_when_tests_produced(self, tmp_path, monkeypatch, capsys):
        """A healthy run (a test is produced) must NOT fire ZERO_TESTS_ALERT."""
        knowledge = _minimal_knowledge()  # no killed families
        good_hyp = {
            "id": "h_good", "strategy": "sma_crossover", "symbol": "MSFT",
            "params": {"fast_window": 10, "slow_window": 30},
            "description": "x", "generated_at": "2026-01-01T00:00:00",
        }

        def _gen(k):
            return good_hyp

        autonomous_loop = _run_loop_with_mocks(
            knowledge, tmp_path, monkeypatch, iterations=1, generate_hyp=_gen,
        )
        with patch("autonomous_loop.run_backtest",
                   return_value=_synthetic_backtest_result()):
            autonomous_loop.run_loop(1)

        captured = capsys.readouterr()
        assert "RUN_SUMMARY" in captured.out
        assert "ZERO_TESTS_ALERT" not in captured.out
        stats = knowledge.get("last_run_stats", {})
        assert stats.get("zero_tests") is False
        assert stats.get("tested") == 1

    def test_backlog_killed_skip_counts_toward_run_stats(self, tmp_path, monkeypatch, capsys):
        """A backlog entry in a killed family (marked DONE_REJECTED) must also
        increment the run's killed_skips counter and feed the alert."""
        from backlog import Backlog, make_param_entry

        knowledge = _killed_knowledge()
        bl = Backlog(str(tmp_path / "backlog.json"))
        accepted, eid = bl.add_entry(make_param_entry(
            strategy="momentum", symbol=SYMBOL_POOL[0],
            params={"lookback": 20, "hold_period": 5},
            priority=0.9, source={"kind": "idea", "ref": "test"},
        ))
        assert accepted

        killed_hyp = {
            "id": "h_killed", "strategy": "momentum", "symbol": SYMBOL_POOL[0],
            "params": {"lookback": 20, "hold_period": 5},
            "description": "x", "generated_at": "2026-01-01T00:00:00",
        }

        def _gen(k):
            return killed_hyp

        autonomous_loop = _run_loop_with_mocks(
            knowledge, tmp_path, monkeypatch, iterations=1, generate_hyp=_gen,
        )
        autonomous_loop.run_loop(1)

        captured = capsys.readouterr()
        assert "KILLED_SKIP momentum|AAPL" in captured.out
        assert "ZERO_TESTS_ALERT" in captured.out
        stats = knowledge.get("last_run_stats", {})
        assert stats.get("killed_skips") >= 1
        assert stats.get("zero_tests") is True

        # Disk round-trip: the backlog entry was marked DONE_REJECTED (family_killed)
        bl2 = Backlog(str(tmp_path / "backlog.json"))
        entries = [e for e in bl2.load()["entries"] if e["id"] == eid]
        assert len(entries) == 1
        assert entries[0]["status"] == "done_rejected"
        assert entries[0].get("result", {}).get("reason") == "family_killed"
