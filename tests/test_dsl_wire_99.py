"""Regression tests for issue #99 — wiring the DSL (logic_spec) path into the
live pipeline.

The ideation prompt instructs the LLM to prefer logic_spec for structured
manifests, but:
  1. `_syntax_preflight_manifest` dropped every manifest with empty code_b64
     (all logic_spec manifests), so none ever reached the backlog.
  2. The remote runner validates code_b64 only (no logic_spec support), so
     even a bypassed manifest would be HTTP 400'd.

Fix: `strategy_dsl.compile_logic_spec_to_code` + `prepare_spec_for_runner`
compile a logic_spec deterministically into structured-mode Python (same
semantics as the trusted interpreter) and the runner-facing POST sites send
the compiled code_b64 with logic_spec stripped.

These tests fail on pre-fix main:
  - syntax preflight returns (False, "empty code_b64") for logic_spec
    manifests;
  - `_consume_manifest_entry` / `_run_manifest_oos` POST the raw spec
    (empty code_b64, logic_spec present) which the runner rejects.
"""

import base64
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_dsl import run_dsl_strategy, DslExecutionError


# ---------------------------------------------------------------------------
# Fixtures — a valid single_asset_rule spec and a valid cross_sectional_rank
# spec, shaped exactly like ideation's select-stage output.
# ---------------------------------------------------------------------------

def _single_asset_spec():
    return {
        "name": "dsl_test_sma_cross",
        "execution_mode": "structured",
        "code_b64": "",
        "logic_spec": {
            "kind": "single_asset_rule",
            "indicators": {
                "fast": {"type": "sma", "window": 5},
                "slow": {"type": "sma", "window": 20},
            },
            "entry_rule": {"op": "crosses_above",
                           "left": {"indicator": "fast"},
                           "right": {"indicator": "slow"}},
            "exit_rule": {"op": "crosses_below",
                          "left": {"indicator": "fast"},
                          "right": {"indicator": "slow"}},
            "position_sizing": "long_only_binary",
        },
        "data_sources": [{"type": "ohlcv", "universe": ["AAA", "BBB"], "start": "2020-01-01"}],
    }


def _cross_spec():
    return {
        "name": "dsl_test_xs_rank",
        "execution_mode": "structured",
        "code_b64": "",
        "logic_spec": {
            "kind": "cross_sectional_rank",
            "indicators": {"mom": {"type": "rolling_zscore", "window": 21}},
            "rank_by": "mom",
            "n_select": 1,
            "direction": "top",
            "weighting": "equal",
        },
        "data_sources": [{"type": "ohlcv", "universe": ["AAA", "BBB", "CCC"], "start": "2020-01-01"}],
    }


def _make_ohlcv(n_days=250, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    out = {}
    for i, sym in enumerate(("AAA", "BBB", "CCC")):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004 * (i - 1), 0.012, n_days)))
        df = pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.001, n_days)),
            "High": close * (1 + np.abs(rng.normal(0, 0.004, n_days))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.004, n_days))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
        }, index=idx)
        out[sym] = df
    return out


# ---------------------------------------------------------------------------
# 1. Syntax preflight must accept logic_spec manifests (the core #99 drop)
# ---------------------------------------------------------------------------

def test_syntax_preflight_accepts_logic_spec_manifest():
    """Pre-fix main returns (False, 'empty code_b64') here — the exact line
    that dropped all 3 proposals in the 08-11 04:13 cycle."""
    from strategy_ideation import _syntax_preflight_manifest
    valid, err, source = _syntax_preflight_manifest(_single_asset_spec())
    assert valid, f"logic_spec manifest must pass syntax preflight, got: {err}"
    assert source, "compiled source must be returned"
    assert "def generate_signals" in source


def test_syntax_preflight_accepts_cross_sectional_manifest():
    from strategy_ideation import _syntax_preflight_manifest
    valid, err, source = _syntax_preflight_manifest(_cross_spec())
    assert valid, f"cross_sectional manifest must pass syntax preflight, got: {err}"
    assert "def generate_cross_signal" in source


def test_syntax_preflight_rejects_malformed_logic_spec():
    from strategy_ideation import _syntax_preflight_manifest
    spec = _single_asset_spec()
    spec["logic_spec"]["entry_rule"] = {"op": "gt", "left": {"indicator": "nope"},
                                        "right": {"const": 1}}
    valid, err, _ = _syntax_preflight_manifest(spec)
    assert not valid
    assert "compile" in err or "indicator" in err.lower()


def test_syntax_preflight_with_fix_no_llm_rewrite_for_dsl(monkeypatch):
    """A deterministic compile failure must NOT trigger an LLM rewrite of the
    compiled output (that would corrupt the logic_spec/code_b64 contract)."""
    import strategy_ideation as si

    def _boom(*a, **k):
        raise AssertionError("LLM fix must not be attempted for logic_spec manifests")

    monkeypatch.setattr(si, "_syntax_fix_attempt", _boom)
    spec = _single_asset_spec()
    spec["logic_spec"]["indicators"] = {}  # invalid → compile fails
    valid, err, n = si._syntax_preflight_with_fix(spec)
    assert not valid
    assert n == 0


# ---------------------------------------------------------------------------
# 2. prepare_spec_for_runner — the wire transform
# ---------------------------------------------------------------------------

def test_prepare_sets_code_b64_and_strips_logic_spec():
    from strategy_dsl import prepare_spec_for_runner
    spec = _single_asset_spec()
    prepared, err = prepare_spec_for_runner(spec)
    assert prepared is not None, f"prepare failed: {err}"
    assert err is None
    assert "logic_spec" not in prepared
    assert prepared["code_b64"], "compiled code_b64 must be non-empty"
    source = base64.b64decode(prepared["code_b64"]).decode("utf-8")
    compile(source, "<prepared>", "exec")  # must be valid Python
    # original spec untouched (deep copy contract)
    assert "logic_spec" in spec and spec["code_b64"] == ""


def test_prepare_passthrough_for_code_b64_manifest():
    from strategy_dsl import prepare_spec_for_runner
    spec = {"name": "plain", "execution_mode": "structured",
            "code_b64": base64.b64encode(b"def generate_signals(data):\n    return {}\n").decode()}
    prepared, err = prepare_spec_for_runner(spec)
    assert prepared is not None and err is None
    assert prepared["code_b64"] == spec["code_b64"]


def test_prepare_rejects_xor_violation():
    from strategy_dsl import prepare_spec_for_runner
    spec = _single_asset_spec()
    spec["code_b64"] = base64.b64encode(b"x = 1\n").decode()
    prepared, err = prepare_spec_for_runner(spec)
    assert prepared is None
    assert "mutually exclusive" in err


def test_prepare_returns_error_for_malformed_dsl():
    from strategy_dsl import prepare_spec_for_runner
    spec = _cross_spec()
    spec["logic_spec"]["rank_by"] = "missing_indicator"
    prepared, err = prepare_spec_for_runner(spec)
    assert prepared is None
    assert err and "compile" in err


# ---------------------------------------------------------------------------
# 3. Compiled output is semantically identical to the interpreter
# ---------------------------------------------------------------------------

def _exec_compiled(source, data):
    import strategy_dsl as dsl
    ns = {"pd": pd, "np": np, "data": data}
    exec(compile(source, "<compiled>", "exec"), ns)
    return ns


def test_compiled_single_asset_matches_interpreter():
    from strategy_dsl import compile_logic_spec_to_code
    data = _make_ohlcv()
    spec = _single_asset_spec()["logic_spec"]
    expected = run_dsl_strategy(spec, data)
    ns = _exec_compiled(compile_logic_spec_to_code(spec), data)
    signals = ns["generate_signals"](data)
    got = pd.DataFrame(signals)
    assert_frame_equal(got, expected)


def test_compiled_cross_sectional_matches_interpreter():
    from strategy_dsl import compile_logic_spec_to_code
    data = _make_ohlcv()
    spec = _cross_spec()["logic_spec"]
    expected = run_dsl_strategy(spec, data)
    ns = _exec_compiled(compile_logic_spec_to_code(spec), data)
    scores = ns["generate_cross_signal"](data)
    assert_frame_equal(pd.DataFrame(scores), expected)


# ---------------------------------------------------------------------------
# 4. Engine-side synthetic preflight accepts both compiled entrypoints
# ---------------------------------------------------------------------------

def test_engine_synthetic_preflight_accepts_compiled_dsl():
    from manifest import StrategyManifest
    from manifest_runner import _validate_manifest_synthetic
    from strategy_dsl import prepare_spec_for_runner

    for spec, marker in ((_single_asset_spec(), "generate_signals"),
                         (_cross_spec(), "generate_cross_signal")):
        prepared, err = prepare_spec_for_runner(spec)
        assert prepared is not None, err
        m = StrategyManifest.from_dict(prepared)
        assert m.validate() == [], f"prepared manifest must satisfy XOR contract: {m.validate()}"
        result = json.loads(_validate_manifest_synthetic(m))
        assert result.get("valid"), f"synthetic preflight rejected compiled {marker}: {result}"


# ---------------------------------------------------------------------------
# 5. Live POST paths send compiled code, never raw logic_spec
# ---------------------------------------------------------------------------

def test_consume_manifest_entry_posts_compiled_code(monkeypatch):
    """autonomous_loop._consume_manifest_entry must transform logic_spec →
    code_b64 before POSTing; pre-fix main POSTs the raw spec (empty code_b64)
    which the runner HTTP 400s."""
    import autonomous_loop as al

    captured = {}

    def fake_post(url, body, timeout):
        captured["url"] = url
        captured["body"] = json.loads(body.decode("utf-8"))
        return {"status": "ok", "metrics": {"sharpe": 1.0}}, None

    monkeypatch.setattr(al, "_http_post_for_result", fake_post)

    class FakeBl:
        def mark(self, *a, **k):
            pass

    entry = {"id": "test_entry_123"}
    hyp, result = al._consume_manifest_entry(entry, _single_asset_spec(),
                                             "http://runner:9000", FakeBl())
    assert hyp is not None
    body = captured["body"]
    assert body.get("code_b64"), "POSTed manifest must carry compiled code_b64"
    assert "logic_spec" not in body, "logic_spec must be stripped from the runner payload"


def test_consume_manifest_entry_marks_error_on_broken_dsl(monkeypatch):
    import autonomous_loop as al

    def fake_post(url, body, timeout):
        raise AssertionError("must not POST a manifest whose DSL cannot compile")

    monkeypatch.setattr(al, "_http_post_for_result", fake_post)

    marked = {}

    class FakeBl:
        def mark(self, entry_id, status, result=None):
            marked["id"] = entry_id
            marked["status"] = status
            marked["result"] = result

    spec = _single_asset_spec()
    spec["logic_spec"]["indicators"] = {}  # invalid
    entry = {"id": "broken_entry"}
    out = al._consume_manifest_entry(entry, spec, "http://runner:9000", FakeBl())
    assert out is None
    assert marked.get("id") == "broken_entry"


def test_oos_manifest_posts_compiled_code(monkeypatch):
    """oos_monitor._run_manifest_oos must apply the same transform."""
    import oos_monitor as om
    from datetime import date

    captured = {}

    def fake_post(url, payload, timeout=None):
        captured["url"] = url
        captured["payload"] = payload
        return {"status": "ok", "metrics": {"sharpe": 1.0}, "n_days": 30,
                "oos_metrics": {"sharpe": 0.5}}

    monkeypatch.setattr(om, "_post_json", fake_post)
    monkeypatch.setattr(om, "_get_loop_config", lambda: {"runner_url": "http://runner:9000"})

    entry = {
        "hypothesis": {"strategy": "manifest:dsl_test_sma_cross",
                       "symbol": "universe:abc"},
        "manifest_spec": _single_asset_spec(),
    }
    om._run_manifest_oos(entry, date(2024, 1, 2), date(2024, 6, 1))
    payload = captured.get("payload")
    assert payload is not None
    assert payload.get("code_b64"), "OOS POST must carry compiled code_b64"
    assert "logic_spec" not in payload
