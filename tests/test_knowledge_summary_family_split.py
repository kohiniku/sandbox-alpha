"""
Tests for _summarise_near_misses — cross-sectional split (PR 4e).
"""
import json

from strategy_ideation import _summarise_near_misses

_FAKE_NM = {
    "strategy": "mean_reversion",
    "symbol": "SPY",
    "params": {"window": 20},
    "val_sharpe": 0.85,
    "deflated_threshold": 0.10,
    "holdout_sharpe": -0.30,
    "failed_gate": "holdout",
}


def _make_nm(**overrides):
    nm = dict(_FAKE_NM)
    nm.update(overrides)
    return nm


# --- Legacy: only near_misses populated -----------------------------------

def test_legacy_near_misses_only():
    """Legacy knowledge (near_misses populated, near_misses_cross absent)
    should produce output indistinguishable from pre-PR."""
    knowledge = {
        "near_misses": [_make_nm()],
    }
    result = _summarise_near_misses(knowledge)
    assert result is not None
    assert "SINGLE-NAME NEAR-MISSES" in result
    assert "CROSS-SECTIONAL NEAR-MISSES" not in result
    assert "NEAR-MISS ARCHIVE (signal on validation but FAILED)" in result
    # Golden string — the per-entry format must match pre-PR exactly
    assert "mean_reversion/SPY params={\"window\": 20}" in result
    assert "val_sharpe=0.85" in result
    assert "(thresh=0.10)" in result
    assert "holdout_sharpe=-0.30" in result
    assert "-- holdout" in result


def test_legacy_near_misses_multiple():
    """Multiple legacy entries with golden string check."""
    knowledge = {
        "near_misses": [
            _make_nm(strategy="s1", symbol="AAPL", params={"w": 5}, val_sharpe=0.5, deflated_threshold=0.05, holdout_sharpe=None, failed_gate="sharpe"),
            _make_nm(strategy="s2", symbol="MSFT", params={"w": 10}, val_sharpe=0.7, deflated_threshold=0.15, holdout_sharpe=0.1, failed_gate="holdout"),
        ],
    }
    result = _summarise_near_misses(knowledge)
    assert result is not None
    assert "s1/AAPL" in result
    assert "s2/MSFT" in result
    assert "CROSS-SECTIONAL NEAR-MISSES" not in result


# --- Both single and cross populated ---------------------------------------

def test_both_types_populated():
    """Both near_misses and near_misses_cross populated — both sections shown."""
    knowledge = {
        "near_misses": [_make_nm(strategy="single_s", symbol="AAPL")],
        "near_misses_cross": [_make_nm(strategy="cross_s", symbol="XLF")],
    }
    result = _summarise_near_misses(knowledge)
    assert result is not None
    assert "SINGLE-NAME NEAR-MISSES" in result
    assert "CROSS-SECTIONAL NEAR-MISSES" in result
    # Order: single first, then cross
    single_idx = result.index("SINGLE-NAME NEAR-MISSES")
    cross_idx = result.index("CROSS-SECTIONAL NEAR-MISSES")
    assert single_idx < cross_idx
    assert "single_s/AAPL" in result
    assert "cross_s/XLF" in result


# --- Only cross populated --------------------------------------------------

def test_only_cross_populated():
    """Only near_misses_cross populated — only cross section shown."""
    knowledge = {
        "near_misses_cross": [_make_nm(strategy="xs_momentum", symbol="N/A")],
    }
    result = _summarise_near_misses(knowledge)
    assert result is not None
    assert "SINGLE-NAME NEAR-MISSES" not in result
    assert "CROSS-SECTIONAL NEAR-MISSES" in result
    assert "xs_momentum" in result


# --- Empty / None cases ----------------------------------------------------

def test_both_empty_returns_none():
    """Both lists empty — return None."""
    knowledge = {"near_misses": [], "near_misses_cross": []}
    result = _summarise_near_misses(knowledge)
    assert result is None


def test_neither_present_returns_none():
    """Neither key present — return None."""
    knowledge: dict = {}
    result = _summarise_near_misses(knowledge)
    assert result is None


# --- Truncation ------------------------------------------------------------

def test_truncation_single():
    """Only last 20 entries are rendered (single list > 20)."""
    entries = [_make_nm(strategy=f"s{i:02d}") for i in range(30)]
    knowledge = {"near_misses": entries}
    result = _summarise_near_misses(knowledge)
    assert result is not None
    # First 10 entries (s00..s09) should be truncated off
    assert "s00" not in result
    assert "s09" not in result
    # Last 20 entries (s10..s29) should be present
    assert "s10" in result
    assert "s29" in result


def test_truncation_cross():
    """Only last 20 cross entries rendered (cross list > 20)."""
    entries = [_make_nm(strategy=f"x{i:02d}") for i in range(25)]
    knowledge = {"near_misses_cross": entries}
    result = _summarise_near_misses(knowledge)
    assert result is not None
    assert "x00" not in result
    assert "x04" not in result
    assert "x05" in result
    assert "x24" in result
