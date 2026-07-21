"""
Tests for walk-forward CV integration in backtest_engine.py (PR #3b).

All tests use synthetic data — no network dependency.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.backtest_engine import run_backtest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_csv_1260(tmp_path_factory) -> Path:
    """Create a 1260-row synthetic OHLCV CSV, deterministic seed=42."""
    n_days = 1260
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    daily_ret = rng.normal(0.0005, 0.015, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_ret)
    df = pd.DataFrame(
        {
            "Open": close * (1 - 0.0005),
            "High": close * (1 + 0.005),
            "Low": close * (1 - 0.005),
            "Close": close,
            "Volume": np.full(n_days, 1_000_000),
        },
        index=dates,
    )
    df.index.name = "Date"
    csv_dir = tmp_path_factory.mktemp("cache")
    csv_path = csv_dir / "AAPL.csv"
    df.to_csv(csv_path)
    return csv_path


@pytest.fixture(scope="module")
def csv_dir(synthetic_csv_1260):
    return str(synthetic_csv_1260.parent)


@pytest.fixture
def base_params():
    return {"fast_window": 10, "slow_window": 30}


# ---------------------------------------------------------------------------
# Test 1: no --cv-folds -> v1 shape only (no cv key)
# ---------------------------------------------------------------------------

V1_KEYS = {"in_sample", "out_of_sample", "holdout", "walkforward"}


def test_no_cv_folds_arg_produces_v1_shape_only(csv_dir, base_params):
    result = run_backtest(
        strategy_name="sma_crossover",
        symbol="AAPL",
        params=base_params,
        walkforward=True,
        data_dir=csv_dir,
        cv_folds=None,
    )
    assert "cv" not in result
    for k in V1_KEYS:
        assert k in result, f"Missing v1 key: {k}"
    assert result["walkforward"]["enabled"] is True


# ---------------------------------------------------------------------------
# Test 2: --cv-folds 3 --embargo-days 21 -> cv block alongside v1
# ---------------------------------------------------------------------------

def test_cv_folds_arg_produces_cv_block_alongside_v1(csv_dir, base_params):
    result = run_backtest(
        strategy_name="sma_crossover",
        symbol="AAPL",
        params=base_params,
        walkforward=True,
        data_dir=csv_dir,
        cv_folds=3,
        embargo_days=21,
    )
    # V1 keys still present
    for k in V1_KEYS:
        assert k in result, f"Missing v1 key: {k}"
    # CV block present
    assert "cv" in result
    cv = result["cv"]
    assert "config" in cv
    assert cv["config"]["n_folds"] == 3
    assert cv["config"]["embargo_days"] == 21
    assert len(cv["folds"]) == 3
    for fdat in cv["folds"]:
        assert "fold" in fdat
        assert "n_train" in fdat
        assert "n_val" in fdat
        assert "train_metrics" in fdat
        assert "val_metrics" in fdat
        assert "val_daily_returns" in fdat
        assert "val_dates" in fdat
    # Holdout block
    h = cv["holdout"]
    assert "n_days" in h
    assert "metrics" in h
    assert "daily_returns" in h
    assert "dates" in h


# ---------------------------------------------------------------------------
# Test 3: v1 bytes identical with CV on vs off
# ---------------------------------------------------------------------------

def test_v1_bytes_identical_cv_off_vs_on(csv_dir, base_params):
    """V1 sub-dicts must be byte-identical whether cv_folds is set or not."""
    result_off = run_backtest(
        strategy_name="sma_crossover",
        symbol="AAPL",
        params=base_params,
        walkforward=True,
        data_dir=csv_dir,
        cv_folds=None,
    )
    result_on = run_backtest(
        strategy_name="sma_crossover",
        symbol="AAPL",
        params=base_params,
        walkforward=True,
        data_dir=csv_dir,
        cv_folds=3,
        embargo_days=21,
    )

    # Strip non-v1 keys for comparison
    v1_keys = {"in_sample", "out_of_sample", "holdout", "walkforward"}
    v1_off = {k: result_off[k] for k in v1_keys}
    v1_on = {k: result_on[k] for k in v1_keys}

    off_json = json.dumps(v1_off, sort_keys=True)
    on_json = json.dumps(v1_on, sort_keys=True)
    assert off_json == on_json, (
        f"V1 output differs! Off: {len(off_json)} bytes, On: {len(on_json)} bytes"
    )


# ---------------------------------------------------------------------------
# Test 4: val_daily_returns length matches n_val
# ---------------------------------------------------------------------------

def test_cv_val_returns_length_matches_split(csv_dir, base_params):
    result = run_backtest(
        strategy_name="sma_crossover",
        symbol="AAPL",
        params=base_params,
        walkforward=True,
        data_dir=csv_dir,
        cv_folds=3,
        embargo_days=21,
    )
    for fdat in result["cv"]["folds"]:
        assert len(fdat["val_daily_returns"]) == fdat["n_val"], (
            f"Fold {fdat['fold']}: val_daily_returns len {len(fdat['val_daily_returns'])} != n_val {fdat['n_val']}"
        )
        assert len(fdat["val_dates"]) == fdat["n_val"], (
            f"Fold {fdat['fold']}: val_dates len {len(fdat['val_dates'])} != n_val {fdat['n_val']}"
        )


# ---------------------------------------------------------------------------
# Test 5: val_dates are ISO format YYYY-MM-DD
# ---------------------------------------------------------------------------

def test_cv_val_dates_iso_format(csv_dir, base_params):
    import re
    iso_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    result = run_backtest(
        strategy_name="sma_crossover",
        symbol="AAPL",
        params=base_params,
        walkforward=True,
        data_dir=csv_dir,
        cv_folds=3,
        embargo_days=21,
    )
    for fdat in result["cv"]["folds"]:
        for d in fdat["val_dates"]:
            assert iso_pat.match(d), f"Non-ISO date: {d} in fold {fdat['fold']}"


# ---------------------------------------------------------------------------
# Test 6: holdout computed once, length matches ~20% of data
# ---------------------------------------------------------------------------

def test_cv_holdout_identical_conceptually(csv_dir, base_params):
    result = run_backtest(
        strategy_name="sma_crossover",
        symbol="AAPL",
        params=base_params,
        walkforward=True,
        data_dir=csv_dir,
        cv_folds=3,
        embargo_days=21,
    )
    h = result["cv"]["holdout"]
    assert len(h["daily_returns"]) == h["n_days"]
    assert len(h["dates"]) == h["n_days"]
    # For 1260-row data with 60/20/20, holdout = last 20% = ~252 rows
    # (strategy signal lag drops 1 day, so net returns = 251)
    assert h["n_days"] >= 249, f"Expected ~251 holdout days, got {h['n_days']}"
    assert h["n_days"] <= 252, f"Too many holdout days: {h['n_days']}"


# ---------------------------------------------------------------------------
# Test 7: --cv-folds out of range rejected by CLI
# ---------------------------------------------------------------------------

ENGINE_SCRIPT = str(Path(__file__).resolve().parent.parent / "backtests" / "backtest_engine.py")


@pytest.mark.parametrize("bad_val", [1, 6])
def test_cv_folds_out_of_range_rejected_by_cli(bad_val, csv_dir):
    proc = subprocess.run(
        [sys.executable, ENGINE_SCRIPT,
         "--strategy", "sma_crossover",
         "--symbol", "AAPL",
         "--params", '{"fast_window":10,"slow_window":30}',
         "--data-dir", csv_dir,
         "--cv-folds", str(bad_val)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0, (
        f"--cv-folds {bad_val} should have been rejected (got rc=0)"
    )


# ---------------------------------------------------------------------------
# Test 8: --embargo-days out of range rejected by CLI
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_val", [-1, 61])
def test_embargo_days_out_of_range_rejected_by_cli(bad_val, csv_dir):
    proc = subprocess.run(
        [sys.executable, ENGINE_SCRIPT,
         "--strategy", "sma_crossover",
         "--symbol", "AAPL",
         "--params", '{"fast_window":10,"slow_window":30}',
         "--data-dir", csv_dir,
         "--cv-folds", "3",
         "--embargo-days", str(bad_val)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0, (
        f"--embargo-days {bad_val} should have been rejected (got rc=0)"
    )
