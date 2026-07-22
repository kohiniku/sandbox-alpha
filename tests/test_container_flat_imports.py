"""Regression guard: the Dockerfile ships the engine in a flat layout.

`COPY backtests/ /backtest/` puts every module as a sibling under /backtest/
without a package parent, and the entrypoint invokes the engine as a script
(`python /backtest/backtest_engine.py`). In that mode, Python treats each
imported sibling as a top-level module — so bare relative imports (`from
.splitter import ...`) raise `ImportError: attempted relative import with no
known parent package`.

This has bitten us once: after PR #39 (splitter re-export in metrics.py) and
PR #41 (cv_folds import inside run_backtest), every /run call 500'd until the
imports were guarded with a try/except fallback (commit ca8e5af). Pytest
missed it because tests run from the repo root, where `backtests/` is a proper
package.

This test replicates the container's flat layout in a tempdir and invokes the
engine as a script, so any future relative-import regression fails here first.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKTESTS_DIR = REPO_ROOT / "backtests"


@pytest.fixture
def flat_layout(tmp_path):
    """Copy backtests/ contents (flat) to tmp_path, mirroring the container."""
    for item in BACKTESTS_DIR.iterdir():
        dst = tmp_path / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy(item, dst)
    return tmp_path


def _isolated_env():
    """Env with PYTHONPATH stripped so the repo's `backtests` package cannot
    accidentally satisfy `from backtests.foo import ...`."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def test_engine_help_succeeds_in_flat_layout(flat_layout):
    """`python engine.py --help` must exit 0 with no import errors.

    Mirrors the container: script-mode invocation auto-populates sys.path[0]
    with the script's directory, so sibling modules resolve as top-level.
    PYTHONPATH is stripped so the repo's `backtests` package cannot
    accidentally satisfy `from backtests.foo import ...` (false pass).
    """
    result = subprocess.run(
        [sys.executable, str(flat_layout / "backtest_engine.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        env=_isolated_env(),
        cwd="/tmp",
    )
    assert result.returncode == 0, (
        "Engine failed to import in flat container layout — likely a bare "
        "relative import in backtests/*.py. Guard with try/except (see "
        "commit ca8e5af).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Sanity: --cv-folds must appear in help (guards against silent CLI drift).
    assert "--cv-folds" in result.stdout, f"--cv-folds missing from help:\n{result.stdout}"
    assert "--embargo-days" in result.stdout, f"--embargo-days missing from help:\n{result.stdout}"


def test_metrics_module_imports_in_flat_layout(flat_layout):
    """`python -c "import metrics"` must succeed from the flat dir.

    -c mode adds cwd to sys.path[0], so with cwd=flat_layout the sibling
    modules become importable — exactly like a container would see them.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import metrics; print('OK')"],
        capture_output=True,
        text=True,
        timeout=10,
        env=_isolated_env(),
        cwd=str(flat_layout),
    )
    assert result.returncode == 0, (
        "metrics.py failed to import in flat layout:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_single_name_subpackage_imports_in_flat_layout(flat_layout):
    """`python -c \"import strategies._single_name.*\"` must succeed from flat dir."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import strategies; "
         "from strategies._single_name import sma_crossover, mean_reversion, momentum, rsi; "
         "assert sma_crossover.NAME == 'sma_crossover'; "
         "print('OK')"],
        capture_output=True,
        text=True,
        timeout=10,
        env=_isolated_env(),
        cwd=str(flat_layout),
    )
    assert result.returncode == 0, (
        "strategies._single_name.* failed to import in flat layout:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_panel_adapter_imports_in_flat_layout(flat_layout):
    """`python -c \"import strategies._panel_adapter\"` must succeed from flat dir."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import strategies._panel_adapter; "
         "assert callable(strategies._panel_adapter.wrap_single_as_cross); "
         "print('OK')"],
        capture_output=True,
        text=True,
        timeout=10,
        env=_isolated_env(),
        cwd=str(flat_layout),
    )
    assert result.returncode == 0, (
        "strategies._panel_adapter failed to import in flat layout:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
