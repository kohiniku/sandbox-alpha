"""
Tests for strategy_harness.py — synthetic data only, no network.
"""
import base64
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.strategy_harness import (
    SafetyVisitor,
    check_safety,
    run_harness,
    _call_signals,
    check_lookahead,
    compute_position_returns,
    MAX_CODE_BYTES,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_ohlc_csv(tmp_path, symbol="TEST", n_days=300):
    """Create a synthetic OHLCV CSV file in tmp_path, return the data dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    daily_ret = rng.normal(0.0005, 0.015, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_ret)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close * 0.9995,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full(n_days, 1_000_000),
        }
    )
    csv_path = data_dir / f"{symbol}.csv"
    df.to_csv(csv_path, index=False)
    return str(data_dir)


def b64(s):
    return base64.b64encode(s.encode()).decode()


# Benign SMA-cross strategy
SMA_CROSS_CODE = """
def generate_signals(df):
    fast = df["Close"].rolling(10).mean()
    slow = df["Close"].rolling(30).mean()
    signals = pd.Series(0, index=df.index)
    signals[fast > slow] = 1
    signals[fast < slow] = -1
    return signals
"""

# Benign strategy with valid numpy usage (causal rolling mean, no lookahead)
BENIGN_NUMPY_CODE = """
def generate_signals(df):
    close = df["Close"].values
    # Causal rolling mean: only use past values, no lookahead
    window = 20
    sma = np.full(len(close), np.nan)
    cumsum = np.cumsum(np.insert(close, 0, 0))
    for i in range(window - 1, len(close)):
        sma[i] = (cumsum[i + 1] - cumsum[i - window + 1]) / window
    signals = pd.Series(0, index=df.index)
    signals[close > sma] = 1
    signals[close < sma] = -1
    return signals
"""


# ---------------------------------------------------------------------------
# AST Safety Check
# ---------------------------------------------------------------------------

class TestASTSafety:
    def test_allowed_import_passes(self):
        code = "import numpy as np\nimport pandas as pd\nimport math\n"
        assert check_safety(code) is None

    def test_forbidden_import_rejected(self):
        code = "import os\n"
        err = check_safety(code)
        assert err is not None
        assert "import os" in err

    def test_forbidden_from_import_rejected(self):
        code = "from subprocess import run\n"
        err = check_safety(code)
        assert err is not None
        assert "subprocess" in err

    def test_eval_rejected(self):
        code = "x = eval('1+1')\n"
        err = check_safety(code)
        assert err is not None
        assert "eval" in err

    def test_exec_rejected(self):
        code = "exec('x=1')\n"
        err = check_safety(code)
        assert err is not None
        assert "exec" in err

    def test_open_rejected(self):
        code = "open('/etc/passwd')\n"
        err = check_safety(code)
        assert err is not None
        assert "open" in err

    def test_dunder_attribute_rejected(self):
        code = "x = obj.__class__\n"
        err = check_safety(code)
        assert err is not None
        assert "dunder" in err

    def test_compile_rejected(self):
        code = "compile('x', '', 'exec')\n"
        err = check_safety(code)
        assert err is not None
        assert "compile" in err


# ---------------------------------------------------------------------------
# Harness integration tests
# ---------------------------------------------------------------------------

class TestStrategyHarness:
    def test_benign_sma_cross_valid_output(self, tmp_path):
        """Benign SMA-cross code → valid JSON metrics, code_hash present."""
        data_dir = make_ohlc_csv(tmp_path)
        result = run_harness(b64(SMA_CROSS_CODE), "TEST", data_dir)

        assert "error" not in result
        assert result["strategy"] == "codegen"
        assert "code_hash" in result
        assert result["symbol"] == "TEST"

        expected_hash = hashlib.sha256(SMA_CROSS_CODE.encode()).hexdigest()
        assert result["code_hash"] == expected_hash

        # Check metric keys exist in each split
        for key in ["in_sample", "out_of_sample", "holdout"]:
            seg = result[key]
            for m in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                      "num_trades", "avg_daily_return_pct", "cost_bps", "num_days"]:
                assert m in seg, f"missing {m} in {key}"

    def test_benign_numpy_code_valid(self, tmp_path):
        """Code using numpy works."""
        data_dir = make_ohlc_csv(tmp_path)
        result = run_harness(b64(BENIGN_NUMPY_CODE), "TEST", data_dir)
        assert "error" not in result
        assert result["code_hash"] is not None

    def test_forbidden_import_rejected(self, tmp_path):
        """Code with import os → error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = "import os\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)\n"
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result
        assert "os" in result["error"]

    def test_eval_code_rejected(self, tmp_path):
        """Code using eval → error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = "def generate_signals(df):\n    eval('1+1')\n    return pd.Series(0, index=df.index)\n"
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result
        assert "eval" in result["error"]

    def test_open_code_rejected(self, tmp_path):
        """Code using open → error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = "def generate_signals(df):\n    open('/etc/passwd')\n    return pd.Series(0, index=df.index)\n"
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result
        assert "open" in result["error"]

    def test_missing_generate_signals(self, tmp_path):
        """Code without generate_signals → error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = "x = 1\n"
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result

    def test_lookahead_detected(self, tmp_path):
        """Code using df['Close'].shift(-5) → lookahead error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = """
def generate_signals(df):
    future = df["Close"].shift(-5)
    signals = pd.Series(0, index=df.index)
    signals[future > df["Close"]] = 1
    return signals
"""
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result
        assert "lookahead" in result["error"].lower()

    def test_invalid_signal_values(self, tmp_path):
        """Signals with value 2.0 → error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = """
def generate_signals(df):
    signals = pd.Series(2, index=df.index)
    return signals
"""
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result
        assert "invalid" in result["error"].lower() or "2" in result["error"]

    def test_code_too_large(self, tmp_path):
        """Code > 64KB → rejected."""
        data_dir = make_ohlc_csv(tmp_path)
        big = "x = " + "1 + " * (MAX_CODE_BYTES + 1000) + "0"
        encoded = base64.b64encode(big.encode()).decode()
        result = run_harness(encoded, "TEST", data_dir)
        assert "error" in result
        assert "too large" in result["error"].lower()

    def test_stdout_suppression(self, tmp_path, capsys):
        """User code print() must not reach harness stdout."""
        data_dir = make_ohlc_csv(tmp_path)
        code = """
def generate_signals(df):
    print("SECRET_MARKER_12345")
    fast = df["Close"].rolling(10).mean()
    slow = df["Close"].rolling(30).mean()
    signals = pd.Series(0, index=df.index)
    signals[fast > slow] = 1
    signals[fast < slow] = -1
    return signals
"""
        result = run_harness(b64(code), "TEST", data_dir)
        # Output should be valid
        assert "error" not in result

        # stdout should NOT contain the printed marker
        captured = capsys.readouterr()
        assert "SECRET_MARKER_12345" not in captured.out
        assert "SECRET_MARKER_12345" not in captured.err

    def test_fabricated_metrics_suppressed(self, tmp_path):
        """Code that prints its own JSON → harness stdout is still harness JSON only."""
        data_dir = make_ohlc_csv(tmp_path)
        # User code prints a fake metrics string — it must not appear in harness stdout
        code = """
def generate_signals(df):
    print('{"strategy": "fake", "in_sample": {"total_return_pct": 99999}}')
    fast = df["Close"].rolling(10).mean()
    slow = df["Close"].rolling(30).mean()
    signals = pd.Series(0, index=df.index)
    signals[fast > slow] = 1
    signals[fast < slow] = -1
    return signals
"""
        import subprocess
        data_b64 = b64(code)
        proc = subprocess.run(
            [
                sys.executable, "-m", "backtests.strategy_harness",
                "--code-b64", data_b64,
                "--symbol", "TEST",
                "--data-dir", data_dir,
            ],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        )
        stdout = proc.stdout.strip()
        # Parse stdout as JSON
        parsed = json.loads(stdout)
        assert "strategy" in parsed
        assert parsed["strategy"] == "codegen"
        # In-sample total_return_pct must NOT be 99999 (fabricated)
        assert parsed["in_sample"]["total_return_pct"] != 99999
        assert "code_hash" in parsed

    def test_lookahead_shift_minus_one(self, tmp_path):
        """Simple lookahead using df['Close'].shift(-1) on the full index position."""
        data_dir = make_ohlc_csv(tmp_path)
        code = """
def generate_signals(df):
    future_close = df["Close"].shift(-1)
    signals = pd.Series(0, index=df.index)
    signals[future_close > df["Close"]] = 1
    signals[future_close < df["Close"]] = -1
    return signals
"""
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result
        assert "lookahead" in result["error"].lower()

    def test_empty_data_symbol(self, tmp_path):
        """Non-existent symbol → error."""
        data_dir = make_ohlc_csv(tmp_path, symbol="AAPL")
        result = run_harness(b64(SMA_CROSS_CODE), "MISSING", data_dir)
        assert "error" in result

    def test_exec_without_generate_signals_callable(self, tmp_path):
        """generate_signals defined but not callable → error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = "generate_signals = 42\n"
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result

    def test_returns_non_series(self, tmp_path):
        """generate_signals returns non-Series → error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = """
def generate_signals(df):
    return [0, 0, 0, 1, 1]
"""
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result

    def test_wrong_index(self, tmp_path):
        """generate_signals returns Series with wrong index → error."""
        data_dir = make_ohlc_csv(tmp_path)
        code = """
def generate_signals(df):
    return pd.Series(0, index=range(len(df)))
"""
        result = run_harness(b64(code), "TEST", data_dir)
        assert "error" in result

    def test_output_json_shape_matches_engine(self, tmp_path):
        """Harness output should have same top-level shape as engine."""
        data_dir = make_ohlc_csv(tmp_path)
        result = run_harness(b64(SMA_CROSS_CODE), "TEST", data_dir)
        assert "error" not in result

        expected_keys = {"strategy", "symbol", "data_points", "date_range",
                         "in_sample", "out_of_sample", "holdout", "walkforward"}
        assert expected_keys.issubset(set(result.keys()))

        # Plus codegen-specific fields
        assert "code_hash" in result
        assert result["strategy"] == "codegen"
