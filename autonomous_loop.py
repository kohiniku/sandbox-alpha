#!/usr/bin/env python3
"""
Autonomous Alpha Discovery Loop
エージェントが仮説生成→バックテスト→評価→蓄積を自律的に回す
"""
import base64
import hashlib
import json
import math
import os
import sys
import time
import subprocess
import random
import uuid
import copy
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # リポジトリ相対パス
RESULTS_DIR = BASE_DIR / "results"
STRATEGIES_DIR = BASE_DIR / "strategies"
KNOWLEDGE_FILE = BASE_DIR / "knowledge.json"

RESULTS_DIR.mkdir(exist_ok=True)
STRATEGIES_DIR.mkdir(exist_ok=True)

# テスト対象の銘柄プール
SYMBOL_POOL = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY", "QQQ", "BTC-USD", "ETH-USD"]

# 戦略テンプレート
STRATEGY_TEMPLATES = {
    "sma_crossover": {
        "description": "移動平均クロスオーバー",
        "param_space": {
            "fast_window": range(5, 30),
            "slow_window": range(20, 100)
        }
    },
    "mean_reversion": {
        "description": "平均回帰",
        "param_space": {
            "window": range(10, 60),
            "threshold": [1.0, 1.5, 2.0, 2.5, 3.0]
        }
    },
    "momentum": {
        "description": "モメンタム",
        "param_space": {
            "lookback": range(5, 60),
            "hold_period": range(1, 20)
        }
    },
    "rsi": {
        "description": "RSI平均回帰",
        "param_space": {
            "rsi_window": range(7, 28),
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80]
        }
    },
}

# --- Overfitting guards ---
MIN_SHARPE_BASE = 0.5  # absolute floor for deflated threshold
MAX_DRAWDOWN_LIMIT = -25.0  # max drawdown gate (validation)


def compute_effective_min_sharpe(N_family, T_val):
    """Deflation formula: threshold rises with more trials to penalize data snooping."""
    N = max(N_family, 2)
    return max(MIN_SHARPE_BASE, math.sqrt(2 * math.log(N)) * math.sqrt(252.0 / max(T_val, 1)))


def _params_within_cluster(p1, p2, templates):
    """Check if p1 and p2 params are within ±15% (numeric) or one-step (list) of each other."""
    all_keys = set(p1.keys()) | set(p2.keys())
    for k in all_keys:
        v1 = p1.get(k)
        v2 = p2.get(k)
        if v1 is None or v2 is None:
            return False

        # First, check if this param is a list-type in any template
        is_list_param = False
        for tmpl_name, tmpl in templates.items():
            space = tmpl.get("param_space", {}).get(k)
            if isinstance(space, list):
                is_list_param = True
                if v1 in space and v2 in space:
                    if abs(space.index(v1) - space.index(v2)) > 1:
                        return False
                    break  # passed list check
                else:
                    # values not in the list space — fall through to exact match
                    if v1 != v2:
                        return False
                    break
        if is_list_param:
            continue  # handled above

        # Numeric param: ±15% tolerance
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if v1 == 0 and v2 == 0:
                continue
            if v1 == 0 or v2 == 0:
                return False
            pct = abs(v1 - v2) / max(abs(v1), abs(v2))
            if pct > 0.15:
                return False
        else:
            if v1 != v2:
                return False
    return True


def load_knowledge():
    """過去の戦略テスト結果を読み込む"""
    if KNOWLEDGE_FILE.exists():
        data = json.loads(KNOWLEDGE_FILE.read_text())
        # 後方互換: 古いナレッジファイルに不足キーを補完
        data.setdefault("tested_combinations", [])
        data.setdefault("superseded", [])
        data.setdefault("errors", [])
        # Migration: rebuild "families" from history if missing
        if "families" not in data:
            data["families"] = _rebuild_families_from_history(data)
            # persist the rebuilt families immediately so the migration is idempotent
            KNOWLEDGE_FILE.write_text(json.dumps(data, indent=2, default=str))
        return data
    return {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
            "superseded": [], "families": {}, "iterations": 0, "errors": []}


def _rebuild_families_from_history(knowledge):
    """One-time idempotent rebuild of families aggregate from full history."""
    families = {}
    all_entries = (
        knowledge.get("adopted", []) +
        knowledge.get("rejected", []) +
        knowledge.get("superseded", [])
    )
    for entry in all_entries:
        hyp = entry.get("hypothesis", {})
        key = _family_key(hyp.get("strategy", ""), hyp.get("symbol", ""))
        if not hyp.get("strategy") or not hyp.get("symbol"):
            continue
        families.setdefault(key, {
            "n_trials": 0,
            "best_val_sharpe": -999.0,
            "best_params": {},
            "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                              "duplicate_cluster": 0, "exhausted_cluster": 0},
            "last_tried": ""
        })
        _apply_entry_to_family(families, key, entry, hyp)
    return families


def _family_key(strategy, symbol):
    return f"{strategy}|{symbol}"


def _apply_entry_to_family(families, key, entry, hyp):
    """Incrementally update a single family aggregate record from one evaluation entry."""
    fam = families.setdefault(key, {
        "n_trials": 0,
        "best_val_sharpe": -999.0,
        "best_params": {},
        "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                          "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": ""
    })
    fam["n_trials"] += 1
    ev = entry.get("evaluation", {})
    val_sharpe = ev.get("sharpe_ratio", -999)
    if val_sharpe > fam["best_val_sharpe"]:
        fam["best_val_sharpe"] = val_sharpe
        fam["best_params"] = hyp.get("params", {})
    tested_at = entry.get("tested_at", "")
    if tested_at > fam["last_tried"]:
        fam["last_tried"] = tested_at
    # Tally gate failures
    gate_results = ev.get("gate_results", {})
    if not gate_results:
        return
    # Determine why it was rejected (or if it passed everything)
    verdict = ev.get("verdict", entry.get("verdict", ""))
    if verdict == "adopted":
        return  # adopted entries don't count as gate failures
    # Count individual gate failures
    if not gate_results.get("validation", True):
        fam["gate_failures"]["validation"] += 1
    elif not gate_results.get("holdout", True):
        fam["gate_failures"]["holdout"] += 1
    elif gate_results.get("cluster") == "duplicate_cluster":
        fam["gate_failures"]["duplicate_cluster"] += 1
    elif gate_results.get("cluster") == "exhausted_cluster":
        fam["gate_failures"]["exhausted_cluster"] += 1
    # deflation gate is informational — count if explicitly failed
    if not gate_results.get("deflation", True):
        fam["gate_failures"]["deflation"] += 1


def update_family_aggregates(knowledge, record):
    """Update families dict after every evaluation (adopted and rejected both count)."""
    hyp = record.get("hypothesis", {})
    key = _family_key(hyp.get("strategy", ""), hyp.get("symbol", ""))
    if not hyp.get("strategy") or not hyp.get("symbol"):
        return
    families = knowledge.setdefault("families", {})
    _apply_entry_to_family(families, key, record, hyp)


def _check_exhausted_cluster(hypothesis, knowledge):
    """Pre-backtest check: if >=3 rejected entries in the same param cluster
    and best val Sharpe among them < 0, skip the backtest entirely.
    Returns (is_exhausted: bool, member_count: int, best_sharpe: float)."""
    strategy = hypothesis["strategy"]
    symbol = hypothesis["symbol"]
    params = hypothesis["params"]

    matching = []
    for entry in knowledge.get("rejected", []):
        eh = entry.get("hypothesis", {})
        if eh.get("strategy") != strategy or eh.get("symbol") != symbol:
            continue
        if _params_within_cluster(params, eh.get("params", {}), STRATEGY_TEMPLATES):
            matching.append(entry)

    if len(matching) < 3:
        return False, len(matching), None

    best_sharpe = max(
        (e.get("evaluation", {}).get("sharpe_ratio", -999) for e in matching),
        default=-999
    )
    if best_sharpe >= 0:
        return False, len(matching), best_sharpe

    return True, len(matching), best_sharpe


def save_knowledge(knowledge):
    """ナレッジベースを更新"""
    KNOWLEDGE_FILE.write_text(json.dumps(knowledge, indent=2, default=str))


def generate_hypothesis(knowledge):
    """
    仮説生成フェーズ
    ナレッジベースを参照して、まだ試していない戦略パラメータを生成。
    既にテスト済みの(strategy, symbol, params)の組み合わせはスキップ。
    30%の確率でadopted戦略のパラメータ近傍を探索する。
    """
    max_attempts = 50  # 重複回避の試行上限

    for _ in range(max_attempts):
        # 30%の確率で adopted 戦略の近傍を探索
        if knowledge.get("adopted") and random.random() < 0.3:
            adopted = random.choice(knowledge["adopted"])
            strategy_name = adopted["hypothesis"]["strategy"]
            # Skip adopted entries whose strategy is not in STRATEGY_TEMPLATES (e.g. codegen)
            if strategy_name not in STRATEGY_TEMPLATES:
                continue
            adopted_params = adopted["hypothesis"]["params"]
            symbol = adopted["hypothesis"]["symbol"]
            template = STRATEGY_TEMPLATES[strategy_name]

            # adoptedパラメータの近傍を生成
            params = {}
            for param_name, param_space in template["param_space"].items():
                base_val = adopted_params.get(param_name)
                if base_val is None:
                    if isinstance(param_space, range):
                        params[param_name] = random.choice(list(param_space))
                    else:
                        params[param_name] = random.choice(param_space)
                elif isinstance(param_space, range):
                    # 近傍 ±30% の範囲でサンプリング
                    offset = int(base_val * random.uniform(-0.3, 0.3))
                    candidate = base_val + offset
                    params[param_name] = max(param_space.start, min(param_space.stop - 1, candidate))
                else:
                    # リストの場合はランダムにずらす
                    idx = param_space.index(base_val) if base_val in param_space else len(param_space) // 2
                    offset = random.randint(-2, 2)
                    new_idx = max(0, min(len(param_space) - 1, idx + offset))
                    params[param_name] = param_space[new_idx]
        else:
            strategy_name = random.choice(list(STRATEGY_TEMPLATES.keys()))
            template = STRATEGY_TEMPLATES[strategy_name]
            symbol = random.choice(SYMBOL_POOL)

            # パラメータをランダムサンプリング
            params = {}
            for param_name, param_space in template["param_space"].items():
                if isinstance(param_space, range):
                    params[param_name] = random.choice(list(param_space))
                else:
                    params[param_name] = random.choice(param_space)

        # 追加制約: SMA crossoverでは fast < slow
        if strategy_name == "sma_crossover":
            params["fast_window"] = min(params["fast_window"], params["slow_window"] - 5)

        # 重複チェック: 既にテスト済みの(strategy, symbol, params)はスキップ
        duplicate = False
        for tested in knowledge.get("tested_combinations", []):
            if (tested["strategy"] == strategy_name and
                tested["symbol"] == symbol and
                tested["params"] == params):
                duplicate = True
                break
        if duplicate:
            continue

        hypothesis = {
            "id": f"hyp_{int(time.time())}_{random.randint(1000,9999)}",
            "strategy": strategy_name,
            "symbol": symbol,
            "params": params,
            "description": f"{template['description']} on {symbol}",
            "generated_at": datetime.now().isoformat()
        }

        print(f"  💡 仮説: {hypothesis['description']}")
        print(f"     パラメータ: {params}")
        return hypothesis

    # 全組み合わせ枯渇時のフォールバック（重複許容）
    strategy_name = random.choice(list(STRATEGY_TEMPLATES.keys()))
    template = STRATEGY_TEMPLATES[strategy_name]
    symbol = random.choice(SYMBOL_POOL)
    params = {}
    for param_name, param_space in template["param_space"].items():
        if isinstance(param_space, range):
            params[param_name] = random.choice(list(param_space))
        else:
            params[param_name] = random.choice(param_space)

    if strategy_name == "sma_crossover":
        params["fast_window"] = min(params["fast_window"], params["slow_window"] - 5)

    hypothesis = {
        "id": f"hyp_{int(time.time())}_{random.randint(1000,9999)}",
        "strategy": strategy_name,
        "symbol": symbol,
        "params": params,
        "description": f"{template['description']} on {symbol}",
        "generated_at": datetime.now().isoformat()
    }
    print(f"  ⚠️ 全組み合わせ枯渇、重複許容で仮説生成")
    print(f"  💡 仮説: {hypothesis['description']}")
    print(f"     パラメータ: {params}")
    return hypothesis


def run_backtest(hypothesis, metrics_since=None):
    """
    バックテスト実行フェーズ
    SANDBOX_RUNNER_URL が設定されている場合は信頼済みサンドボックスランナーにHTTPリクエストを送信。
    未設定の場合はサブプロセスで戦略を検証（従来の venv 実行パス）。
    metrics_since: YYYY-MM-DD形式の日付。指定された場合、その日以降のデータを対象にメトリクスを計算。
    """
    strategy = hypothesis["strategy"]
    symbol = hypothesis["symbol"]
    params = hypothesis["params"]
    runner_url = os.environ.get("SANDBOX_RUNNER_URL")

    if runner_url:
        return _run_backtest_sandbox(runner_url, strategy, symbol, params, metrics_since)
    else:
        return _run_backtest_subprocess(strategy, symbol, params, metrics_since)


def _run_backtest_sandbox(runner_url, strategy, symbol, params, metrics_since=None):
    """信頼済みサンドボックスランナー（Trusted Runner）経由でバックテストを実行"""
    url = f"{runner_url.rstrip('/')}/run"
    payload = {
        "strategy": strategy,
        "symbol": symbol,
        "params": params
    }
    if metrics_since:
        payload["metrics_since"] = metrics_since
    body = json.dumps(payload).encode("utf-8")

    print(f"  🔬 バックテスト実行中 (sandbox): {strategy} on {symbol}...")

    try:
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            response_body = resp.read().decode("utf-8")
            status = resp.status

        if status != 200:
            return {"error": f"Sandbox runner returned HTTP {status}: {response_body[:500]}", "error_type": "infra"}

        result = json.loads(response_body)
        # Runner-reported error (strategy code failure) → tag as code
        if isinstance(result, dict) and "error" in result:
            result.setdefault("error_type", "code")
        return result

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        return {"error": f"Sandbox runner HTTP {e.code}: {error_body}", "error_type": "infra"}
    except urllib.error.URLError as e:
        return {"error": f"Sandbox runner connection error: {e.reason}", "error_type": "infra"}
    except json.JSONDecodeError as e:
        return {"error": f"Sandbox runner JSON parse error: {e}", "error_type": "infra"}
    except Exception as e:
        return {"error": f"Sandbox runner error: {e}", "error_type": "infra"}


def _run_backtest_subprocess(strategy, symbol, params, metrics_since=None):
    """サブプロセスでバックテストを実行（従来の venv 実行パス）"""
    cmd = [
        sys.executable,
        str(BASE_DIR / "backtests" / "backtest_engine.py"),
        "--strategy", strategy,
        "--symbol", symbol,
        "--params", json.dumps(params),
    ]
    if metrics_since:
        cmd.extend(["--metrics-since", metrics_since])

    print(f"  🔬 バックテスト実行中: {strategy} on {symbol}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return {"error": result.stderr, "error_type": "code"}

        # stdoutからJSONブロックを抽出（最初の{から最後の}まで）
        stdout = result.stdout
        start_idx = stdout.find('{')
        end_idx = stdout.rfind('}')

        if start_idx >= 0 and end_idx > start_idx:
            json_str = stdout[start_idx:end_idx + 1]
            return json.loads(json_str)

        return {"error": "Could not parse output", "error_type": "infra", "raw": stdout[:500]}

    except subprocess.TimeoutExpired:
        return {"error": "Timeout (120s)", "error_type": "infra"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "error_type": "infra"}


def evaluate_result(hypothesis, result, knowledge):
    """
    Multi-gate evaluation with overfitting countermeasures.
    Returns (verdict, evaluation_dict).
    """
    if "error" in result:
        error_type = result.get("error_type", "error")  # safe default: unknown = error
        if error_type == "infra":
            return "error", {"verdict": "error", "error": result["error"],
                             "error_type": "infra",
                             "reasons": [f"🔧 Infra error (not counted as rejection): {result['error']}"]}
        elif error_type == "code":
            return "code_error", {"verdict": "code_error", "error": result["error"],
                                  "error_type": "code",
                                  "reasons": [f"💻 Code error (strategy code crashed): {result['error']}"]}
        else:
            # Unknown error_type → safe default: error (never rejection)
            return "error", {"verdict": "error", "error": result["error"],
                             "error_type": "unknown",
                             "reasons": [f"🔧 Unknown error (not counted as rejection): {result['error']}"]}

    wf = result.get("walkforward", {})
    if not wf.get("enabled"):
        return "rejected", {"verdict": "rejected", "error": "walkforward disabled", "reasons": ["❌ Walkforward not enabled"]}

    val_metrics = result.get("out_of_sample", {})
    holdout_metrics = result.get("holdout", {})
    is_metrics = result.get("in_sample", {})

    val_sharpe = val_metrics.get("sharpe_ratio", -999)
    val_return = val_metrics.get("total_return_pct", -999)
    val_max_dd = val_metrics.get("max_drawdown_pct", -999)
    T_val = val_metrics.get("num_days", 252)

    strategy = hypothesis["strategy"]
    symbol = hypothesis["symbol"]

    # Count N_family: tested_combinations with same (strategy, symbol)
    N_family = sum(
        1 for tc in knowledge.get("tested_combinations", [])
        if tc.get("strategy") == strategy and tc.get("symbol") == symbol
    )
    effective_min_sharpe = compute_effective_min_sharpe(N_family, T_val)

    reasons = []
    gate_results = {}

    # --- Gate (a): Validation gate ---
    val_pass = (
        val_sharpe >= effective_min_sharpe
        and val_return > 0
        and val_max_dd >= MAX_DRAWDOWN_LIMIT
    )
    gate_results["validation"] = val_pass

    if val_sharpe >= effective_min_sharpe:
        reasons.append(f"✅ Val Sharpe {val_sharpe:.2f} >= {effective_min_sharpe:.2f} (deflated, N={N_family}, T={T_val})")
    else:
        reasons.append(f"❌ Val Sharpe {val_sharpe:.2f} < {effective_min_sharpe:.2f} (deflated, N={N_family}, T={T_val})")

    if val_return > 0:
        reasons.append(f"✅ Val Return {val_return:.1f}% > 0%")
    else:
        reasons.append(f"❌ Val Return {val_return:.1f}% <= 0%")

    if val_max_dd >= MAX_DRAWDOWN_LIMIT:
        reasons.append(f"✅ Val Drawdown {val_max_dd:.1f}% >= {MAX_DRAWDOWN_LIMIT}%")
    else:
        reasons.append(f"❌ Val Drawdown {val_max_dd:.1f}% < {MAX_DRAWDOWN_LIMIT}%")

    if not val_pass:
        evaluation = {
            "verdict": "rejected",
            "sharpe_ratio": val_sharpe,
            "total_return_pct": val_return,
            "max_drawdown_pct": val_max_dd,
            "reasons": reasons,
            "gate_results": gate_results,
            "effective_min_sharpe": round(effective_min_sharpe, 4),
        }
        if is_metrics:
            evaluation["in_sample"] = is_metrics
        return "rejected", evaluation

    # --- Gate (b): Deflation gate (embedded in (a) via effective_min_sharpe) ---
    reasons.append(f"📐 Deflated threshold: {effective_min_sharpe:.2f} (base={MIN_SHARPE_BASE}, N_family={N_family}, T_val={T_val})")
    gate_results["deflation"] = True

    # --- Gate (c): Holdout confirmation ---
    holdout_sharpe = holdout_metrics.get("sharpe_ratio", -999)
    holdout_return = holdout_metrics.get("total_return_pct", -999)
    # Stricter gate: holdout Sharpe must reach min(0.5, 0.5 * val_sharpe).
    # Floor at 0.5 absolute; scaled down for modest val Sharpe so the bar
    # is never harsher than half the validation performance.
    holdout_threshold = min(0.5, 0.5 * val_sharpe)
    holdout_pass = (holdout_sharpe >= holdout_threshold) and (holdout_return > 0)
    gate_results["holdout"] = holdout_pass

    if holdout_sharpe >= holdout_threshold:
        reasons.append(f"✅ Holdout Sharpe {holdout_sharpe:.2f} >= {holdout_threshold:.2f} (threshold=min(0.5, 0.5*val))")
    else:
        reasons.append(f"❌ Holdout Sharpe {holdout_sharpe:.2f} < {holdout_threshold:.2f} (threshold=min(0.5, 0.5*val))")

    if holdout_return > 0:
        reasons.append(f"✅ Holdout Return {holdout_return:.1f}% > 0")
    else:
        reasons.append(f"❌ Holdout Return {holdout_return:.1f}% <= 0")

    if not holdout_pass:
        evaluation = {
            "verdict": "rejected",
            "sharpe_ratio": val_sharpe,
            "total_return_pct": val_return,
            "max_drawdown_pct": val_max_dd,
            "holdout_sharpe": holdout_sharpe,
            "holdout_return_pct": holdout_return,
            "reasons": reasons,
            "gate_results": gate_results,
            "effective_min_sharpe": round(effective_min_sharpe, 4),
        }
        if is_metrics:
            evaluation["in_sample"] = is_metrics
        return "rejected", evaluation

    # --- Gate (d): Cluster dedup ---
    adopted = knowledge.get("adopted", [])
    cluster_id = str(uuid.uuid4())[:8]
    incumbent_idx = None
    for idx, entry in enumerate(adopted):
        eh = entry.get("hypothesis", {})
        if eh.get("strategy") != strategy or eh.get("symbol") != symbol:
            continue
        if _params_within_cluster(hypothesis["params"], eh.get("params", {}), STRATEGY_TEMPLATES):
            incumbent_idx = idx
            cluster_id = entry.get("cluster_id", cluster_id)
            break

    if incumbent_idx is not None:
        incumbent = adopted[incumbent_idx]
        inc_holdout = incumbent.get("evaluation", {}).get("holdout_sharpe", -999)
        if holdout_sharpe > inc_holdout:
            reasons.append(f"🔄 Cluster replace: holdout Sharpe {holdout_sharpe:.2f} > incumbent {inc_holdout:.2f}")
            knowledge.setdefault("superseded", []).append(incumbent)
            adopted.pop(incumbent_idx)
            gate_results["cluster"] = "replaced"
        else:
            reasons.append(f"❌ Cluster dedup: holdout Sharpe {holdout_sharpe:.2f} <= incumbent {inc_holdout:.2f}")
            gate_results["cluster"] = "duplicate_cluster"
            evaluation = {
                "verdict": "rejected",
                "sharpe_ratio": val_sharpe,
                "total_return_pct": val_return,
                "max_drawdown_pct": val_max_dd,
                "holdout_sharpe": holdout_sharpe,
                "holdout_return_pct": holdout_return,
                "reasons": reasons,
                "gate_results": gate_results,
                "effective_min_sharpe": round(effective_min_sharpe, 4),
            }
            if is_metrics:
                evaluation["in_sample"] = is_metrics
            return "rejected", evaluation
    else:
        gate_results["cluster"] = "new"

    # All gates passed
    reasons.insert(0, "🏆 ALL GATES PASSED")
    evaluation = {
        "verdict": "adopted",
        "sharpe_ratio": val_sharpe,
        "total_return_pct": val_return,
        "max_drawdown_pct": val_max_dd,
        "holdout_sharpe": holdout_sharpe,
        "holdout_return_pct": holdout_return,
        "reasons": reasons,
        "gate_results": gate_results,
        "effective_min_sharpe": round(effective_min_sharpe, 4),
        "cluster_id": cluster_id,
    }
    if is_metrics:
        evaluation["in_sample"] = is_metrics

    return "adopted", evaluation


# ---------------------------------------------------------------------------
# Near-miss classification
# ---------------------------------------------------------------------------

def _classify_near_miss(hypothesis, evaluation):
    """Classify a rejected evaluation as near-miss if conditions met.

    Returns a near-miss dict or None.
    """
    gate_results = evaluation.get("gate_results", {})
    val_sharpe = evaluation.get("sharpe_ratio", -999)
    eff_threshold = evaluation.get("effective_min_sharpe", 0)
    holdout_sharpe = evaluation.get("holdout_sharpe")  # None if holdout never ran

    failed_gate = None

    # (b) passed validation Sharpe gate but failed later non-dedup gate
    # Check this FIRST: it implies stronger signal than the 90% threshold rule
    if not gate_results.get("validation", True) and eff_threshold > 0:
        # Validation failed but Sharpe itself passed -- drawdown or return
        if val_sharpe >= eff_threshold:
            if evaluation.get("max_drawdown_pct", 0) < MAX_DRAWDOWN_LIMIT:
                failed_gate = "max_drawdown"
            else:
                failed_gate = "val_return"

    elif gate_results.get("validation", False):
        # Validation passed -- check later gates
        if not gate_results.get("holdout", True):
            failed_gate = "holdout"
        elif gate_results.get("cluster") == "duplicate_cluster":
            return None  # dedup excluded
        else:
            # All structural gates passed -- check extra_criteria
            reasons = evaluation.get("reasons", [])
            if any("Extra criterion failed" in r for r in reasons):
                failed_gate = "extra_criteria"

    # (a) val_sharpe >= 90% of deflated threshold but validation failed
    # Fallback: only applies when (b) didn't match
    if failed_gate is None:
        if not gate_results.get("validation", True) and eff_threshold > 0:
            if val_sharpe >= 0.9 * eff_threshold:
                failed_gate = "val_sharpe_90pct"

    if failed_gate is None:
        return None

    return {
        "id": hypothesis["id"],
        "strategy": hypothesis["strategy"],
        "symbol": hypothesis["symbol"],
        "params": hypothesis["params"],
        "val_sharpe": val_sharpe,
        "deflated_threshold": round(eff_threshold, 4),
        "holdout_sharpe": holdout_sharpe,
        "failed_gate": failed_gate,
        "date": datetime.utcnow().isoformat() + "Z",
    }


def _record_near_miss(hypothesis, evaluation, knowledge):
    """If the rejected evaluation qualifies as a near-miss, record it.

    Caps the near_misses list at 30 most recent entries.
    """
    nm = _classify_near_miss(hypothesis, evaluation)
    if nm is None:
        return
    # Machine-greppable line so the cron reporter doesn't have to do arithmetic
    holdout = nm["holdout_sharpe"]
    print(f"NEAR_MISS {nm['strategy']}/{nm['symbol']} val={nm['val_sharpe']:.2f} "
          f"thresh={nm['deflated_threshold']:.2f} holdout={holdout if holdout is not None else 'n/a'} "
          f"gate={nm['failed_gate']}")
    near_misses = knowledge.setdefault("near_misses", [])
    near_misses.append(nm)
    # Cap at 30 most recent
    if len(near_misses) > 30:
        knowledge["near_misses"] = near_misses[-30:]


def save_result(hypothesis, result, verdict, evaluation):
    """結果を保存"""
    record = {
        "hypothesis": hypothesis,
        "backtest_result": result,
        "evaluation": evaluation,
        "verdict": verdict,
        "tested_at": datetime.now().isoformat()
    }
    
    result_file = RESULTS_DIR / f"{hypothesis['id']}.json"
    result_file.write_text(json.dumps(record, indent=2, default=str))
    
    return record


def print_report(knowledge):
    """ナレッジベースのサマリー"""
    adopted = len(knowledge["adopted"])
    rejected = len(knowledge["rejected"])
    superseded = len(knowledge.get("superseded", []))
    errors = knowledge.get("errors", [])
    n_infra = sum(1 for e in errors if e.get("evaluation", {}).get("error_type") == "infra")
    n_code = sum(1 for e in errors if e.get("evaluation", {}).get("error_type") == "code")
    total = adopted + rejected

    print(f"\n{'='*60}")
    print(f"📊 Alpha Discovery Report (Iteration {knowledge['iterations']})")
    print(f"{'='*60}")
    print(f"  採用: {adopted}件 / テスト: {total}件 (採用率: {adopted/max(total,1)*100:.1f}%)")
    if superseded > 0:
        print(f"  上書き: {superseded}件 (superseded)")
    if n_infra > 0:
        print(f"  🔧 インフラエラー: {n_infra}件 (not counted as rejection)")
    if n_code > 0:
        print(f"  💻 コードエラー: {n_code}件 (strategy code crashed)")

    if adopted > 0:
        print(f"\n  🏆 採用された戦略:")
        for s in knowledge["adopted"][-5:]:
            print(f"     • {s['hypothesis']['description']} | Sharpe: {s['evaluation']['sharpe_ratio']:.2f} | Return: {s['evaluation']['total_return_pct']:.1f}%")
    
    print(f"{'='*60}\n")


def _generate_with_llm_fallback(knowledge):
    """
    LLM-driven hypothesis generation with transparent fallback to random.
    Returns (hypothesis, source_label) where source_label is "(llm)" or "(random)".
    Never raises — always returns a valid hypothesis.
    """
    try:
        from llm_hypothesis import generate as llm_generate
        hypothesis = llm_generate(knowledge, STRATEGY_TEMPLATES)

        # Duplicate check against already-tested combinations
        for tested in knowledge.get("tested_combinations", []):
            if (tested["strategy"] == hypothesis["strategy"] and
                tested["symbol"] == hypothesis["symbol"] and
                tested["params"] == hypothesis["params"]):
                print(f"  ⚠️ LLM提案が既存と重複 → (random) にフォールバック")
                return generate_hypothesis(knowledge), "(random)"

        return hypothesis, "(llm)"
    except Exception as e:
        print(f"  ⚠️ LLM仮説生成失敗: {e} → (random) にフォールバック")
        return generate_hypothesis(knowledge), "(random)"


# ---------------------------------------------------------------------------
# Manifest evaluation helper (Phase 1 PR-G)
# ---------------------------------------------------------------------------

def _evaluate_manifest_result(runner_result, hypothesis, knowledge):
    """Evaluate manifest runner results using the same gates as evaluate_result.

    The manifest runner already performs train/val/holdout splits internally,
    so we apply gates directly to its returned metrics.  This avoids
    re-interpreting walkforward/full-sample splits that don't exist here.

    Returns (verdict, evaluation_dict) — same shape as evaluate_result.
    """
    # Note: runner_result is the FULL parsed runner response (includes metrics + n_days)
    if "metrics" not in runner_result:
        return "error", {
            "verdict": "error",
            "error": "Malformed runner response: missing 'metrics'",
            "error_type": "infra",
            "reasons": [f"🔧 Infra error (not counted as rejection): missing 'metrics' in runner response"],
        }

    metrics = runner_result["metrics"]
    val_sharpe = metrics.get("val_sharpe", -999)
    val_max_dd = metrics.get("val_max_drawdown_pct", -999)
    val_return = metrics.get("val_total_return_pct", -999)
    holdout_sharpe = metrics.get("holdout_sharpe", -999)
    holdout_return = metrics.get("holdout_total_return_pct", -999)
    T_val = runner_result.get("n_days", 252)

    strategy = hypothesis["strategy"]
    symbol = hypothesis["symbol"]

    # N_family: count from families dict (updated by update_family_aggregates)
    fam_key = _family_key(strategy, symbol)
    families = knowledge.get("families", {})
    N_family = families.get(fam_key, {}).get("n_trials", 0)
    effective_min_sharpe = compute_effective_min_sharpe(N_family, T_val)

    reasons = []
    gate_results = {}

    # --- Gate (a): Validation gate ---
    val_pass = (
        val_sharpe >= effective_min_sharpe
        and val_return > 0
        and val_max_dd >= MAX_DRAWDOWN_LIMIT
    )
    gate_results["validation"] = val_pass

    if val_sharpe >= effective_min_sharpe:
        reasons.append(f"✅ Val Sharpe {val_sharpe:.2f} >= {effective_min_sharpe:.2f} (deflated, N={N_family}, T={T_val})")
    else:
        reasons.append(f"❌ Val Sharpe {val_sharpe:.2f} < {effective_min_sharpe:.2f} (deflated, N={N_family}, T={T_val})")

    if val_return > 0:
        reasons.append(f"✅ Val Return {val_return:.1f}% > 0%")
    else:
        reasons.append(f"❌ Val Return {val_return:.1f}% <= 0%")

    if val_max_dd >= MAX_DRAWDOWN_LIMIT:
        reasons.append(f"✅ Val Drawdown {val_max_dd:.1f}% >= {MAX_DRAWDOWN_LIMIT}%")
    else:
        reasons.append(f"❌ Val Drawdown {val_max_dd:.1f}% < {MAX_DRAWDOWN_LIMIT}%")

    if not val_pass:
        evaluation = {
            "verdict": "rejected",
            "sharpe_ratio": val_sharpe,
            "total_return_pct": val_return,
            "max_drawdown_pct": val_max_dd,
            "reasons": reasons,
            "gate_results": gate_results,
            "effective_min_sharpe": round(effective_min_sharpe, 4),
        }
        return "rejected", evaluation

    # --- Gate (b): Deflation gate (informational — embedded in (a)) ---
    reasons.append(f"📐 Deflated threshold: {effective_min_sharpe:.2f} (base={MIN_SHARPE_BASE}, N_family={N_family}, T_val={T_val})")
    gate_results["deflation"] = True

    # --- Gate (c): Holdout confirmation ---
    holdout_threshold = min(0.5, 0.5 * val_sharpe)
    holdout_pass = (holdout_sharpe >= holdout_threshold) and (holdout_return > 0)
    gate_results["holdout"] = holdout_pass

    if holdout_sharpe >= holdout_threshold:
        reasons.append(f"✅ Holdout Sharpe {holdout_sharpe:.2f} >= {holdout_threshold:.2f} (threshold=min(0.5, 0.5*val))")
    else:
        reasons.append(f"❌ Holdout Sharpe {holdout_sharpe:.2f} < {holdout_threshold:.2f} (threshold=min(0.5, 0.5*val))")

    if holdout_return > 0:
        reasons.append(f"✅ Holdout Return {holdout_return:.1f}% > 0")
    else:
        reasons.append(f"❌ Holdout Return {holdout_return:.1f}% <= 0")

    if not holdout_pass:
        evaluation = {
            "verdict": "rejected",
            "sharpe_ratio": val_sharpe,
            "total_return_pct": val_return,
            "max_drawdown_pct": val_max_dd,
            "holdout_sharpe": holdout_sharpe,
            "holdout_return_pct": holdout_return,
            "reasons": reasons,
            "gate_results": gate_results,
            "effective_min_sharpe": round(effective_min_sharpe, 4),
        }
        return "rejected", evaluation

    # No cluster dedup for manifest entries — each manifest name / universe
    # pair is a unique cluster by construction.
    gate_results["cluster"] = "new"

    # All gates passed
    reasons.insert(0, "🏆 ALL GATES PASSED")
    evaluation = {
        "verdict": "adopted",
        "sharpe_ratio": val_sharpe,
        "total_return_pct": val_return,
        "max_drawdown_pct": val_max_dd,
        "holdout_sharpe": holdout_sharpe,
        "holdout_return_pct": holdout_return,
        "reasons": reasons,
        "gate_results": gate_results,
        "effective_min_sharpe": round(effective_min_sharpe, 4),
    }

    # Attach expert_extras if present
    if runner_result.get("expert_extras"):
        evaluation["expert_extras"] = runner_result["expert_extras"]

    return "adopted", evaluation


# ---------------------------------------------------------------------------
# Backlog consumption helpers
# ---------------------------------------------------------------------------

_EXTRA_CRITERIA_METRICS = {"sharpe_ratio", "total_return_pct", "max_drawdown_pct"}
_EXTRA_CRITERIA_OPS = {">=", "<=", ">", "<"}


def _parse_extra_criterion(crit_str):
    """Parse a criterion string of the form '<metric> <op> <number>'.

    Returns (metric, op, value) or (None, None, None) for unparseable.
    """
    parts = crit_str.split()
    if len(parts) < 3:
        return None, None, None
    # Metric may be "sharpe_ratio" or "max_drawdown_pct" (multi-word)?
    # The spec says single-metric — try the first token as metric and last token as value
    metric = parts[0].strip()
    if metric not in _EXTRA_CRITERIA_METRICS:
        return None, None, None
    # op should be the second token
    op = parts[1].strip()
    if op not in _EXTRA_CRITERIA_OPS:
        return None, None, None
    try:
        value = float(parts[2])
    except (ValueError, IndexError):
        return None, None, None
    return metric, op, value


def _check_extra_criteria(metrics, extra_criteria):
    """Evaluate extra_criteria against validation-split metrics.

    Returns (all_pass: bool, failures: list of str).
    Unparseable criteria are logged with a warning and ignored.
    Failures are additive: they can only ADD strictness on top of global gates.
    """
    failures = []
    for crit_str in extra_criteria:
        metric, op, value = _parse_extra_criterion(crit_str)
        if metric is None:
            print(f"  ⚠️ 解釈不能なcriteria (無視): {crit_str}")
            continue

        actual = metrics.get(metric, -999)
        passed = False
        if op == ">=":
            passed = actual >= value
        elif op == "<=":
            passed = actual <= value
        elif op == ">":
            passed = actual > value
        elif op == "<":
            passed = actual < value

        if not passed:
            failures.append(f"❌ Extra criterion failed: {metric} {op} {value} (actual={actual:.2f})")

    return len(failures) == 0, failures


def _consume_backlog_entry(knowledge):
    """Try to consume one pending backlog entry.

    Returns (entry, hypothesis, result, verdict, evaluation) if consumed,
    or None if backlog is empty or entry cannot run.
    """
    from backlog import Backlog

    backlog_path = os.environ.get("BACKLOG_PATH", str(BASE_DIR / "backlog.json"))
    bl = Backlog(backlog_path)
    entry = bl.next_pending()

    if entry is None:
        return None

    etype = entry["type"]
    spec = entry["spec"]
    source_info = entry["source"]
    priority = entry["priority"]
    eval_plan = entry.get("eval_plan", {})
    extra_criteria = eval_plan.get("extra_criteria", [])

    # Log
    if etype == "param":
        tag = f"[param] {spec['strategy']} on {spec['symbol']}"
    elif etype == "manifest":
        manifest_name = spec.get("name", "?")
        universe_size = 0
        for ds in spec.get("data_sources", []):
            if ds.get("type") == "ohlcv" and ds.get("universe"):
                universe_size = len(ds["universe"])
                break
        tag = f"[manifest] {manifest_name} on {universe_size} symbols"
    else:
        tag = f"[code] {spec.get('name', '?')} on {spec['symbol']}"
    print(f"📥 バックログ消化: {tag} (priority {priority}, source {source_info.get('kind', '?')}:{source_info.get('ref', '?')})")

    # Mark as testing
    bl.mark(entry["id"], "testing")

    runner_url = os.environ.get("SANDBOX_RUNNER_URL")

    # ── Route by type ──
    if etype == "manifest":
        # Manifest entry: POST to /run_manifest (Phase 1 PR-G)
        if not runner_url:
            print(f"  ⚠️ SANDBOX_RUNNER_URL未設定 — マニフェストエントリをpendingに戻します")
            bl.mark(entry["id"], "pending")
            return None

        manifest_spec = spec  # full manifest dict from ideation (manifest.to_dict())
        manifest_name = manifest_spec.get("name", "unknown")

        # Extract universe from data_sources
        universe = []
        for ds in manifest_spec.get("data_sources", []):
            if ds.get("type") == "ohlcv" and ds.get("universe"):
                universe = ds["universe"]
                break
        sorted_universe = sorted(universe)
        universe_hash = hashlib.sha256(
            json.dumps(sorted_universe, sort_keys=True).encode()
        ).hexdigest()[:8]

        execution_mode = manifest_spec.get("execution_mode", "structured")

        hypothesis = {
            "id": f"bl_{entry['id'][:12]}",
            "strategy": f"manifest:{manifest_name}",
            "symbol": f"universe:{universe_hash}",
            "params": {
                "universe_size": len(universe),
                "execution_mode": execution_mode,
                "primary_metric": "sharpe",
            },
            "description": f"Backlog manifest: {manifest_name} on {len(universe)} symbols [{execution_mode}]",
            "generated_at": datetime.now().isoformat(),
        }

        # POST to /run_manifest
        url = f"{runner_url.rstrip('/')}/run_manifest"
        body = json.dumps(manifest_spec).encode("utf-8")

        print(f"  🔬 マニフェスト実行中 (sandbox): {manifest_name} on {len(universe)} symbols...")

        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                response_body = resp.read().decode("utf-8")
                status = resp.status
            if status != 200:
                result = {"error": f"Sandbox runner returned HTTP {status}: {response_body[:500]}",
                          "error_type": "infra"}
            else:
                parsed = json.loads(response_body)
                if not isinstance(parsed, dict):
                    result = {"error": "Malformed runner response (not a dict)", "error_type": "infra"}
                elif parsed.get("status") == "error":
                    err_type = parsed.get("error_type", "")
                    if err_type in ("manifest", "code"):
                        # Agent's fault → code_error (do not retry as infra).
                        # 'manifest' = manifest validation failed.
                        # 'code' = user code raised at import/exec time (e.g.
                        # ModuleNotFoundError, SyntaxError, or runtime error
                        # inside generate_signals/run).
                        result = {"error": parsed.get("error", f"{err_type} error"),
                                  "error_type": "code"}
                    else:
                        # 'infra' or unknown → treat as infra (retry-eligible).
                        result = {"error": parsed.get("error", "runner reported error"),
                                  "error_type": "infra"}
                elif parsed.get("status") == "ok":
                    result = parsed
                else:
                    result = {"error": f"Malformed runner response: unknown status '{parsed.get('status', 'missing')}'",
                              "error_type": "infra"}
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            result = {"error": f"Sandbox runner HTTP {e.code}: {error_body}", "error_type": "infra"}
        except urllib.error.URLError as e:
            result = {"error": f"Sandbox runner connection error: {e.reason}", "error_type": "infra"}
        except json.JSONDecodeError as e:
            result = {"error": f"Sandbox runner JSON parse error: {e}", "error_type": "infra"}
        except Exception as e:
            result = {"error": f"Sandbox runner error: {e}", "error_type": "infra"}
    elif etype == "param":
        # Param entry: use existing sandbox backtest path
        hypothesis = {
            "id": f"bl_{entry['id'][:12]}",
            "strategy": spec["strategy"],
            "symbol": spec["symbol"],
            "params": spec["params"],
            "description": f"Backlog param: {spec['strategy']} on {spec['symbol']}",
            "generated_at": datetime.now().isoformat(),
        }
        result = run_backtest(hypothesis)

    elif etype == "code":
        # Code entry: POST to SANDBOX_RUNNER_URL/run_code
        if not runner_url:
            print(f"  ⚠️ SANDBOX_RUNNER_URL未設定 — コードエントリをpendingに戻します")
            bl.mark(entry["id"], "pending")
            return None

        code_str = spec["code"]
        code_b64 = base64.b64encode(code_str.encode("utf-8")).decode("ascii")

        # Compute code_hash locally before sending
        code_hash = hashlib.sha256(code_str.encode("utf-8")).hexdigest()

        # Check duplicate: same code_hash + symbol in tested history
        for rec in knowledge.get("adopted", []) + knowledge.get("rejected", []):
            ev = rec.get("evaluation", {})
            hyp = rec.get("hypothesis", {})
            if ev.get("code_hash") == code_hash and hyp.get("symbol") == spec["symbol"]:
                print(f"  ⚠️ 重複コードハッシュ (code_hash={code_hash[:12]}..., symbol={spec['symbol']}) → スキップ")
                bl.mark(entry["id"], "done_rejected", {
                    "verdict": "rejected",
                    "reason": "duplicate_code",
                    "summary": f"コードハッシュ重複: code_hash={code_hash[:12]}... on {spec['symbol']}",
                    "finished_at": datetime.now().isoformat(),
                })
                return None

        url = f"{runner_url.rstrip('/')}/run_code"
        body = json.dumps({
            "code_b64": code_b64,
            "symbol": spec["symbol"],
        }).encode("utf-8")

        hypothesis = {
            "id": f"bl_{entry['id'][:12]}",
            "strategy": "codegen",
            "symbol": spec["symbol"],
            "params": {},
            "description": f"Backlog code: {spec.get('name', 'codegen')} on {spec['symbol']}",
            "generated_at": datetime.now().isoformat(),
        }

        print(f"  🔬 コードバックテスト実行中 (sandbox): {spec.get('name', 'codegen')} on {spec['symbol']}...")

        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=240) as resp:
                response_body = resp.read().decode("utf-8")
                status = resp.status
            if status != 200:
                result = {"error": f"Sandbox runner returned HTTP {status}: {response_body[:500]}", "error_type": "infra"}
            else:
                result = json.loads(response_body)
                # Runner-reported error (strategy code failure) → tag as code
                if isinstance(result, dict) and "error" in result:
                    result.setdefault("error_type", "code")
                # Inject code_hash from harness response (or fallback to local)
                if "code_hash" not in result:
                    result["code_hash"] = code_hash
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            result = {"error": f"Sandbox runner HTTP {e.code}: {error_body}", "error_type": "infra"}
        except urllib.error.URLError as e:
            result = {"error": f"Sandbox runner connection error: {e.reason}", "error_type": "infra"}
        except json.JSONDecodeError as e:
            result = {"error": f"Sandbox runner JSON parse error: {e}", "error_type": "infra"}
        except Exception as e:
            result = {"error": f"Sandbox runner error: {e}", "error_type": "infra"}
    else:
        bl.mark(entry["id"], "pending")
        return None

    # ── Evaluate ──
    if etype == "manifest" and isinstance(result, dict) and result.get("status") == "ok":
        # Manifest runner already did train/val/holdout splits internally.
        # Use the dedicated manifest evaluator that reads its metrics directly.
        verdict, evaluation = _evaluate_manifest_result(result, hypothesis, knowledge)
    else:
        verdict, evaluation = evaluate_result(hypothesis, result, knowledge)

    # ── Extra criteria ──
    if verdict == "adopted" and extra_criteria:
        # Use validation-split metrics (out_of_sample in result)
        metrics = result.get("out_of_sample", result)
        extra_pass, extra_failures = _check_extra_criteria(metrics, extra_criteria)
        if not extra_pass:
            evaluation["reasons"].extend(extra_failures)
            verdict = "rejected"
            evaluation["verdict"] = "rejected"

    # ── Summary line ──
    if verdict == "adopted":
        val_sharpe = evaluation.get("sharpe_ratio", 0)
        holdout_sharpe = evaluation.get("holdout_sharpe", 0)
        summary = f"val Sharpe {val_sharpe:.2f} / holdout Sharpe {holdout_sharpe:.2f}"
    else:
        gate_results = evaluation.get("gate_results", {})
        if evaluation.get("error"):
            summary = f"failed gate: {evaluation['error'][:100]}"
        elif not gate_results.get("validation", True):
            summary = f"failed gate: validation (Sharpe {evaluation.get('sharpe_ratio', -999):.2f})"
        elif not gate_results.get("holdout", True):
            summary = f"failed gate: holdout (Sharpe {evaluation.get('holdout_sharpe', -999):.2f})"
        else:
            # Check for extra_criteria failures in reasons
            reasons = evaluation.get("reasons", [])
            extra_fail = [r for r in reasons if "Extra criterion failed" in r]
            if extra_fail:
                summary = extra_fail[0].replace("❌ Extra criterion failed: ", "")
            else:
                summary = f"failed gate: cluster dedup or other"

    # Route error verdicts for backlog entries
    if verdict == "error":
        # Infra error: retry with attempts counter
        # attempts persists inside result (backlog.mark writes only status + result),
        # so read from result dict, not from entry top-level.
        attempts = (entry.get("result") or {}).get("attempts", 0) + 1
        if attempts < 3:
            error_text = evaluation.get("error", "")[:200]
            bl.mark(entry["id"], "pending", {
                "verdict": verdict,
                "error": error_text,
                "summary": f"infra error (attempt {attempts}/3): {error_text}",
                "finished_at": datetime.now().isoformat(),
                "attempts": attempts,
            })
        else:
            bl.mark(entry["id"], "done_error", {
                "verdict": verdict,
                "error": evaluation.get("error", "")[:200],
                "summary": f"infra error after {attempts} attempts: {evaluation.get('error', '')[:200]}",
                "finished_at": datetime.now().isoformat(),
                "attempts": attempts,
            })
    elif verdict == "code_error":
        bl.mark(entry["id"], "done_error", {
            "verdict": verdict,
            "error": evaluation.get("error", "")[:200],
            "summary": f"code error: {evaluation.get('error', '')[:200]}",
            "finished_at": datetime.now().isoformat(),
        })
    else:
        finish_status = "done_adopted" if verdict == "adopted" else "done_rejected"
        bl.mark(entry["id"], finish_status, {
            "verdict": verdict,
            "summary": summary,
            "finished_at": datetime.now().isoformat(),
        })

    return entry, hypothesis, result, verdict, evaluation


# ======================================================================


def run_loop(num_iterations=3):
    """
    メインPDCAループ
    """
    use_llm = os.environ.get("USE_LLM_HYPOTHESIS") == "1"

    print("=" * 60)
    print("🚀 Autonomous Alpha Discovery Loop 開始")
    print(f"   開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print(f"   イテレーション数: {num_iterations}")
    print(f"   LLM仮説生成: {'有効' if use_llm else '無効'}")
    print("=" * 60)
    
    knowledge = load_knowledge()
    
    for i in range(num_iterations):
        knowledge["iterations"] += 1
        print(f"\n🔄 Iteration {i+1}/{num_iterations}")
        print("-" * 40)

        # 0. Backlog consumption: try to consume a pending entry first
        consumed = _consume_backlog_entry(knowledge)
        if consumed:
            entry_bk, hypothesis, result, verdict, evaluation = consumed

            print(f"  📋 判定: {verdict.upper()}")
            if isinstance(evaluation, dict) and "reasons" in evaluation:
                for reason in evaluation["reasons"]:
                    print(f"     {reason}")

            # 5. 蓄積 (with source attribution)
            record = save_result(hypothesis, result, verdict, evaluation)
            # Add source attribution from backlog entry
            record["source"] = entry_bk.get("source", {})

            if verdict in ("error", "code_error"):
                knowledge.setdefault("errors", []).append(record)
                if len(knowledge["errors"]) > 100:
                    knowledge["errors"] = knowledge["errors"][-100:]
            elif verdict == "adopted":
                record["cluster_id"] = evaluation.get("cluster_id", "unknown")
                knowledge["adopted"].append(record)
            else:
                _record_near_miss(hypothesis, evaluation, knowledge)
                knowledge["rejected"].append(record)

            if verdict not in ("error", "code_error"):
                update_family_aggregates(knowledge, record)
                knowledge["tested"].append(hypothesis["id"])
                knowledge["tested_combinations"].append({
                    "strategy": hypothesis["strategy"],
                    "symbol": hypothesis["symbol"],
                    "params": hypothesis["params"]
                })

            save_knowledge(knowledge)

            if i < num_iterations - 1:
                time.sleep(2)
            continue

        # 1. 仮説生成 (fallback: no backlog entry available)
        if use_llm:
            hypothesis, source_label = _generate_with_llm_fallback(knowledge)
            print(f"  🧠 ソース: {source_label}")
            if "rationale" in hypothesis:
                print(f"  💭 根拠: {hypothesis['rationale']}")
        else:
            hypothesis = generate_hypothesis(knowledge)
            source_label = "(random)"
        
        # 2. Exhausted-cluster pre-block check
        exhausted, n_failures, best_fail_sharpe = _check_exhausted_cluster(hypothesis, knowledge)
        if exhausted:
            print(f"  ⛔ 枯渇クラスタ ({n_failures} failures, best Sharpe {best_fail_sharpe:.2f}) → バックテスト省略")
            evaluation = {
                "verdict": "rejected",
                "sharpe_ratio": best_fail_sharpe,
                "total_return_pct": -999,
                "max_drawdown_pct": -999,
                "reasons": [f"⛔ Exhausted cluster: {n_failures} nearby failures, best val Sharpe {best_fail_sharpe:.2f} < 0"],
                "gate_results": {"validation": False, "cluster": "exhausted_cluster"},
                "effective_min_sharpe": 0,
            }
            verdict = "rejected"
            record = save_result(hypothesis, {"error": "exhausted_cluster_pre_block"}, verdict, evaluation)
            knowledge["tested"].append(hypothesis["id"])
            knowledge["tested_combinations"].append({
                "strategy": hypothesis["strategy"],
                "symbol": hypothesis["symbol"],
                "params": hypothesis["params"]
            })
            knowledge["rejected"].append(record)
            update_family_aggregates(knowledge, record)
            save_knowledge(knowledge)
            continue

        # 3. バックテスト実行（インフラエラーの場合は1回だけリトライ）
        result = run_backtest(hypothesis)
        if "error" in result and result.get("error_type") == "infra":
            print(f"  🔄 インフラエラー → リトライ: {result['error'][:80]}")
            time.sleep(2)
            result = run_backtest(hypothesis)

        # 4. 評価
        verdict, evaluation = evaluate_result(hypothesis, result, knowledge)

        print(f"  📋 判定: {verdict.upper()}")
        if isinstance(evaluation, dict) and "reasons" in evaluation:
            for reason in evaluation["reasons"]:
                print(f"     {reason}")

        # 5. 蓄積
        record = save_result(hypothesis, result, verdict, evaluation)

        if verdict in ("error", "code_error"):
            knowledge.setdefault("errors", []).append(record)
            if len(knowledge["errors"]) > 100:
                knowledge["errors"] = knowledge["errors"][-100:]
        elif verdict == "adopted":
            record["cluster_id"] = evaluation.get("cluster_id", "unknown")
            knowledge["adopted"].append(record)
        else:
            _record_near_miss(hypothesis, evaluation, knowledge)
            knowledge["rejected"].append(record)

        if verdict not in ("error", "code_error"):
            update_family_aggregates(knowledge, record)
            knowledge["tested"].append(hypothesis["id"])
            knowledge["tested_combinations"].append({
                "strategy": hypothesis["strategy"],
                "symbol": hypothesis["symbol"],
                "params": hypothesis["params"]
            })

        save_knowledge(knowledge)

        # 少し待機（APIレート制限対策）
        if i < num_iterations - 1:
            time.sleep(2)
    
    # 最終レポート
    print_report(knowledge)

    # Machine-greppable error summary for cron reporters
    errors = knowledge.get("errors", [])
    n_infra = sum(1 for e in errors if e.get("evaluation", {}).get("error_type") == "infra")
    n_code = sum(1 for e in errors if e.get("evaluation", {}).get("error_type") == "code")
    print(f"ERRORS_SUMMARY infra={n_infra} code={n_code}")

    return knowledge


def run_revalidation(knowledge):
    """
    Re-run every currently adopted strategy through the full new pipeline
    (backtest + all gates). Demote failures to rejected with reason "revalidation_failed".
    """
    adopted = knowledge.get("adopted", [])
    if not adopted:
        print("📭 再検証対象の採用戦略がありません。")
        return knowledge

    print("=" * 60)
    print("🔁 再検証モード (Revalidation)")
    print(f"   対象: {len(adopted)}件の採用戦略")
    print(f"   開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 60)

    survivors = []
    demoted = 0

    for i, entry in enumerate(copy.deepcopy(adopted)):
        hyp = entry["hypothesis"]
        print(f"\n🔄 [{i+1}/{len(adopted)}] {hyp['description']}")
        print(f"   パラメータ: {hyp['params']}")

        # Re-run backtest
        result = run_backtest(hyp)
        if "error" in result:
            print(f"  ❌ バックテスト失敗: {result['error']}")
            demoted += 1
            error_type = result.get("error_type", "unknown")
            if error_type == "infra":
                entry["verdict"] = "error"
                entry["evaluation"] = {"verdict": "error", "error": result["error"],
                                       "error_type": "infra",
                                       "reasons": [f"revalidation failed (infra): {result['error']}"],
                                       "gate_results": {}}
                knowledge.setdefault("errors", []).append(entry)
            elif error_type == "code":
                entry["verdict"] = "code_error"
                entry["evaluation"] = {"verdict": "code_error", "error": result["error"],
                                       "error_type": "code",
                                       "reasons": [f"revalidation failed (code): {result['error']}"],
                                       "gate_results": {}}
                knowledge.setdefault("errors", []).append(entry)
            else:
                entry["verdict"] = "rejected"
                entry["evaluation"] = {"verdict": "rejected", "error": result["error"],
                                       "reasons": [f"revalidation_failed: {result['error']}"],
                                       "gate_results": {}}
                knowledge["rejected"].append(entry)
                update_family_aggregates(knowledge, entry)
            continue

        # Re-evaluate with current gates
        verdict, evaluation = evaluate_result(hyp, result, knowledge)
        print(f"  📋 再判定: {verdict.upper()}")
        if isinstance(evaluation, dict) and "reasons" in evaluation:
            for reason in evaluation["reasons"]:
                print(f"     {reason}")

        if verdict == "adopted":
            entry["backtest_result"] = result
            entry["evaluation"] = evaluation
            entry["verdict"] = verdict
            entry["tested_at"] = datetime.now().isoformat()
            entry["cluster_id"] = evaluation.get("cluster_id", entry.get("cluster_id", "unknown"))
            survivors.append(entry)
        else:
            entry["verdict"] = "rejected"
            entry["evaluation"] = evaluation
            entry["rejected_reason"] = "revalidation_failed"
            _record_near_miss(entry["hypothesis"], evaluation, knowledge)
            knowledge["rejected"].append(entry)
            demoted += 1
        update_family_aggregates(knowledge, entry)

    knowledge["adopted"] = survivors
    save_knowledge(knowledge)

    print(f"\n{'='*60}")
    print(f"📊 再検証完了")
    print(f"   Before: {len(adopted)}件採用")
    print(f"   After:  {len(survivors)}件採用 / {demoted}件降格")
    print(f"{'='*60}\n")

    return knowledge


if __name__ == "__main__":
    if "--revalidate" in sys.argv:
        knowledge = load_knowledge()
        run_revalidation(knowledge)
    else:
        iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        run_loop(iterations)