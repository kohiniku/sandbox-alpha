#!/usr/bin/env python3
"""
Autonomous Alpha Discovery Loop
エージェントが仮説生成→バックテスト→評価→蓄積を自律的に回す
"""
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
        # Migration: rebuild "families" from history if missing
        if "families" not in data:
            data["families"] = _rebuild_families_from_history(data)
            # persist the rebuilt families immediately so the migration is idempotent
            KNOWLEDGE_FILE.write_text(json.dumps(data, indent=2, default=str))
        return data
    return {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
            "superseded": [], "families": {}, "iterations": 0}


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
            adopted_params = adopted["hypothesis"]["params"]
            strategy_name = adopted["hypothesis"]["strategy"]
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


def run_backtest(hypothesis):
    """
    バックテスト実行フェーズ
    SANDBOX_RUNNER_URL が設定されている場合は信頼済みサンドボックスランナーにHTTPリクエストを送信。
    未設定の場合はサブプロセスで戦略を検証（従来の venv 実行パス）。
    """
    strategy = hypothesis["strategy"]
    symbol = hypothesis["symbol"]
    params = hypothesis["params"]
    runner_url = os.environ.get("SANDBOX_RUNNER_URL")

    if runner_url:
        return _run_backtest_sandbox(runner_url, strategy, symbol, params)
    else:
        return _run_backtest_subprocess(strategy, symbol, params)


def _run_backtest_sandbox(runner_url, strategy, symbol, params):
    """信頼済みサンドボックスランナー（Trusted Runner）経由でバックテストを実行"""
    url = f"{runner_url.rstrip('/')}/run"
    body = json.dumps({
        "strategy": strategy,
        "symbol": symbol,
        "params": params
    }).encode("utf-8")

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
            return {"error": f"Sandbox runner returned HTTP {status}: {response_body[:500]}"}

        return json.loads(response_body)

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        return {"error": f"Sandbox runner HTTP {e.code}: {error_body}"}
    except urllib.error.URLError as e:
        return {"error": f"Sandbox runner connection error: {e.reason}"}
    except json.JSONDecodeError as e:
        return {"error": f"Sandbox runner JSON parse error: {e}"}
    except Exception as e:
        return {"error": f"Sandbox runner error: {e}"}


def _run_backtest_subprocess(strategy, symbol, params):
    """サブプロセスでバックテストを実行（従来の venv 実行パス）"""
    cmd = [
        sys.executable,
        str(BASE_DIR / "backtests" / "backtest_engine.py"),
        "--strategy", strategy,
        "--symbol", symbol,
        "--params", json.dumps(params),
    ]

    print(f"  🔬 バックテスト実行中: {strategy} on {symbol}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return {"error": result.stderr}

        # stdoutからJSONブロックを抽出（最初の{から最後の}まで）
        stdout = result.stdout
        start_idx = stdout.find('{')
        end_idx = stdout.rfind('}')

        if start_idx >= 0 and end_idx > start_idx:
            json_str = stdout[start_idx:end_idx + 1]
            return json.loads(json_str)

        return {"error": "Could not parse output", "raw": stdout[:500]}

    except subprocess.TimeoutExpired:
        return {"error": "Timeout (120s)"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}


def evaluate_result(hypothesis, result, knowledge):
    """
    Multi-gate evaluation with overfitting countermeasures.
    Returns (verdict, evaluation_dict).
    """
    if "error" in result:
        return "rejected", {"verdict": "rejected", "error": result["error"], "reasons": [f"❌ Error: {result['error']}"]}

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
    holdout_pass = holdout_sharpe > 0 and holdout_return > 0
    gate_results["holdout"] = holdout_pass

    if holdout_sharpe > 0:
        reasons.append(f"✅ Holdout Sharpe {holdout_sharpe:.2f} > 0")
    else:
        reasons.append(f"❌ Holdout Sharpe {holdout_sharpe:.2f} <= 0")

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
    total = adopted + rejected
    
    print(f"\n{'='*60}")
    print(f"📊 Alpha Discovery Report (Iteration {knowledge['iterations']})")
    print(f"{'='*60}")
    print(f"  採用: {adopted}件 / テスト: {total}件 (採用率: {adopted/max(total,1)*100:.1f}%)")
    if superseded > 0:
        print(f"  上書き: {superseded}件 (superseded)")
    
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
        
        # 1. 仮説生成
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

        # 3. バックテスト実行
        result = run_backtest(hypothesis)
        
        # 4. 評価
        verdict, evaluation = evaluate_result(hypothesis, result, knowledge)
        
        print(f"  📋 判定: {verdict.upper()}")
        if isinstance(evaluation, dict) and "reasons" in evaluation:
            for reason in evaluation["reasons"]:
                print(f"     {reason}")
        
        # 5. 蓄積
        record = save_result(hypothesis, result, verdict, evaluation)
        
        if verdict == "adopted":
            record["cluster_id"] = evaluation.get("cluster_id", "unknown")
            knowledge["adopted"].append(record)
        else:
            knowledge["rejected"].append(record)
        
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