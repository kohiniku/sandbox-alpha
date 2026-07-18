#!/usr/bin/env python3
"""
Autonomous Alpha Discovery Loop
エージェントが仮説生成→バックテスト→評価→蓄積を自律的に回す
"""
import json
import os
import sys
import time
import subprocess
import random
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

# 採用閾値
THRESHOLDS = {
    "min_sharpe": 0.3,
    "min_return_pct": 5.0,
    "max_drawdown_pct": -25.0
}


def load_knowledge():
    """過去の戦略テスト結果を読み込む"""
    if KNOWLEDGE_FILE.exists():
        data = json.loads(KNOWLEDGE_FILE.read_text())
        # 後方互換: 古いナレッジファイルに不足キーを補完
        data.setdefault("tested_combinations", [])
        return data
    return {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [], "iterations": 0}


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


def evaluate_result(hypothesis, result):
    """
    評価フェーズ
    閾値に基づいて戦略の採否を判定。
    walk-forward有効時は out-of-sample 指標で判定する。
    """
    if "error" in result:
        return "rejected", result["error"]

    # walk-forward有効時はOOS指標を使用
    if result.get("walkforward", {}).get("enabled"):
        metrics = result.get("out_of_sample", {})
        is_metrics = result.get("in_sample", {})
    else:
        metrics = result
        is_metrics = None

    sharpe = metrics.get("sharpe_ratio", -999)
    total_return = metrics.get("total_return_pct", -999)
    max_dd = metrics.get("max_drawdown_pct", -999)
    num_trades = metrics.get("num_trades", 0)
    
    reasons = []
    
    if sharpe >= THRESHOLDS["min_sharpe"]:
        reasons.append(f"✅ Sharpe {sharpe:.2f} >= {THRESHOLDS['min_sharpe']}")
    else:
        reasons.append(f"❌ Sharpe {sharpe:.2f} < {THRESHOLDS['min_sharpe']}")
    
    if total_return >= THRESHOLDS["min_return_pct"]:
        reasons.append(f"✅ Return {total_return:.1f}% >= {THRESHOLDS['min_return_pct']}%")
    else:
        reasons.append(f"❌ Return {total_return:.1f}% < {THRESHOLDS['min_return_pct']}%")
    
    if max_dd >= THRESHOLDS["max_drawdown_pct"]:
        reasons.append(f"✅ Drawdown {max_dd:.1f}% >= {THRESHOLDS['max_drawdown_pct']}%")
    else:
        reasons.append(f"❌ Drawdown {max_dd:.1f}% < {THRESHOLDS['max_drawdown_pct']}%")
    
    # 全条件を満たせば採用
    is_adopted = (
        sharpe >= THRESHOLDS["min_sharpe"] and
        total_return >= THRESHOLDS["min_return_pct"] and
        max_dd >= THRESHOLDS["max_drawdown_pct"]
    )
    
    verdict = "adopted" if is_adopted else "rejected"
    evaluation = {
        "verdict": verdict,
        "sharpe_ratio": sharpe,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "num_trades": num_trades,
        "reasons": reasons,
    }
    if is_metrics:
        evaluation["in_sample"] = is_metrics
    
    return verdict, evaluation


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
    total = adopted + rejected
    
    print(f"\n{'='*60}")
    print(f"📊 Alpha Discovery Report (Iteration {knowledge['iterations']})")
    print(f"{'='*60}")
    print(f"  採用: {adopted}件 / テスト: {total}件 (採用率: {adopted/max(total,1)*100:.1f}%)")
    
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
        
        # 2. バックテスト実行
        result = run_backtest(hypothesis)
        
        # 3. 評価
        verdict, evaluation = evaluate_result(hypothesis, result)
        
        print(f"  📋 判定: {verdict.upper()}")
        if isinstance(evaluation, dict) and "reasons" in evaluation:
            for reason in evaluation["reasons"]:
                print(f"     {reason}")
        
        # 4. 蓄積
        record = save_result(hypothesis, result, verdict, evaluation)
        
        if verdict == "adopted":
            knowledge["adopted"].append(record)
        else:
            knowledge["rejected"].append(record)
        
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


if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_loop(iterations)
