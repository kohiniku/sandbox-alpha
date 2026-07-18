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
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("/opt/data/sandbox-alpha")
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
    }
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
        return json.loads(KNOWLEDGE_FILE.read_text())
    return {"tested": [], "adopted": [], "rejected": [], "iterations": 0}


def save_knowledge(knowledge):
    """ナレッジベースを更新"""
    KNOWLEDGE_FILE.write_text(json.dumps(knowledge, indent=2, default=str))


def generate_hypothesis(knowledge):
    """
    仮説生成フェーズ
    ナレッジベースを参照して、まだ試していない戦略パラメータを生成
    """
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


def run_backtest(hypothesis):
    """
    バックテスト実行フェーズ
    使い捨てvenvプロセスで戦略を検証
    """
    strategy = hypothesis["strategy"]
    symbol = hypothesis["symbol"]
    params = hypothesis["params"]
    
    cmd = [
        "/opt/data/sandbox-alpha/.venv/bin/python3",
        str(BASE_DIR / "backtests" / "backtest_engine.py"),
        strategy, symbol
    ]
    
    # パラメータを追加
    for val in params.values():
        cmd.append(str(val))
    
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
    閾値に基づいて戦略の採否を判定
    """
    if "error" in result:
        return "rejected", result["error"]
    
    sharpe = result.get("sharpe_ratio", -999)
    total_return = result.get("total_return_pct", -999)
    max_dd = result.get("max_drawdown_pct", -999)
    
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
        "reasons": reasons
    }
    
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


def run_loop(num_iterations=3):
    """
    メインPDCAループ
    """
    print("=" * 60)
    print("🚀 Autonomous Alpha Discovery Loop 開始")
    print(f"   開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print(f"   イテレーション数: {num_iterations}")
    print("=" * 60)
    
    knowledge = load_knowledge()
    
    for i in range(num_iterations):
        knowledge["iterations"] += 1
        print(f"\n🔄 Iteration {i+1}/{num_iterations}")
        print("-" * 40)
        
        # 1. 仮説生成
        hypothesis = generate_hypothesis(knowledge)
        
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
