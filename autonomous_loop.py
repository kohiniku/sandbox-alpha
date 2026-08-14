#!/usr/bin/env python3
"""
Autonomous Alpha Discovery Loop
エージェントが仮説生成→バックテスト→評価→蓄積を自律的に回す
"""
import base64
import fcntl
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

# Re-seed pool for the single-name search grid (issue #83): once killed-family
# coverage of the base grid (STRATEGY_TEMPLATES x SYMBOL_POOL) crosses
# RE_SEED_KILLED_COVERAGE, generate_hypothesis expands the symbol pool with
# these liquid tickers so the search space cannot be consumed to zero. Kept
# disjoint from SYMBOL_POOL. Symbols already present in the running universe
# (GLD/META/AMD appear in knowledge.json families) are included first.
EXTRA_SYMBOL_POOL = ["META", "AMD", "GLD", "NFLX", "XOM", "JPM", "WMT", "KO",
                     "DIS", "V", "INTC", "CSCO", "ORCL", "ADBE", "PEP", "BAC",
                     "UNH", "HD", "MCD", "NKE"]

# Fraction of the base grid that must be killed before re-seeding kicks in.
# Production hit 39/40 = 0.975 killed (issue #83).
RE_SEED_KILLED_COVERAGE = 0.9

# Per-run statistics accumulator for issue #83 (killed-skips, sources, tests).
# Reset at the top of run_loop; also incremented from _consume_backlog_entry,
# which run_loop calls. Module-level so the two functions share it without
# threading a stats object through every call site.
_RUN_STATS: dict = {}

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

from loop_constants import MISSING_METRIC, Verdict, BacklogStatus, FamilyLifecycle
from loop_constants import (BOOTSTRAP_ALPHA, BOOTSTRAP_N_RESAMPLE,
                            CV_FOLDS, EMBARGO_DAYS, BLOCK_LEN_MIN)
from backtests.bootstrap import BootstrapLCB


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


def _update_param_clusters(fam, params, templates):
    """Incrementally update param_clusters and distinct_clusters count.
    
    Uses greedy clustering: if the new params are within cluster distance
    of any existing cluster representative, don't add a new cluster.
    Otherwise, add this params as a new cluster representative.
    
    Args:
        fam: family dict to update (must have 'param_clusters' list field)
        params: new parameter dict to cluster
        templates: STRATEGY_TEMPLATES for cluster distance check
    """
    clusters = fam.setdefault("param_clusters", [])
    
    # Check if this params is within cluster distance of any existing representative
    for existing_params in clusters:
        if _params_within_cluster(params, existing_params, templates):
            # Within existing cluster, don't add new representative
            fam["distinct_clusters"] = len(clusters)
            return
    
    # New distinct cluster
    clusters.append(params)
    fam["distinct_clusters"] = len(clusters)


def _knowledge_lock_path():
    """Sidecar lock file for knowledge.json (issue #96).

    Deliberately a SEPARATE file that is never replaced: os.replace() swaps
    the knowledge.json inode, so flock()ing knowledge.json itself would leave
    concurrent lockers waiting on the old inode while a new one appears.
    """
    return KNOWLEDGE_FILE.with_name(KNOWLEDGE_FILE.name + ".lock")


def _atomic_write_knowledge(data):
    """Serialize + atomically replace knowledge.json under an exclusive lock.

    Guards against the three-way race described in issue #96: concurrent
    validation/review runs doing read-modify-write on the same 2.7MB file
    (truncated JSON from a mid-write kill, interleaved writers, and readers
    observing a half-written file). Mirrors backlog.py's flock discipline.
    """
    KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, default=str)
    lock_path = _knowledge_lock_path()
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            tmp_path = KNOWLEDGE_FILE.with_name(KNOWLEDGE_FILE.name + ".tmp")
            tmp_path.write_text(payload)
            os.replace(tmp_path, KNOWLEDGE_FILE)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def save_knowledge(knowledge):
    """ナレッジベースを更新 (issue #96: locked + atomic, no truncated/interleaved writes)"""
    _atomic_write_knowledge(knowledge)


def load_knowledge():
    """過去の戦略テスト結果を読み込む (issue #96: shared lock during read)"""
    if not KNOWLEDGE_FILE.exists():
        return {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [],
                "superseded": [], "families": {}, "iterations": 0, "errors": []}
    lock_path = _knowledge_lock_path()
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        try:
            data = json.loads(KNOWLEDGE_FILE.read_text())
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    # 後方互換: 古いナレッジファイルに不足キーを補完
    data.setdefault("tested_combinations", [])
    data.setdefault("superseded", [])
    data.setdefault("errors", [])
    # Migration: rebuild "families" from history if missing
    if "families" not in data:
        data["families"] = _rebuild_families_from_history(data)
        # persist the rebuilt families immediately so the migration is idempotent
        _atomic_write_knowledge(data)

    # Migration: stamp family_type on legacy family entries that lack it.
    for key, fam in data.get("families", {}).items():
        if "family_type" not in fam:
            # Derive from key shape; keys of the form "manifest:...|universe:..."
            # are cross, everything else is single.
            fam["family_type"] = "cross" if key.startswith("manifest:") else "single"

    # Migration: legacy knowledge only had a single near_misses list; introduce
    # near_misses_cross alongside without touching the existing data.
    data.setdefault("near_misses_cross", [])

    # Migration: add lifecycle fields to families that lack them.
    for fam in data.get("families", {}).values():
        if "lifecycle" not in fam:
            fam["lifecycle"] = FamilyLifecycle.CANDIDATE
            fam["refine_count"] = 0
            fam["kill_reason"] = ""
        
        # Migration: add param_clusters and distinct_clusters for issue #66 fix
        if "param_clusters" not in fam:
            fam["param_clusters"] = []
            fam["distinct_clusters"] = 0
            # Rebuild clusters from best_params if available
            if fam.get("best_params"):
                fam["param_clusters"].append(fam["best_params"])
                fam["distinct_clusters"] = 1

    return data


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
        key = _family_key(hyp.get("strategy", ""), hyp.get("symbol", ""), _derive_family_type(hyp))
        if not hyp.get("strategy") or not hyp.get("symbol"):
            continue
        families.setdefault(key, {
            "n_trials": 0,
            "best_val_sharpe": MISSING_METRIC,
            "best_params": {},
            "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                              "duplicate_cluster": 0, "exhausted_cluster": 0},
            "last_tried": "",
            "family_type": _derive_family_type(hyp),
            "lifecycle": FamilyLifecycle.CANDIDATE,
            "refine_count": 0,
            "kill_reason": "",
        })
        _apply_entry_to_family(families, key, entry, hyp, _derive_family_type(hyp))
    return families


def _family_key(strategy: str, symbol: str, family_type: str) -> str:
    if family_type not in ("single", "cross"):
        raise ValueError(f"family_type must be 'single' or 'cross', got {family_type!r}")
    return f"{strategy}|{symbol}"


def _derive_family_type(hyp: dict) -> str:
    """Classify a hypothesis into 'single' or 'cross'.

    'cross' if the strategy name uses the manifest namespace prefix
    ('manifest:'). Everything else is 'single'. Used to migrate old
    knowledge entries and to route new records into the correct
    family_type / near_miss bucket without threading the classification
    through every caller.
    """
    strategy = (hyp or {}).get("strategy", "") or ""
    return "cross" if strategy.startswith("manifest:") else "single"


def get_killed_families(knowledge) -> dict[str, str]:
    """Return family_key -> kill_reason for families with lifecycle == KILLED."""
    families = knowledge.get("families", {})
    return {
        key: fam.get("kill_reason", "")
        for key, fam in families.items()
        if fam.get("lifecycle") == FamilyLifecycle.KILLED
    }


def _grid_killed_coverage(knowledge) -> float:
    """Fraction of the base single-name grid (STRATEGY_TEMPLATES x SYMBOL_POOL)
    whose families are currently killed (issue #83: production reached 39/40 =
    0.975 and the random fallback degenerated into a KILLED_SKIP lottery)."""
    killed = get_killed_families(knowledge)
    grid = [(s, sym) for s in STRATEGY_TEMPLATES for sym in SYMBOL_POOL]
    if not grid:
        return 1.0
    n_killed = sum(1 for s, sym in grid if _family_key(s, sym, "single") in killed)
    return n_killed / len(grid)


def _get_active_symbol_pool(knowledge) -> list:
    """SYMBOL_POOL, expanded with EXTRA_SYMBOL_POOL once killed coverage of the
    base grid crosses RE_SEED_KILLED_COVERAGE, so the single-name search space
    cannot be consumed to zero."""
    if _grid_killed_coverage(knowledge) >= RE_SEED_KILLED_COVERAGE:
        return SYMBOL_POOL + [s for s in EXTRA_SYMBOL_POOL if s not in SYMBOL_POOL]
    return list(SYMBOL_POOL)


def _non_killed_grid(knowledge) -> list:
    """(strategy, symbol) pairs from the active symbol pool whose family is not
    killed. Empty when every family in the (possibly re-seeded) grid is dead."""
    killed = get_killed_families(knowledge)
    pool = _get_active_symbol_pool(knowledge)
    return [
        (s, sym)
        for s in STRATEGY_TEMPLATES
        for sym in pool
        if _family_key(s, sym, "single") not in killed
    ]


def _apply_entry_to_family(families, key, entry, hyp, family_type="single"):
    """Incrementally update a single family aggregate record from one evaluation entry."""
    fam = families.setdefault(key, {
        "n_trials": 0,
        "best_val_sharpe": MISSING_METRIC,
        "best_params": {},
        "gate_failures": {"validation": 0, "deflation": 0, "holdout": 0,
                          "duplicate_cluster": 0, "exhausted_cluster": 0},
        "last_tried": "",
        "family_type": family_type,
        "lifecycle": FamilyLifecycle.CANDIDATE,
        "refine_count": 0,
        "kill_reason": "",
        "param_clusters": [],
        "distinct_clusters": 0,
    })
    fam["n_trials"] += 1
    
    # Track parameter clusters for deflation calculation
    params = hyp.get("params", {})
    if params:
        _update_param_clusters(fam, params, STRATEGY_TEMPLATES)
    
    ev = entry.get("evaluation", {})
    val_sharpe = ev.get("sharpe_ratio", MISSING_METRIC)
    if val_sharpe > fam["best_val_sharpe"]:
        fam["best_val_sharpe"] = val_sharpe
        fam["best_params"] = params
    tested_at = entry.get("tested_at", "")
    if tested_at > fam["last_tried"]:
        fam["last_tried"] = tested_at
    # Tally gate failures
    gate_results = ev.get("gate_results", {})
    if not gate_results:
        return
    # Determine why it was rejected (or if it passed everything)
    verdict = ev.get("verdict", entry.get("verdict", ""))
    if verdict == Verdict.ADOPTED:
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
    key = _family_key(hyp.get("strategy", ""), hyp.get("symbol", ""), _derive_family_type(hyp))
    if not hyp.get("strategy") or not hyp.get("symbol"):
        return
    families = knowledge.setdefault("families", {})
    _apply_entry_to_family(families, key, record, hyp, _derive_family_type(hyp))


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
        (e.get("evaluation", {}).get("sharpe_ratio", MISSING_METRIC) for e in matching),
        default=-999
    )
    if best_sharpe >= 0:
        return False, len(matching), best_sharpe

    return True, len(matching), best_sharpe


def generate_hypothesis(knowledge):
    """
    仮説生成フェーズ
    ナレッジベースを参照して、まだ試していない戦略パラメータを生成。
    既にテスト済みの(strategy, symbol, params)の組み合わせはスキップ。
    30%の確率でadopted戦略のパラメータ近傍を探索する。
    Killed family は決して選択しない (issue #83: グリッドの 97.5% が kill 済みで
    ランダム生成が KILLED_SKIP の宝くじと化していた)。
    """
    max_attempts = 50  # 重複回避の試行上限
    killed = get_killed_families(knowledge)
    candidates = _non_killed_grid(knowledge)

    if not candidates:
        # 再シード後も全ファミリーが kill 済み — 下のフォールバックに落ちる。
        # ループ側の KILLED_SKIP ガードと run_stats の ZERO_TESTS_ALERT が
        # この状態を可視化する。
        print("  ⚠️ 非killファミリー枯渇（再シード後も全滅）→ 重複許容フォールバック")

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
            # Skip adopted entries whose family has been killed (issue #83)
            if _family_key(strategy_name, symbol, "single") in killed:
                continue
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
        elif candidates:
            strategy_name, symbol = random.choice(candidates)
            template = STRATEGY_TEMPLATES[strategy_name]

            # パラメータをランダムサンプリング
            params = {}
            for param_name, param_space in template["param_space"].items():
                if isinstance(param_space, range):
                    params[param_name] = random.choice(list(param_space))
                else:
                    params[param_name] = random.choice(param_space)
        else:
            # 非kill候補なし — フォールバックへ
            break

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
    if candidates:
        # 非kill候補が残っている場合はそちらを優先（kill済みファミリーを避ける）
        strategy_name, symbol = random.choice(candidates)
    else:
        strategy_name = random.choice(list(STRATEGY_TEMPLATES.keys()))
        # 再シード後のプールを使う（issue #83）
        symbol = random.choice(_get_active_symbol_pool(knowledge))
    template = STRATEGY_TEMPLATES[strategy_name]
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
    runner_url = _get_loop_config()["runner_url"]

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
    # Gate v2: conditionally include CV params when SANDBOX_GATE_V2=1
    if os.environ.get("SANDBOX_GATE_V2", "0") == "1":
        from loop_constants import CV_FOLDS, EMBARGO_DAYS
        payload["cv_folds"] = CV_FOLDS
        payload["embargo_days"] = EMBARGO_DAYS
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
            return {"error": f"Sandbox runner returned HTTP {status}: {response_body[:500]}", "error_type": _classify_http_status(status)}

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
        return {"error": f"Sandbox runner HTTP {e.code}: {error_body}", "error_type": _classify_http_status(e.code)}
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


# ---------------------------------------------------------------------------
# Shared evaluation gate helpers (used by evaluate_result + _evaluate_manifest_result)
# ---------------------------------------------------------------------------


def _triage_eval_error(result):
    """If result carries an error, return (verdict, evaluation); else None."""
    if "error" not in result:
        return None
    error_type = result.get("error_type", "error")  # safe default: unknown = error
    if error_type == "infra":
        return Verdict.ERROR, {"verdict": Verdict.ERROR, "error": result["error"],
                         "error_type": "infra",
                         "reasons": [f"🔧 Infra error (not counted as rejection): {result['error']}"]}
    elif error_type == "code":
        return Verdict.CODE_ERROR, {"verdict": Verdict.CODE_ERROR, "error": result["error"],
                              "error_type": "code",
                              "reasons": [f"💻 Code error (strategy code crashed): {result['error']}"]}
    else:
        # Unknown error_type → safe default: error (never rejection)
        return Verdict.ERROR, {"verdict": Verdict.ERROR, "error": result["error"],
                         "error_type": "unknown",
                         "reasons": [f"🔧 Unknown error (not counted as rejection): {result['error']}"]}


def _eval_val_gate(val_sharpe, val_return, val_max_dd, effective_min_sharpe,
                   N_family, T_val):
    """Validation gate: returns (passed: bool, reasons: list[str])."""
    passed = (
        val_sharpe >= effective_min_sharpe
        and val_return > 0
        and val_max_dd >= MAX_DRAWDOWN_LIMIT
    )
    reasons = []
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
    return passed, reasons


def _eval_holdout_gate(holdout_sharpe, holdout_return, val_sharpe):
    """Holdout confirmation gate: returns (passed: bool, reasons: list[str]).

    Stricter gate: holdout Sharpe must reach min(0.5, 0.5 * val_sharpe).
    Floor at 0.5 absolute; scaled down for modest val Sharpe so the bar
    is never harsher than half the validation performance.
    """
    holdout_threshold = min(0.5, 0.5 * val_sharpe)
    passed = (holdout_sharpe >= holdout_threshold) and (holdout_return > 0)
    reasons = []
    if holdout_sharpe >= holdout_threshold:
        reasons.append(f"✅ Holdout Sharpe {holdout_sharpe:.2f} >= {holdout_threshold:.2f} (threshold=min(0.5, 0.5*val))")
    else:
        reasons.append(f"❌ Holdout Sharpe {holdout_sharpe:.2f} < {holdout_threshold:.2f} (threshold=min(0.5, 0.5*val))")
    if holdout_return > 0:
        reasons.append(f"✅ Holdout Return {holdout_return:.1f}% > 0")
    else:
        reasons.append(f"❌ Holdout Return {holdout_return:.1f}% <= 0")
    return passed, reasons


def _eval_val_gate_cv(
    per_fold_returns: list,
    N_family: int,
    *,
    alpha: float = BOOTSTRAP_ALPHA,
    n_resample: int = BOOTSTRAP_N_RESAMPLE,
    block_len: int | None = None,
):
    """Gate v2 validation.  Returns (passed: bool, lcb_sharpe: float, reasons: list[str]).

    Computes the bootstrap LCB Sharpe on concatenated fold returns, then
    applies the same deflation + return + drawdown checks as the v1 gate.
    """
    import pandas as pd  # local import — autonomous_loop primarily uses stdlib

    concat = pd.concat(per_fold_returns) if len(per_fold_returns) > 1 else per_fold_returns[0]
    T_val_cv = sum(len(r) for r in per_fold_returns)

    lcb_sharpe = BootstrapLCB.compute(concat, block_len=block_len,
                                      n_resample=n_resample, alpha=alpha)

    val_return_pct = ((1 + concat).prod() - 1) * 100

    cumulative = (1 + concat).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    val_max_dd_pct = float(drawdowns.min() * 100)

    effective_threshold = compute_effective_min_sharpe(N_family, T_val_cv)

    passed = (
        lcb_sharpe >= effective_threshold
        and val_return_pct > 0
        and val_max_dd_pct >= MAX_DRAWDOWN_LIMIT
    )
    reasons = []
    if lcb_sharpe >= effective_threshold:
        reasons.append(
            f"✅ LCB Sharpe {lcb_sharpe:.2f} >= {effective_threshold:.2f}"
            f" (deflated, N={N_family}, T_cv={T_val_cv})"
        )
    else:
        reasons.append(
            f"❌ LCB Sharpe {lcb_sharpe:.2f} < {effective_threshold:.2f}"
            f" (deflated, N={N_family}, T_cv={T_val_cv})"
        )
    if val_return_pct > 0:
        reasons.append(f"✅ Val Return {val_return_pct:.1f}% > 0%")
    else:
        reasons.append(f"❌ Val Return {val_return_pct:.1f}% <= 0%")
    if val_max_dd_pct >= MAX_DRAWDOWN_LIMIT:
        reasons.append(f"✅ Val Drawdown {val_max_dd_pct:.1f}% >= {MAX_DRAWDOWN_LIMIT}%")
    else:
        reasons.append(f"❌ Val Drawdown {val_max_dd_pct:.1f}% < {MAX_DRAWDOWN_LIMIT}%")
    return passed, lcb_sharpe, reasons


def _eval_holdout_gate_cv(
    holdout_returns,
    lcb_val_sharpe: float,
    *,
    alpha: float = BOOTSTRAP_ALPHA,
    n_resample: int = BOOTSTRAP_N_RESAMPLE,
    block_len: int | None = None,
):
    """Gate v2 holdout.  Returns (passed: bool, lcb_holdout_sharpe: float, reasons: list[str]).

    Computes bootstrap LCB on the holdout series and applies a
    threshold that is half the validation LCB, capped at 0.5.
    """
    lcb_holdout_sharpe = BootstrapLCB.compute(holdout_returns, block_len=block_len,
                                              n_resample=n_resample, alpha=alpha)
    holdout_return_pct = ((1 + holdout_returns).prod() - 1) * 100
    threshold = min(0.5, 0.5 * lcb_val_sharpe)

    passed = (lcb_holdout_sharpe >= threshold) and (holdout_return_pct > 0)
    reasons = []
    if lcb_holdout_sharpe >= threshold:
        reasons.append(
            f"✅ Holdout Sharpe {lcb_holdout_sharpe:.2f} >= {threshold:.2f}"
            f" (threshold=min(0.5, 0.5*val))"
        )
    else:
        reasons.append(
            f"❌ Holdout Sharpe {lcb_holdout_sharpe:.2f} < {threshold:.2f}"
            f" (threshold=min(0.5, 0.5*val))"
        )
    if holdout_return_pct > 0:
        reasons.append(f"✅ Holdout Return {holdout_return_pct:.1f}% > 0")
    else:
        reasons.append(f"❌ Holdout Return {holdout_return_pct:.1f}% <= 0")
    return passed, lcb_holdout_sharpe, reasons


def _print_gate_v2_diff(strategy, symbol, v1_verdict, evaluation, gate_v2):
    """Emit a machine-greppable GATE_V2_DIFF log line."""
    v2_verdict = gate_v2["verdict"]
    agree = "1" if v1_verdict == v2_verdict else "0"
    v1_val = evaluation.get("sharpe_ratio", MISSING_METRIC)
    v2_lcb = gate_v2["val_lcb_sharpe"]
    v1_ho = evaluation.get("holdout_sharpe", MISSING_METRIC)
    v2_lcb_h = gate_v2.get("holdout_lcb_sharpe", MISSING_METRIC)
    print(f"GATE_V2_DIFF {strategy}/{symbol} v1={v1_verdict} v2={v2_verdict} "
          f"v1_val_sharpe={v1_val:.2f} v2_lcb={v2_lcb:.2f} "
          f"v1_holdout_sharpe={v1_ho:.2f} v2_lcb_h={v2_lcb_h} agree={agree}")


# ---------------------------------------------------------------------------
# Public evaluators
# ---------------------------------------------------------------------------


def _compute_gate_v2_cv(result, N_family):
    """Compute gate v2 verdict from cv block (passive/observational only).

    Returns a gate_v2 dict or None if the flag is off or cv block is absent.
    """
    if os.environ.get("SANDBOX_GATE_V2", "0") != "1":
        return None
    cv = result.get("cv")
    if not cv or "folds" not in cv:
        return None
    try:
        import pandas as pd
        per_fold_returns = [
            pd.Series(f["val_daily_returns"], index=pd.to_datetime(f["val_dates"]))
            for f in cv["folds"]
        ]
        holdout_returns = pd.Series(
            cv["holdout"]["daily_returns"],
            index=pd.to_datetime(cv["holdout"]["dates"])
        )
        v2_val_passed, v2_lcb_val, v2_val_reasons = _eval_val_gate_cv(per_fold_returns, N_family)
        if v2_val_passed:
            v2_holdout_passed, v2_lcb_holdout, v2_holdout_reasons = _eval_holdout_gate_cv(holdout_returns, v2_lcb_val)
        else:
            v2_holdout_passed, v2_lcb_holdout, v2_holdout_reasons = None, None, None
        v2_verdict = "adopted" if (v2_val_passed and v2_holdout_passed) else "rejected"
        return {
            "val_passed": v2_val_passed,
            "val_lcb_sharpe": round(v2_lcb_val, 4),
            "val_reasons": v2_val_reasons,
            "holdout_passed": v2_holdout_passed,
            "holdout_lcb_sharpe": round(v2_lcb_holdout, 4) if v2_lcb_holdout is not None else None,
            "holdout_reasons": v2_holdout_reasons,
            "verdict": v2_verdict,
        }
    except Exception:
        return None


def evaluate_result(hypothesis, result, knowledge):
    """
    Multi-gate evaluation with overfitting countermeasures.
    Returns (verdict, evaluation_dict).
    """
    # Degenerate metrics: numeric but non-finite (NaN/inf) from zero-variance
    # returns caused by no trades in a segment.  This is an honest "no edge"
    # outcome, not a code error.  Short-circuit to rejected so it counts as
    # a normal tested/rejected result rather than an error.
    if result.get("degenerate"):
        degenerate_reason = result.get("degenerate_reason", "unknown")
        strategy = hypothesis.get("strategy", "?")
        symbol = hypothesis.get("symbol", "?")
        print(f"DEGENERATE {strategy}/{symbol} reason={degenerate_reason}")
        return "rejected", {
            "verdict": Verdict.REJECTED,
            "reasons": [f"degenerate: {degenerate_reason}"],
        }

    triage = _triage_eval_error(result)
    if triage is not None:
        return triage

    wf = result.get("walkforward", {})
    if not wf.get("enabled"):
        return Verdict.REJECTED, {"verdict": Verdict.REJECTED, "error": "walkforward disabled", "reasons": ["❌ Walkforward not enabled"]}

    val_metrics = result.get("out_of_sample", {})
    holdout_metrics = result.get("holdout", {})
    is_metrics = result.get("in_sample", {})

    val_sharpe = val_metrics.get("sharpe_ratio", MISSING_METRIC)
    val_return = val_metrics.get("total_return_pct", MISSING_METRIC)
    val_max_dd = val_metrics.get("max_drawdown_pct", MISSING_METRIC)
    T_val = val_metrics.get("num_days", 252)

    strategy = hypothesis["strategy"]
    symbol = hypothesis["symbol"]

    # Count N_family: use distinct parameter clusters instead of raw trial count
    # This prevents the review loop from inflating the deflation bar when refining
    # parameters within the same cluster (issue #66)
    fam_key = _family_key(strategy, symbol, _derive_family_type(hypothesis))
    families = knowledge.get("families", {})
    N_family = families.get(fam_key, {}).get("distinct_clusters", 
                                              families.get(fam_key, {}).get("n_trials", 0))
    effective_min_sharpe = compute_effective_min_sharpe(N_family, T_val)

    # --- Gate v2: passive/observational CV evaluation (only when flag on + cv block present) ---
    gate_v2 = _compute_gate_v2_cv(result, N_family)

    # --- Gate (a): Validation gate ---
    val_pass, reasons = _eval_val_gate(val_sharpe, val_return, val_max_dd,
                                        effective_min_sharpe, N_family, T_val)
    gate_results = {"validation": val_pass}

    if not val_pass:
        evaluation = {
            "verdict": Verdict.REJECTED,
            "sharpe_ratio": val_sharpe,
            "total_return_pct": val_return,
            "max_drawdown_pct": val_max_dd,
            "reasons": reasons,
            "gate_results": gate_results,
            "effective_min_sharpe": round(effective_min_sharpe, 4),
        }
        if is_metrics:
            evaluation["in_sample"] = is_metrics
        if gate_v2:
            evaluation["gate_v2"] = gate_v2
            _print_gate_v2_diff(strategy, symbol, Verdict.REJECTED, evaluation, gate_v2)
        return "rejected", evaluation

    # --- Gate (b): Deflation gate (embedded in (a) via effective_min_sharpe) ---
    reasons.append(f"📐 Deflated threshold: {effective_min_sharpe:.2f} (base={MIN_SHARPE_BASE}, N_family={N_family}, T_val={T_val})")
    gate_results["deflation"] = True

    # --- Gate (c): Holdout confirmation ---
    holdout_sharpe = holdout_metrics.get("sharpe_ratio", MISSING_METRIC)
    holdout_return = holdout_metrics.get("total_return_pct", MISSING_METRIC)
    holdout_pass, holdout_reasons = _eval_holdout_gate(holdout_sharpe, holdout_return, val_sharpe)
    reasons.extend(holdout_reasons)
    gate_results["holdout"] = holdout_pass

    if not holdout_pass:
        evaluation = {
            "verdict": Verdict.REJECTED,
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
        if gate_v2:
            evaluation["gate_v2"] = gate_v2
            _print_gate_v2_diff(strategy, symbol, Verdict.REJECTED, evaluation, gate_v2)
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
        inc_holdout = incumbent.get("evaluation", {}).get("holdout_sharpe", MISSING_METRIC)
        if holdout_sharpe > inc_holdout:
            reasons.append(f"🔄 Cluster replace: holdout Sharpe {holdout_sharpe:.2f} > incumbent {inc_holdout:.2f}")
            knowledge.setdefault("superseded", []).append(incumbent)
            adopted.pop(incumbent_idx)
            gate_results["cluster"] = "replaced"
        else:
            reasons.append(f"❌ Cluster dedup: holdout Sharpe {holdout_sharpe:.2f} <= incumbent {inc_holdout:.2f}")
            gate_results["cluster"] = "duplicate_cluster"
            evaluation = {
                "verdict": Verdict.REJECTED,
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
            if gate_v2:
                evaluation["gate_v2"] = gate_v2
                _print_gate_v2_diff(strategy, symbol, Verdict.REJECTED, evaluation, gate_v2)
            return "rejected", evaluation
    else:
        gate_results["cluster"] = "new"

    # All gates passed
    reasons.insert(0, "🏆 ALL GATES PASSED")
    evaluation = {
        "verdict": Verdict.ADOPTED,
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
    if gate_v2:
        evaluation["gate_v2"] = gate_v2
        _print_gate_v2_diff(strategy, symbol, Verdict.ADOPTED, evaluation, gate_v2)

    return Verdict.ADOPTED, evaluation


# ---------------------------------------------------------------------------
# Near-miss classification
# ---------------------------------------------------------------------------

def _classify_near_miss(hypothesis, evaluation):
    """Classify a rejected evaluation as near-miss if conditions met.

    Returns a near-miss dict or None.
    """
    gate_results = evaluation.get("gate_results", {})
    val_sharpe = evaluation.get("sharpe_ratio", MISSING_METRIC)
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
    ft = _derive_family_type(hypothesis)
    list_key = "near_misses_cross" if ft == "cross" else "near_misses"
    near_misses = knowledge.setdefault(list_key, [])
    near_misses.append(nm)
    # Cap at 30 most recent (per list — cross and single are independent)
    if len(near_misses) > 30:
        knowledge[list_key] = near_misses[-30:]


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
    # Degenerate metrics: numeric but non-finite (NaN/inf) from zero-variance
    # returns caused by no trades in a segment.  Short-circuit to rejected.
    if runner_result.get("degenerate"):
        degenerate_reason = runner_result.get("degenerate_reason", "unknown")
        strategy = hypothesis.get("strategy", "?")
        symbol = hypothesis.get("symbol", "?")
        print(f"DEGENERATE {strategy}/{symbol} reason={degenerate_reason}")
        return "rejected", {
            "verdict": Verdict.REJECTED,
            "reasons": [f"degenerate: {degenerate_reason}"],
        }

    # Note: runner_result is the FULL parsed runner response (includes metrics + n_days)
    if "metrics" not in runner_result:
        return "error", {
            "verdict": Verdict.ERROR,
            "error": "Malformed runner response: missing 'metrics'",
            "error_type": "infra",
            "reasons": [f"🔧 Infra error (not counted as rejection): missing 'metrics' in runner response"],
        }

    metrics = runner_result["metrics"]
    val_sharpe = metrics.get("val_sharpe", MISSING_METRIC)
    val_max_dd = metrics.get("val_max_drawdown_pct", MISSING_METRIC)
    val_return = metrics.get("val_total_return_pct", MISSING_METRIC)
    holdout_sharpe = metrics.get("holdout_sharpe", MISSING_METRIC)
    holdout_return = metrics.get("holdout_total_return_pct", MISSING_METRIC)
    T_val = runner_result.get("n_days", 252)

    strategy = hypothesis["strategy"]
    symbol = hypothesis["symbol"]

    # N_family: count from families dict (updated by update_family_aggregates)
    # Use distinct_clusters to avoid inflating the bar for correlated refinements (issue #66)
    fam_key = _family_key(strategy, symbol, "cross")
    families = knowledge.get("families", {})
    N_family = families.get(fam_key, {}).get("distinct_clusters",
                                              families.get(fam_key, {}).get("n_trials", 0))
    effective_min_sharpe = compute_effective_min_sharpe(N_family, T_val)

    # --- Gate (a): Validation gate ---
    val_pass, reasons = _eval_val_gate(val_sharpe, val_return, val_max_dd,
                                        effective_min_sharpe, N_family, T_val)
    gate_results = {"validation": val_pass}

    if not val_pass:
        evaluation = {
            "verdict": Verdict.REJECTED,
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
    holdout_pass, holdout_reasons = _eval_holdout_gate(holdout_sharpe, holdout_return, val_sharpe)
    reasons.extend(holdout_reasons)
    gate_results["holdout"] = holdout_pass

    if not holdout_pass:
        evaluation = {
            "verdict": Verdict.REJECTED,
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
        "verdict": Verdict.ADOPTED,
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

    return Verdict.ADOPTED, evaluation


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

        actual = metrics.get(metric, MISSING_METRIC)
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


# ---------------------------------------------------------------------------
# Backlog consumption helpers
# ---------------------------------------------------------------------------


def _extract_universe_meta(manifest_spec):
    """Extract universe list + hash + size from a manifest spec.

    Returns (universe: list, universe_hash: str, universe_size: int).
    Deduplicated helper — the universe-extraction loop appeared twice.
    """
    universe = []
    for ds in manifest_spec.get("data_sources", []):
        if ds.get("type") == "ohlcv" and ds.get("universe"):
            universe = ds["universe"]
            break
    sorted_universe = sorted(universe)
    universe_hash = hashlib.sha256(
        json.dumps(sorted_universe, sort_keys=True).encode()
    ).hexdigest()[:8]
    return universe, universe_hash, len(universe)


def _classify_http_status(status):
    """Return 'code' for 4xx client errors, 'infra' for everything else non-200.

    4xx errors (400, 404, 422, …) are deterministic client/contract errors —
    retrying them will never succeed.  5xx, network, and timeout errors are
    transient infrastructure problems that may resolve on retry.
    """
    if 400 <= status < 500:
        return "code"
    return "infra"


def _classify_runner_response(status, response_body):
    """Parse and classify a sandbox runner HTTP response.

    Table-driven mapping of runner status / error_type → infra | code.

    Returns (result: dict, is_ok: bool).
    If is_ok is True, result is the parsed runner payload.
    If is_ok is False, result is an error dict with 'error' and 'error_type'.
    """
    if status != 200:
        return {"error": f"Sandbox runner returned HTTP {status}: {response_body[:500]}",
                "error_type": _classify_http_status(status)}, False

    parsed = json.loads(response_body)
    if not isinstance(parsed, dict):
        return {"error": "Malformed runner response (not a dict)",
                "error_type": "infra"}, False

    if parsed.get("status") == "error":
        # Table: runner error_type -> our classification
        # 'manifest'/'code' → agent's fault (code_error, do not retry as infra)
        # 'infra' or unknown → infra (retry-eligible)
        err_type = parsed.get("error_type", "")
        if err_type in ("manifest", "code"):
            return {"error": parsed.get("error", f"{err_type} error"),
                    "error_type": "code"}, False
        else:
            return {"error": parsed.get("error", "runner reported error"),
                    "error_type": "infra"}, False

    if parsed.get("status") == "ok":
        return parsed, True

    return {"error": f"Malformed runner response: unknown status '{parsed.get('status', 'missing')}'",
            "error_type": "infra"}, False


def _http_post_for_result(url, body, timeout):
    """HTTP POST to sandbox runner, returning (result: dict, is_ok: bool).

    Catches all HTTP/network/parse errors → returns (error_dict, False).
    On success returns (parsed_dict, True).
    """
    try:
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read().decode("utf-8")
            status = resp.status
        return _classify_runner_response(status, response_body)
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        return {"error": f"Sandbox runner HTTP {e.code}: {error_body}",
                "error_type": _classify_http_status(e.code)}, False
    except urllib.error.URLError as e:
        return {"error": f"Sandbox runner connection error: {e.reason}",
                "error_type": "infra"}, False
    except json.JSONDecodeError as e:
        return {"error": f"Sandbox runner JSON parse error: {e}",
                "error_type": "infra"}, False
    except Exception as e:
        return {"error": f"Sandbox runner error: {e}",
                "error_type": "infra"}, False


def _consume_param_entry(entry, spec):
    """Handle a param-type backlog entry. Returns (hypothesis, result)."""
    hypothesis = {
        "id": f"bl_{entry['id'][:12]}",
        "strategy": spec["strategy"],
        "symbol": spec["symbol"],
        "params": spec["params"],
        "description": f"Backlog param: {spec['strategy']} on {spec['symbol']}",
        "generated_at": datetime.now().isoformat(),
    }
    result = run_backtest(hypothesis)
    return hypothesis, result


def _consume_manifest_entry(entry, spec, runner_url, bl):
    """Handle a manifest-type backlog entry.

    Returns (hypothesis, result) or None if entry cannot run.
    """
    if not runner_url:
        print(f"  ⚠️ SANDBOX_RUNNER_URL未設定 — マニフェストエントリをpendingに戻します")
        bl.mark(entry["id"], BacklogStatus.PENDING)
        return None

    manifest_spec = spec
    manifest_name = manifest_spec.get("name", "unknown")
    universe, universe_hash, universe_size = _extract_universe_meta(manifest_spec)
    execution_mode = manifest_spec.get("execution_mode", "structured")

    hypothesis = {
        "id": f"bl_{entry['id'][:12]}",
        "strategy": f"manifest:{manifest_name}",
        "symbol": f"universe:{universe_hash}",
        "params": {
            "universe_size": universe_size,
            "execution_mode": execution_mode,
            "primary_metric": "sharpe",
        },
        "description": f"Backlog manifest: {manifest_name} on {universe_size} symbols [{execution_mode}]",
        "generated_at": datetime.now().isoformat(),
    }

    url = f"{runner_url.rstrip('/')}/run_manifest"
    body = json.dumps(manifest_spec).encode("utf-8")

    print(f"  🔬 マニフェスト実行中 (sandbox): {manifest_name} on {universe_size} symbols...")
    result, _ = _http_post_for_result(url, body, 300)
    return hypothesis, result


def _consume_code_entry(entry, spec, knowledge, runner_url, bl):
    """Handle a code-type backlog entry.

    Returns (hypothesis, result) or None if entry cannot run or is duplicate.
    """
    if not runner_url:
        print(f"  ⚠️ SANDBOX_RUNNER_URL未設定 — コードエントリをpendingに戻します")
        bl.mark(entry["id"], BacklogStatus.PENDING)
        return None

    code_str = spec["code"]
    code_b64 = base64.b64encode(code_str.encode("utf-8")).decode("ascii")
    code_hash = hashlib.sha256(code_str.encode("utf-8")).hexdigest()

    # Check duplicate: same code_hash + symbol in tested history
    for rec in knowledge.get("adopted", []) + knowledge.get("rejected", []):
        ev = rec.get("evaluation", {})
        hyp = rec.get("hypothesis", {})
        if ev.get("code_hash") == code_hash and hyp.get("symbol") == spec["symbol"]:
            print(f"  ⚠️ 重複コードハッシュ (code_hash={code_hash[:12]}..., symbol={spec['symbol']}) → スキップ")
            bl.mark(entry["id"], BacklogStatus.DONE_REJECTED, {
                "verdict": Verdict.REJECTED,
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
            url, data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            response_body = resp.read().decode("utf-8")
            status = resp.status
        if status != 200:
            result = {"error": f"Sandbox runner returned HTTP {status}: {response_body[:500]}",
                      "error_type": _classify_http_status(status)}
        else:
            result = json.loads(response_body)
            if isinstance(result, dict) and "error" in result:
                result.setdefault("error_type", "code")
            if "code_hash" not in result:
                result["code_hash"] = code_hash
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        result = {"error": f"Sandbox runner HTTP {e.code}: {error_body}", "error_type": _classify_http_status(e.code)}
    except urllib.error.URLError as e:
        result = {"error": f"Sandbox runner connection error: {e.reason}", "error_type": "infra"}
    except json.JSONDecodeError as e:
        result = {"error": f"Sandbox runner JSON parse error: {e}", "error_type": "infra"}
    except Exception as e:
        result = {"error": f"Sandbox runner error: {e}", "error_type": "infra"}

    return hypothesis, result


def _consume_backlog_entry(knowledge):
    """Try to consume one pending backlog entry.

    Returns (entry, hypothesis, result, verdict, evaluation) if consumed,
    or None if backlog is empty or entry cannot run.
    """
    from backlog import Backlog

    backlog_path = _get_loop_config()["backlog_path"]
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
        _, _, universe_size = _extract_universe_meta(spec)
        tag = f"[manifest] {spec.get('name', '?')} on {universe_size} symbols"
    else:
        tag = f"[code] {spec.get('name', '?')} on {spec['symbol']}"
    print(f"📥 バックログ消化: {tag} (priority {priority}, source {source_info.get('kind', '?')}:{source_info.get('ref', '?')})")

    # Mark as testing
    bl.mark(entry["id"], BacklogStatus.TESTING)

    # KILLED family guard (defense in depth — ideation-side filter is primary)
    if etype == "param":
        bk_family_key = _family_key(spec["strategy"], spec["symbol"], "single")
    elif etype == "manifest":
        manifest_name = spec.get("name", "unknown")
        _, universe_hash, _ = _extract_universe_meta(spec)
        bk_family_key = _family_key(f"manifest:{manifest_name}", f"universe:{universe_hash}", "cross")
    elif etype == "code":
        bk_family_key = _family_key("codegen", spec.get("symbol", "?"), "single")
    else:
        bk_family_key = None

    if bk_family_key and bk_family_key in get_killed_families(knowledge):
        print(f"KILLED_SKIP {bk_family_key}")
        # Issue #83: count toward the run's killed_skips so a run that only
        # skips is loud (RUN_SUMMARY / ZERO_TESTS_ALERT) instead of a silent no-op.
        _RUN_STATS["killed_skips"] = _RUN_STATS.get("killed_skips", 0) + 1
        bl.mark(entry["id"], BacklogStatus.DONE_REJECTED,
                result={"reason": "family_killed"})
        return None

    runner_url = _get_loop_config()["runner_url"]

    # ── Route to per-type handler ──
    if etype == "param":
        consumed = _consume_param_entry(entry, spec)
    elif etype == "manifest":
        consumed = _consume_manifest_entry(entry, spec, runner_url, bl)
    elif etype == "code":
        consumed = _consume_code_entry(entry, spec, knowledge, runner_url, bl)
    else:
        bl.mark(entry["id"], BacklogStatus.PENDING)
        return None

    if consumed is None:
        return None

    hypothesis, result = consumed

    # ── Evaluate ──
    if etype == "manifest" and isinstance(result, dict) and result.get("status") == "ok":
        verdict, evaluation = _evaluate_manifest_result(result, hypothesis, knowledge)
    else:
        verdict, evaluation = evaluate_result(hypothesis, result, knowledge)

    # ── Extra criteria ──
    if verdict == Verdict.ADOPTED and extra_criteria:
        metrics = result.get("out_of_sample", result)
        extra_pass, extra_failures = _check_extra_criteria(metrics, extra_criteria)
        if not extra_pass:
            evaluation["reasons"].extend(extra_failures)
            verdict = "rejected"
            evaluation["verdict"] = Verdict.REJECTED

    # ── Summary line ──
    if verdict == Verdict.ADOPTED:
        val_sharpe = evaluation.get("sharpe_ratio", 0)
        holdout_sharpe = evaluation.get("holdout_sharpe", 0)
        summary = f"検証Sharpe {val_sharpe:.2f} / ホールドアウトSharpe {holdout_sharpe:.2f}"
    else:
        gate_results = evaluation.get("gate_results", {})
        if evaluation.get("error"):
            summary = f"ゲート不通過: {evaluation['error'][:100]}"
        elif not gate_results.get("validation", True):
            summary = f"ゲート不通過: 検証 (Sharpe {evaluation.get('sharpe_ratio', MISSING_METRIC):.2f})"
        elif not gate_results.get("holdout", True):
            summary = f"ゲート不通過: ホールドアウト (Sharpe {evaluation.get('holdout_sharpe', MISSING_METRIC):.2f})"
        else:
            reasons = evaluation.get("reasons", [])
            extra_fail = [r for r in reasons if "Extra criterion failed" in r]
            if extra_fail:
                summary = extra_fail[0].replace("❌ Extra criterion failed: ", "追加基準不通過: ")
            else:
                summary = f"ゲート不通過: クラスタ重複または他要因"

    # ── Route error verdicts for backlog entries ──
    if verdict == Verdict.ERROR:
        # Infra error: retry with attempts counter.
        # attempts persists inside result (backlog.mark writes only status + result),
        # so we must read from the result dict, not from entry top-level —
        # this was a hard-won lesson from PR #17 review.
        attempts = (entry.get("result") or {}).get("attempts", 0) + 1
        if attempts < 3:
            error_text = evaluation.get("error", "")[:200]
            bl.mark(entry["id"], "pending", {
                "verdict": verdict,
                "error": error_text,
                "summary": f"インフラエラー (試行 {attempts}/3): {error_text}",
                "finished_at": datetime.now().isoformat(),
                "attempts": attempts,
            })
        else:
            bl.mark(entry["id"], BacklogStatus.DONE_ERROR, {
                "verdict": verdict,
                "error": evaluation.get("error", "")[:200],
                "summary": f"インフラエラー ({attempts}回試行後): {evaluation.get('error', '')[:200]}",
                "finished_at": datetime.now().isoformat(),
                "attempts": attempts,
            })
    elif verdict == Verdict.CODE_ERROR:
        bl.mark(entry["id"], BacklogStatus.DONE_ERROR, {
            "verdict": verdict,
            "error": evaluation.get("error", "")[:200],
            "summary": f"コードエラー: {evaluation.get('error', '')[:200]}",
            "finished_at": datetime.now().isoformat(),
        })
    else:
        finish_status = BacklogStatus.DONE_ADOPTED if verdict == Verdict.ADOPTED else BacklogStatus.DONE_REJECTED
        bl.mark(entry["id"], finish_status, {
            "verdict": verdict,
            "summary": summary,
            "finished_at": datetime.now().isoformat(),
        })

    return entry, hypothesis, result, verdict, evaluation


# ======================================================================


def _get_loop_config():
    """Return a dict of all environment-driven config values.

    Read at call time (not import time) so tests can monkeypatch env vars.
    """
    return {
        "runner_url": os.environ.get("SANDBOX_RUNNER_URL"),
        "backlog_path": os.environ.get("BACKLOG_PATH", str(BASE_DIR / "backlog.json")),
        "use_llm": os.environ.get("USE_LLM_HYPOTHESIS") == "1",
        "gate_v2": os.environ.get("SANDBOX_GATE_V2", "0"),
    }


def run_loop(num_iterations=3):
    """
    メインPDCAループ
    """
    config = _get_loop_config()
    use_llm = config["use_llm"]
    gate_v2_enabled = config["gate_v2"] == "1"

    print("=" * 60)
    print("🚀 Autonomous Alpha Discovery Loop 開始")
    print(f"   開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S JST')}")
    print(f"   イテレーション数: {num_iterations}")
    print(f"   LLM仮説生成: {'有効' if use_llm else '無効'}")
    print(f"   Gate v2: {'有効' if gate_v2_enabled else '無効'}")
    print(f"CONFIG runner_url={'set' if config['runner_url'] else 'unset'} backlog_path={config['backlog_path']} use_llm={use_llm} gate_v2={'1' if gate_v2_enabled else '0'}")
    if use_llm:
        # Issue #83: HYPO_LLM_BASE_URL / HYPO_LLM_MODEL / HYPO_LLM_API_KEY_ENV come
        # from independent env vars and nothing validated the combination — a
        # deepseek model sent to an Alibaba/MaaS endpoint silently failed every
        # call (403/429) and the loop fell back to random for 7+ days. Fail
        # loudly at startup instead.
        try:
            from llm_hypothesis import validate_llm_config
            _llm_cfg = validate_llm_config()
            print(f"LLM_CONFIG base_url={_llm_cfg['base_url']} model={_llm_cfg['model']} "
                  f"key_env={os.environ.get('HYPO_LLM_API_KEY_ENV', 'DEEPSEEK_API_KEY')}")
        except Exception as e:
            print(f"LLM_CONFIG_ERROR {type(e).__name__}: {e}")
            raise
    print("=" * 60)
    
    knowledge = load_knowledge()
    # Snapshot error count so ERRORS_SUMMARY reports THIS RUN, not cumulative
    _errors_before = len(knowledge.get("errors", []))
    # Snapshot tested count so RUN_SUMMARY reports THIS RUN, not cumulative
    _tested_before = len(knowledge.get("tested", []))

    # Issue #83: aggregate per-run KILLED_SKIP / source / test counts so a run
    # that produced zero tests is loud (RUN_SUMMARY + ZERO_TESTS_ALERT + exit 3)
    # instead of an indistinguishable [SILENT] no-op.
    _RUN_STATS.clear()
    _RUN_STATS.update({
        "ts": datetime.now().isoformat(),
        "iterations": num_iterations,
        "killed_skips": 0,
        "source_llm": 0,
        "source_random": 0,
        "tests_run": 0,
        "tested": 0,
        "zero_tests": False,
    })

    for i in range(num_iterations):
        knowledge["iterations"] += 1
        print(f"\n🔄 Iteration {i+1}/{num_iterations}")
        print("-" * 40)

        # 0. Backlog consumption: try to consume a pending entry first
        consumed = _consume_backlog_entry(knowledge)
        if consumed:
            entry_bk, hypothesis, result, verdict, evaluation = consumed
            _RUN_STATS["tests_run"] = _RUN_STATS.get("tests_run", 0) + 1

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
            elif verdict == Verdict.ADOPTED:
                record["cluster_id"] = evaluation.get("cluster_id", "unknown")
                # Issue #88: persist the consumed spec on the adopted record so
                # the OOS monitor can re-run manifest/codegen strategies later
                # without depending on backlog history surviving forever.
                if entry_bk.get("type") == "manifest":
                    record["manifest_spec"] = entry_bk.get("spec", {})
                elif entry_bk.get("type") == "code":
                    _bk_spec = entry_bk.get("spec", {})
                    record["code_spec"] = {
                        "name": _bk_spec.get("name"),
                        "symbol": _bk_spec.get("symbol"),
                        "code": _bk_spec.get("code"),
                    }
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
        if source_label == "(llm)":
            _RUN_STATS["source_llm"] = _RUN_STATS.get("source_llm", 0) + 1
        else:
            _RUN_STATS["source_random"] = _RUN_STATS.get("source_random", 0) + 1

        # 1a. KILLED family guard (defense in depth)
        param_family_key = _family_key(hypothesis["strategy"], hypothesis["symbol"],
                                       _derive_family_type(hypothesis))
        if param_family_key in get_killed_families(knowledge):
            print(f"KILLED_SKIP {param_family_key}")
            _RUN_STATS["killed_skips"] = _RUN_STATS.get("killed_skips", 0) + 1
            save_knowledge(knowledge)
            continue
        
        # 2. Exhausted-cluster pre-block check
        exhausted, n_failures, best_fail_sharpe = _check_exhausted_cluster(hypothesis, knowledge)
        if exhausted:
            print(f"  ⛔ 枯渇クラスタ ({n_failures} failures, best Sharpe {best_fail_sharpe:.2f}) → バックテスト省略")
            evaluation = {
                "verdict": Verdict.REJECTED,
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
        _RUN_STATS["tests_run"] = _RUN_STATS.get("tests_run", 0) + 1

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
        elif verdict == Verdict.ADOPTED:
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

    # Machine-greppable THIS-RUN error summary for cron reporters.
    # Previously reported the cumulative knowledge.errors count which made
    # every run look like it had 9 errors even when it produced zero.
    errors_this_run = knowledge.get("errors", [])[_errors_before:]
    n_infra = sum(1 for e in errors_this_run if e.get("evaluation", {}).get("error_type") == "infra")
    n_code = sum(1 for e in errors_this_run if e.get("evaluation", {}).get("error_type") == "code")
    print(f"ERRORS_SUMMARY infra={n_infra} code={n_code}")

    # Issue #83: aggregate per-run stats into knowledge.json and make a run that
    # produced zero tests loud (machine-greppable ZERO_TESTS_ALERT) instead of an
    # indistinguishable [SILENT] no-op with last_status=ok.
    tested_this_run = len(knowledge.get("tested", [])) - _tested_before
    _RUN_STATS["tested"] = tested_this_run
    _RUN_STATS["zero_tests"] = bool(
        num_iterations > 0
        and tested_this_run == 0
        and _RUN_STATS.get("killed_skips", 0) > 0
    )
    knowledge["last_run_stats"] = dict(_RUN_STATS)
    knowledge.setdefault("run_stats", []).append(dict(_RUN_STATS))
    if len(knowledge["run_stats"]) > 100:
        knowledge["run_stats"] = knowledge["run_stats"][-100:]
    save_knowledge(knowledge)

    print(
        f"RUN_SUMMARY iterations={num_iterations} tested={tested_this_run} "
        f"killed_skips={_RUN_STATS.get('killed_skips', 0)} "
        f"source_llm={_RUN_STATS.get('source_llm', 0)} "
        f"source_random={_RUN_STATS.get('source_random', 0)}"
    )
    if _RUN_STATS["zero_tests"]:
        print(
            f"ZERO_TESTS_ALERT reason=killed_grid_or_llm_failure "
            f"iterations={num_iterations} tested=0 "
            f"killed_skips={_RUN_STATS.get('killed_skips', 0)}"
        )

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
                entry["evaluation"] = {"verdict": Verdict.ERROR, "error": result["error"],
                                       "error_type": "infra",
                                       "reasons": [f"revalidation failed (infra): {result['error']}"],
                                       "gate_results": {}}
                knowledge.setdefault("errors", []).append(entry)
            elif error_type == "code":
                entry["verdict"] = "code_error"
                entry["evaluation"] = {"verdict": Verdict.CODE_ERROR, "error": result["error"],
                                       "error_type": "code",
                                       "reasons": [f"revalidation failed (code): {result['error']}"],
                                       "gate_results": {}}
                knowledge.setdefault("errors", []).append(entry)
            else:
                entry["verdict"] = "rejected"
                entry["evaluation"] = {"verdict": Verdict.REJECTED, "error": result["error"],
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

        if verdict == Verdict.ADOPTED:
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
    import cp_emit
    _cp_run_id = cp_emit.emit_run_started("9d6c833bd3c5", "autonomous_loop.py")
    _exit_code = 0
    try:
        if "--revalidate" in sys.argv:
            knowledge = load_knowledge()
            run_revalidation(knowledge)
        else:
            iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 3
            knowledge = run_loop(iterations)
        # Issue #83 watchdog: a run that produced zero tests (killed-grid wall or
        # LLM failure) must be loud — non-zero exit + cp_emit error — instead of
        # [SILENT] with last_status=ok.
        _run_stats = knowledge.get("last_run_stats", {})
        if _run_stats.get("zero_tests"):
            _exit_code = 3
            cp_emit.emit_run_finished(
                _cp_run_id, "error",
                error=f"zero tests produced: killed_skips={_run_stats.get('killed_skips')} "
                      f"iterations={_run_stats.get('iterations')}",
            )
        else:
            cp_emit.emit_run_finished(_cp_run_id, "ok")
    except BaseException as _cp_exc:
        cp_emit.emit_run_finished(
            _cp_run_id, "error", error=f"{type(_cp_exc).__name__}: {_cp_exc}"
        )
        raise
    sys.exit(_exit_code)