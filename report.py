from autonomous_loop import MISSING_METRIC
#!/usr/bin/env python3
"""
Sandbox Alpha 結果レポート生成モジュール
knowledge.json と results/*.json を読み込み、Markdownレポートを生成する。

Usage:
    python3 report.py [--output report.md] [--recent N]
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge.json"
RESULTS_DIR = BASE_DIR / "results"


def load_knowledge():
    """knowledge.json を読み込む。なければ空データを返す。"""
    if not KNOWLEDGE_FILE.exists():
        return {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [], "iterations": 0}
    try:
        return json.loads(KNOWLEDGE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"tested": [], "tested_combinations": [], "adopted": [], "rejected": [], "iterations": 0}


def load_results():
    """results/*.json をすべて読み込む。なければ空リストを返す。"""
    if not RESULTS_DIR.exists():
        return []
    records = []
    for fpath in sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            records.append(json.loads(fpath.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def safe_get(d, *keys, default=""):
    """ネスト辞書から安全に値を取り出す。"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, {})
        else:
            return default
    return d if d != {} else default


def fmt_pct(val):
    """パーセント値を安全にフォーマット。"""
    try:
        v = float(val)
        if v <= MISSING_METRIC:
            return "N/A"
        return f"{v:+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_num(val, precision=2):
    """数値を安全にフォーマット。"""
    try:
        v = float(val)
        if v <= MISSING_METRIC:
            return "N/A"
        return f"{v:.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"


def collect_all_records():
    """knowledge.json の adopted/rejected と results/ を統合し、一意なレコードリストを返す。"""
    knowledge = load_knowledge()
    id_set = set()
    records = []

    for rec in knowledge.get("adopted", []) + knowledge.get("rejected", []):
        hid = safe_get(rec, "hypothesis", "id")
        if hid and hid not in id_set:
            id_set.add(hid)
            records.append(rec)

    for rec in load_results():
        hid = safe_get(rec, "hypothesis", "id")
        if hid and hid not in id_set:
            id_set.add(hid)
            records.append(rec)

    records.sort(key=lambda r: safe_get(r, "tested_at"), reverse=True)
    return records


def build_summary(records):
    """採用/棄却サマリー。"""
    adopted = [r for r in records if r.get("verdict") == "adopted"]
    rejected = [r for r in records if r.get("verdict") == "rejected"]
    errors = [r for r in records if "error" in r.get("backtest_result", {})]
    total = len(records)
    rate = (len(adopted) / max(total, 1)) * 100
    return adopted, rejected, errors, total, rate


def build_strategy_table(records):
    """戦略別集計。"""
    by_strategy = defaultdict(lambda: {"adopted": 0, "rejected": 0, "sharpe_sum": 0.0, "sharpe_count": 0, "return_sum": 0.0, "return_count": 0})
    for r in records:
        strategy = safe_get(r, "hypothesis", "strategy")
        if not strategy:
            continue
        verd = r.get("verdict", "rejected")
        by_strategy[strategy][verd] += 1
        ev = r.get("evaluation", {})
        if isinstance(ev, dict):
            sr = ev.get("sharpe_ratio")
            tr = ev.get("total_return_pct")
            if sr is not None and sr != MISSING_METRIC:
                by_strategy[strategy]["sharpe_sum"] += float(sr)
                by_strategy[strategy]["sharpe_count"] += 1
            if tr is not None and tr != MISSING_METRIC:
                by_strategy[strategy]["return_sum"] += float(tr)
                by_strategy[strategy]["return_count"] += 1
    return by_strategy


def build_symbol_table(records):
    """銘柄別集計。"""
    by_symbol = defaultdict(lambda: {"adopted": 0, "rejected": 0, "sharpe_sum": 0.0, "sharpe_count": 0, "return_sum": 0.0, "return_count": 0})
    for r in records:
        symbol = safe_get(r, "hypothesis", "symbol")
        if not symbol:
            continue
        verd = r.get("verdict", "rejected")
        by_symbol[symbol][verd] += 1
        ev = r.get("evaluation", {})
        if isinstance(ev, dict):
            sr = ev.get("sharpe_ratio")
            tr = ev.get("total_return_pct")
            if sr is not None and sr != MISSING_METRIC:
                by_symbol[symbol]["sharpe_sum"] += float(sr)
                by_symbol[symbol]["sharpe_count"] += 1
            if tr is not None and tr != MISSING_METRIC:
                by_symbol[symbol]["return_sum"] += float(tr)
                by_symbol[symbol]["return_count"] += 1
    return by_symbol


def build_is_oos_comparison(records, top_n=5):
    """上位戦略のIS/OOS指標比較。adopted 戦略を Sharpe の高い順にソート。"""
    adopted = [r for r in records if r.get("verdict") == "adopted"]
    adopted.sort(
        key=lambda r: float(safe_get(r, "evaluation", "sharpe_ratio", default=-999)),
        reverse=True
    )
    return adopted[:top_n]


def build_recent_history(records, n=10):
    """直近N件のテスト履歴。"""
    return records[:n]


def generate_report(records=None, recent_n=10, title=None):
    """Markdownレポートを生成して文字列で返す。"""
    if records is None:
        records = collect_all_records()

    if not records:
        return "📊 Sandbox Alpha 結果レポート\n\nデータがありません。`python3 autonomous_loop.py N` を実行してバックテストを開始してください。\n"

    adopted, rejected, errors, total, rate = build_summary(records)
    by_strategy = build_strategy_table(records)
    by_symbol = build_symbol_table(records)
    top_adopted = build_is_oos_comparison(records)
    recent = build_recent_history(records, recent_n)
    knowledge = load_knowledge()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")
    lines = []
    lines.append(f"# 📊 Sandbox Alpha 結果レポート")
    if title:
        lines.append(f"> {title}")
    lines.append(f"**生成日時**: {ts}")
    lines.append(f"**総イテレーション数**: {knowledge.get('iterations', 0)}")
    lines.append("")

    # ── 採用/棄却サマリー ──
    lines.append("---")
    lines.append("## 📋 採用/棄却サマリー")
    lines.append("")
    lines.append(f"- 総テスト数: **{total}**")
    lines.append(f"- 採用: **{len(adopted)}** 件（採用率: {rate:.1f}%）")
    lines.append(f"- 棄却: **{len(rejected)}** 件")
    lines.append(f"- エラー: **{len(errors)}** 件")
    lines.append("")

    if not adopted:
        lines.append("⚠️ 採用された戦略はまだありません。")
        lines.append("")

    # ── 戦略別集計 ──
    lines.append("---")
    lines.append("## 🎯 戦略別集計")
    lines.append("")
    lines.append("| 戦略 | 採用 | 棄却 | 平均Sharpe | 平均Return |")
    lines.append("|------|------|------|------------|------------|")
    for sname in sorted(by_strategy.keys()):
        s = by_strategy[sname]
        avg_sharpe = s["sharpe_sum"] / max(s["sharpe_count"], 1)
        avg_return = s["return_sum"] / max(s["return_count"], 1)
        lines.append(f"| {sname} | {s['adopted']} | {s['rejected']} | {avg_sharpe:.2f} | {avg_return:.1f}% |")
    lines.append("")

    # ── 銘柄別集計 ──
    lines.append("---")
    lines.append("## 🏷️ 銘柄別集計")
    lines.append("")
    lines.append("| 銘柄 | 採用 | 棄却 | 平均Sharpe | 平均Return |")
    lines.append("|------|------|------|------------|------------|")
    for sym in sorted(by_symbol.keys()):
        s = by_symbol[sym]
        avg_sharpe = s["sharpe_sum"] / max(s["sharpe_count"], 1)
        avg_return = s["return_sum"] / max(s["return_count"], 1)
        lines.append(f"| {sym} | {s['adopted']} | {s['rejected']} | {avg_sharpe:.2f} | {avg_return:.1f}% |")
    lines.append("")

    # ── 上位戦略 IS/OOS比較 ──
    if top_adopted:
        lines.append("---")
        lines.append("## 🏆 上位採用戦略（IS/OOS 指標比較）")
        lines.append("")
        for idx, rec in enumerate(top_adopted, 1):
            hyp = rec.get("hypothesis", {})
            ev = rec.get("evaluation", {})
            bt = rec.get("backtest_result", {})

            strat = hyp.get("strategy", "N/A")
            symbol = hyp.get("symbol", "N/A")
            desc = hyp.get("description", f"{strat} on {symbol}")
            params = hyp.get("params", {})

            lines.append(f"### {idx}. {desc}")
            lines.append("")
            if params:
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"パラメータ: `{params_str}`")
                lines.append("")

            lines.append("| 指標 | In-Sample | Out-of-Sample |")
            lines.append("|------|-----------|---------------|")

            is_data = ev.get("in_sample", {})
            oos_data = bt.get("out_of_sample", {})

            oos_sharpe = ev.get("sharpe_ratio", "")
            oos_return = ev.get("total_return_pct", "")
            oos_dd = ev.get("max_drawdown_pct", "")

            is_sharpe = is_data.get("sharpe_ratio", "")
            is_return = is_data.get("total_return_pct", "")
            is_dd = is_data.get("max_drawdown_pct", "")

            if oos_data:
                oos_sharpe = oos_data.get("sharpe_ratio", oos_sharpe)
                oos_return = oos_data.get("total_return_pct", oos_return)
                oos_dd = oos_data.get("max_drawdown_pct", oos_dd)

            lines.append(f"| Sharpe Ratio | {fmt_num(is_sharpe, 3)} | {fmt_num(oos_sharpe, 3)} |")
            lines.append(f"| Total Return | {fmt_pct(is_return)} | {fmt_pct(oos_return)} |")
            lines.append(f"| Max Drawdown | {fmt_pct(is_dd)} | {fmt_pct(oos_dd)} |")
            lines.append(f"| Num Trades | {safe_get(is_data, 'num_trades') if is_data else 'N/A'} | {safe_get(ev, 'num_trades', default=safe_get(oos_data, 'num_trades'))} |")
            lines.append("")

            reasons = ev.get("reasons", [])
            if reasons:
                lines.append("**判定理由**:")
                for reason in reasons:
                    lines.append(f"- {reason}")
                lines.append("")

    # ── 直近N件のテスト履歴 ──
    lines.append("---")
    lines.append(f"## 📜 直近{recent_n}件のテスト履歴")
    lines.append("")
    if not recent:
        lines.append("テスト履歴はありません。")
    else:
        for rec in recent:
            hyp = rec.get("hypothesis", {})
            ev = rec.get("evaluation", {})
            verd = rec.get("verdict", "unknown").upper()
            tested_at = rec.get("tested_at", "N/A")
            desc = hyp.get("description", "N/A")
            sharpe = fmt_num(safe_get(ev, "sharpe_ratio"), 2)
            ret = fmt_pct(safe_get(ev, "total_return_pct"))
            dd = fmt_pct(safe_get(ev, "max_drawdown_pct"))

            icon = "✅" if verd == "ADOPTED" else "❌"
            lines.append(f"- {icon} **{verd}** | {desc} | Sharpe: {sharpe} | Return: {ret} | DD: {dd} | {tested_at}")
        lines.append("")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by Sandbox Alpha report.py*")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Sandbox Alpha 結果レポート生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 report.py                     # stdoutに出力
  python3 report.py --output report.md  # ファイルに出力
  python3 report.py --recent 20         # 直近20件の履歴を表示
  python3 report.py --title "週次レポート" # タイトル付き
        """
    )
    parser.add_argument("--output", "-o", default=None, help="出力ファイルパス（デフォルト: stdout）")
    parser.add_argument("--recent", "-n", type=int, default=10, help="直近N件のテスト履歴を表示（デフォルト: 10）")
    parser.add_argument("--title", "-t", default=None, help="レポートタイトル")
    args = parser.parse_args()

    records = collect_all_records()
    report = generate_report(records=records, recent_n=args.recent, title=args.title)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report)
        print(f"✅ レポートを {out_path.resolve()} に出力しました。")
    else:
        print(report)


if __name__ == "__main__":
    main()
