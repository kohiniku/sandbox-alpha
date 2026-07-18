# Sandbox Alpha 🧪

**自律的AIエージェントによる投資戦略発見フレームワーク**

AIエージェントが仮説生成→バックテスト→評価→蓄積のPDCAサイクルを反復することで、アルファ（超過収益）戦略を自動発見する実験的プロジェクト。

---

## 🎯 コンセプト

### 問題意識

従来の投資戦略開発は：
- 人間が手動で仮説を立て、バックテストコードを書き、結果を評価
- 環境構築・データ取得・評価ロジックが各戦略ごとに重複
- 失敗した戦略の記録・教訓が散逸

### 解決策

**自律的PDCAループ**：
```
[仮説生成] → [バックテスト] → [評価]
    ↑                            ↓
[戦略破棄/採用] ← [ナレッジ更新]
```

- エージェントがナレッジベース参照で仮説を生成（重複回避・adopted近傍探索）
- 常設venv環境でサブプロセスとしてバックテストを実行
- 閾値（Sharpe≥0.3, Return≥5%, Drawdown≥-25%）で自動判定
- 結果をナレッジベースに蓄積し、次の仮説生成に活用

---

## 🚀 クイックスタート

### 必要環境

- Python 3.8+
- 金融データAPI (yfinance使用)

### セットアップ

```bash
# 1. リポジトリをクローン
git clone https://github.com/kohiniku/sandbox-alpha.git
cd sandbox-alpha

# 2. 仮想環境を作成
python3 -m venv .venv
source .venv/bin/activate

# 3. 依存パッケージをインストール
pip install -r requirements.txt

# 4. PDCAループを実行（3回イテレーション）
python3 autonomous_loop.py 3
```

バックテストエンジン単体でも実行可能：

```bash
python3 backtests/backtest_engine.py \
  --strategy sma_crossover \
  --symbol AAPL \
  --params '{"fast_window": 10, "slow_window": 30}'
```

### 結果レポート生成

`report.py` で knowledge.json と results/ から Markdown レポートを生成できます：

```bash
# 標準出力に表示
python3 report.py

# ファイルに出力
python3 report.py --output report.md

# 直近20件の履歴 + タイトル付き
python3 report.py --recent 20 --title "週次レポート"
```

レポート内容：
- 採用/棄却サマリー（件数・採用率）
- 戦略別・銘柄別の集計表（平均Sharpe・平均Return）
- 上位採用戦略の IS/OOS 指標比較
- 直近N件のテスト履歴

### 出力例

```
🚀 Autonomous Alpha Discovery Loop 開始
   開始: 2026-07-17 16:38:12 JST
   イテレーション数: 3

🔄 Iteration 1/3
  💡 仮説: 平均回帰 on NVDA
     パラメータ: {'window': 23, 'threshold': 2.0}
  🔬 バックテスト実行中...
  📋 判定: ADOPTED
     ✅ OOS Sharpe 0.85 >= 0.3
     ✅ OOS Return 26.4% >= 5.0%
     ✅ OOS Drawdown -17.4% >= -25.0%

...

📊 Alpha Discovery Report (Iteration 3)
  採用: 1件 / テスト: 3件 (採用率: 33.3%)

  🏆 採用された戦略:
     • 平均回帰 on NVDA | Sharpe: 0.85 | Return: 26.4% | Trades: 23
```

---

## 📁 プロジェクト構成

```
sandbox-alpha/
├── README.md                    # このファイル
├── requirements.txt             # 依存パッケージ
├── .gitignore                   # gitignore
│
├── autonomous_loop.py           # メインPDCAループ
├── report.py                    # 結果レポート生成
├── knowledge.json               # ナレッジベース（自動生成）
│
├── backtests/
│   └── backtest_engine.py       # バックテストエンジン
│
├── results/                     # 各テスト結果JSON（自動生成）
│   └── hyp_*.json
│
├── strategies/                  # 採用戦略保存先（自動生成）
│
└── docs/
    ├── CONCEPT.md               # 詳細コンセプト
    ├── ARCHITECTURE.md          # アーキテクチャ設計
    └── RESEARCH.md              # 参考論文・リソース
```

---

## 🔧 実装されている戦略

### 1. SMA Crossover（移動平均クロスオーバー）
- パラメータ: `fast_window`, `slow_window`
- ロジック: 短期MAが長期MAを上回ったら買い、下回ったら売り

### 2. Mean Reversion（平均回帰）
- パラメータ: `window`, `threshold`
- ロジック: Z-Scoreが閾値を超えたら逆張り

### 3. Momentum（モメンタム）
- パラメータ: `lookback`, `hold_period`
- ロジック: 過去N日間のリターンが正なら買い、負なら売り

---

## 📊 評価メトリクス

| メトリクス | 閾値 | 説明 |
|-----------|------|------|
| Sharpe Ratio | ≥ 0.3 | リスク調整後リターン |
| Total Return | ≥ 5% | 期間中の累積リターン（複利ベース） |
| Max Drawdown | ≥ -25% | エクイティカーブベースの最大ドローダウン |
| Num Trades | — | ポジション変化回数（取引コスト計算に使用） |

### 取引コスト

片道 5.0 bps（0.05%）をポジション変化ごとに控除。`backtests/backtest_engine.py` の `COST_BPS` 定数で調整可能。

### Walk-Forward 検証

データを train 60% / validation 20% / holdout 20% に3分割（時系列順）。採用判定は validation 指標で行い、holdout は最終確認のみに使用。全3期間の指標と `num_days` を結果JSONに含める。

### 🛡️ 過学習対策 (Overfitting Guards)

自律的PDCAループにおけるデータスヌーピング（p-hacking）を防ぐ3つの仕組み。

**1. 3-Way Walk-Forward Split（データ分割）**

データを時系列で train 60% / validation 20% / holdout 20% に分割。パラメータ選択・採用判定は validation 期間の指標のみで行う。holdout は最終確認のみに使用され、一度も選択/ランキングに使われない。真のアウトオブサンプル性能を担保する。

**2. Deflated Sharpe Threshold（試行回数補正）**

複数回の試行（N_family: 同一 strategy + symbol の組み合わせ数）に応じて、採用閾値を自動的に引き上げる。

```
effective_min_sharpe = max(0.5, √(2 × ln(max(N, 2))) × √(252 / T_val))
```

試行回数が増えるほど閾値が上昇し、偶然の良い結果を排除する。検証期間が短いほど閾値も上昇。計算された閾値は評価レコードに `effective_min_sharpe` として記録される。

**3. Cluster Dedup（クラスタ重複排除）**

同じ (strategy, symbol) で全数値パラメータが ±15% 以内（リストパラメータは1ステップ以内）の候補は同一クラスタとみなす。クラスタ内では holdout Sharpe が最も高い採用戦略のみを保持し、それ以外は棄却。置き換えられた戦略は `superseded` リストに移動。

**再検証モード**: `python3 autonomous_loop.py --revalidate` で全採用戦略を最新パイプラインで再評価し、不合格のものを降格する。

---

## 🔬 実験結果（2026-07-17時点）

**10回のPDCAループで4件のアルファ戦略を発見：**

- 🥇 **平均回帰 on NVDA** — Sharpe 0.85, Return 26.4%, DD -17.4%
- 🥈 **移動平均クロスオーバー on SPY** — Sharpe 0.73, Return 7.8%, DD -3.9%
- 🥉 **平均回帰 on MSFT** — Sharpe 0.49, Return 9.6%, DD -14.6%
- 🏅 **移動平均クロスオーバー on BTC-USD** — Sharpe 0.66, Return 16.1%, DD -7.4%

詳細は `results/` ディレクトリ参照。

---

## 🔒 サンドボックス実行（Trusted Runner）

LLM生成コードを安全に実行するための隔離実行モード。環境変数 `SANDBOX_RUNNER_URL` を設定すると、バックテストはローカルの subprocess ではなく、信頼済みの外部 sandbox-runner サービスに HTTP で委譲される。

```bash
# サンドボックスモードで実行
export SANDBOX_RUNNER_URL="http://sandbox-runner:8080"
python3 autonomous_loop.py 10

# 未設定の場合は従来の subprocess / venv 実行（開発・テスト用）
python3 autonomous_loop.py 10
```

**仕組み**:
- `autonomous_loop.py` の `run_backtest()` が `SANDBOX_RUNNER_URL` を検出すると、`urllib.request`（stdlibのみ）で `POST /run` を送信
- sandbox-runner は strategy / symbol / params だけを受け取り、固定イメージを `--network=none --read-only --cap-drop=ALL` で起動
- エージェント（Hermes）は Docker に一切アクセスしない ― データ送信のみ

詳細は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) の "Future Work: Sandbox Isolation" を参照。

---

## 🧠 LLM仮説生成（データのみ・バリデーション付き）

LLMによるインテリジェントな仮説生成モード。戦略テンプレートのパラメータ空間とナレッジベースをDeepSeek APIに送信し、未テストかつ有望なパラメータを提案させる。

```bash
# 環境変数で有効化
export USE_LLM_HYPOTHESIS=1
export DEEPSEEK_API_KEY="sk-..."
python3 autonomous_loop.py 5
```

**動作**: LLMは戦略名・銘柄・パラメータ（データのみ）をJSONで返す。コードを出力することは一切ない。提案は厳格にバリデーションされ（戦略存在チェック、パラメータ範囲チェック、重複チェック）、失敗時は自動でランダム生成にフォールバック。

**設定用環境変数**:
| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `USE_LLM_HYPOTHESIS` | — | `1` でLLMパスを有効化 |
| `HYPO_LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI互換APIのベースURL |
| `HYPO_LLM_MODEL` | `deepseek-v4-pro` | 使用するモデル名 |
| `HYPO_LLM_API_KEY_ENV` | `DEEPSEEK_API_KEY` | APIキーを持つ環境変数名 |

---

## 🚧 今後の拡張

### 短期（1-2週間）
- [ ] Polymarket API統合（予測市場でのペーパートレード）
- [ ] ポートフォリオ結合（採用戦略を相関考慮して組み合わせ）

### 中期（1-2ヶ月）
- [ ] 取引コスト精緻化（スプレッド・スリッページ）
- [ ] 並列バックテスト（複数戦略を同時検証）
- [ ] 自動定期実行（cronジョブで毎日自律的に探索）

### 長期（3-6ヶ月）
- [ ] マルチエージェント協調（リサーチ担当・リスク管理担当・実行担当）
- [ ] 強化学習統合（FinRLと連携して動的ポートフォリオ最適化）
- [ ] 実運用ゲート（ペーパートレード→少額実資金→本格運用の段階的移行）

---

## 📚 関連研究

### 直接関連
- **RD-Agent-Quant (Microsoft, NeurIPS 2025)** — 仮説→実装→バックテストの完全自律ループ
- **EVOQUANT (2026)** — 自己進化型戦略最適化
- **QuantAgents** — マルチエージェント金融システム

### 方法論基盤
- **AI Scientist (Sakana AI)** — 科学的研究の完全自動化
- **Voyager (NVIDIA)** — 自律的スキル獲得・蓄積
- **AutoGPT** — 自律的タスク分解・実行

詳細は [`docs/RESEARCH.md`](docs/RESEARCH.md) 参照。

---

## 🤝 貢献

Issues・Pull Requests歓迎。特に以下に興味ある方：
- 新しい戦略テンプレートの追加
- 評価メトリクスの改善
- Polymarket以外のオルタナティブデータ統合

---

## 📄 ライセンス

MIT License

---

## ⚠️ 免責事項

本プロジェクトは研究・教育目的です。実際の投資判断には使用しないでください。過去のバックテスト結果は将来の収益を保証するものではありません。

---

**Author**: kohiniku  
**Created**: 2026-07-17  
**Last Updated**: 2026-07-18
