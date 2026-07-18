# Sandbox Alpha 🧪

**自律的AIエージェントによる投資戦略発見フレームワーク**

AIエージェントがサンドボックス環境を自律的に構築し、仮説生成→ペーパートレード→評価→破棄のPDCAサイクルを反復することで、アルファ（超過収益）戦略を自動発見する実験的プロジェクト。

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
[仮説生成] → [サンドボックス構築] → [ペーパートレード] → [評価]
    ↑                                                      ↓
[戦略破棄/採用] ← [ナレッジ更新] ← [パフォーマンス分析]
```

- エージェントがランダムまたはナレッジベース参照で仮説を生成
- 使い捨てvenv環境でバックテストを実行
- 閾値（Sharpe≥0.3, Return≥5%, Drawdown≥-25%）で自動判定
- 結果をナレッジベースに蓄積し、次の仮説生成に活用

---

## 🚀 クイックスタート

### 必要環境

- Python 3.8+
- Docker (オプション、使い捨てコンテナ用)
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

# 4. PDCAループを実行（10回イテレーション）
python3 autonomous_loop.py 10
```

### 出力例

```
🚀 Autonomous Alpha Discovery Loop 開始
   開始: 2026-07-17 16:38:12 JST
   イテレーション数: 10

🔄 Iteration 1/10
  💡 仮説: 平均回帰 on NVDA
     パラメータ: {'window': 23, 'threshold': 2.0}
  🔬 バックテスト実行中...
  📋 判定: ADOPTED
     ✅ Sharpe 0.85 >= 0.3
     ✅ Return 26.4% >= 5.0%
     ✅ Drawdown -17.4% >= -25.0%

...

📊 Alpha Discovery Report (Iteration 10)
  採用: 4件 / テスト: 10件 (採用率: 40.0%)

  🏆 採用された戦略:
     • 平均回帰 on NVDA | Sharpe: 0.85 | Return: 26.4%
     • 移動平均クロスオーバー on SPY | Sharpe: 0.73 | Return: 7.8%
     • 平均回帰 on MSFT | Sharpe: 0.49 | Return: 9.6%
     • 移動平均クロスオーバー on BTC-USD | Sharpe: 0.66 | Return: 16.1%
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
| Total Return | ≥ 5% | 期間中の累積リターン |
| Max Drawdown | ≥ -25% | 最大ドローダウン |

---

## 🔬 実験結果（2026-07-17時点）

**10回のPDCAループで4件のアルファ戦略を発見：**

- 🥇 **平均回帰 on NVDA** — Sharpe 0.85, Return 26.4%, DD -17.4%
- 🥈 **移動平均クロスオーバー on SPY** — Sharpe 0.73, Return 7.8%, DD -3.9%
- 🥉 **平均回帰 on MSFT** — Sharpe 0.49, Return 9.6%, DD -14.6%
- 🏅 **移動平均クロスオーバー on BTC-USD** — Sharpe 0.66, Return 16.1%, DD -7.4%

詳細は `results/` ディレクトリ参照。

---

## 🚧 今後の拡張

### 短期（1-2週間）
- [ ] Polymarket API統合（予測市場でのペーパートレード）
- [ ] LLM仮説生成（ランダムではなく、ナレッジベースを参照して賢い仮説を生成）
- [ ] ポートフォリオ結合（採用戦略を相関考慮して組み合わせ）

### 中期（1-2ヶ月）
- [ ] 多期間検証（採用戦略をout-of-sample期間でもテスト）
- [ ] Docker使い捨てコンテナ（完全隔離環境での並列テスト）
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
**Last Updated**: 2026-07-17
