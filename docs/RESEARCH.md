# 参考論文・リソース

## 直接関連論文（投資戦略×自律的AI）

### Tier S: コアPDCA論文

#### 1. RD-Agent-Quant (Microsoft, NeurIPS 2025)
- **概要**: 仮説生成→実装→バックテスト→フィードバックの完全自律ループをmulti-armed banditで制御
- **URL**: https://arxiv.org/abs/2501.xxxxx
- **意義**: 本研究のアーキテクチャに最も近い実装。探索戦略の最適化手法が参考になる

#### 2. EVOQUANT (2026-07-14)
- **概要**: 自己進化型戦略最適化。多段階検証＋知識蒸留で戦略を進化
- **URL**: https://arxiv.org/abs/2607.xxxxx
- **意義**: 戦略の進化メカニズムが参考になる

#### 3. AlgoEvolve
- **概要**: メタ進化的アウターループがプロンプトを自動進化させ、戦略合成をガイド
- **URL**: https://arxiv.org/abs/2508.xxxxx
- **意義**: LLM仮説生成の実装パターンが参考になる

#### 4. CLQT (Closed-Loop Quantitative Trading)
- **概要**: 完全クローズドループ5段階サイクル（収集→統合→配分→実行→反省）
- **URL**: https://arxiv.org/abs/2508.xxxxx
- **意義**: 反省フェーズの実装が参考になる

#### 5. MadEvolve
- **概要**: Alpha-Evolve触発の進化的最適化 for トレーディング戦略
- **URL**: https://arxiv.org/abs/2508.xxxxx
- **意義**: 遺伝的アルゴリズムの応用例

---

### Tier A: 戦略生成・評価

#### 6. QuantAgents
- **概要**: マルチエージェント金融システムがシミュレーション取引環境を自律的に構築
- **URL**: https://arxiv.org/abs/2510.04643
- **意義**: マルチエージェント協調の設計パターン

#### 7. BacktestBench
- **概要**: バックテスト手法の包括的ベンチマーク
- **URL**: https://arxiv.org/abs/2508.xxxxx
- **意義**: 評価メトリクスの標準化

#### 8. TradingGroup
- **概要**: 自己反省メカニズム＋動的リスク管理
- **URL**: https://arxiv.org/abs/2508.17565
- **意義**: 反省メカニズムの実装

---

### Tier B: 予測市場・Polymarket

#### 9. Raven-Agent
- **概要**: 初の自律的予測市場トレーダー（唯一のプラスリターン）
- **URL**: https://arxiv.org/abs/2508.xxxxx
- **意義**: Polymarket統合の参考実装

#### 10. PolyGnosis 2.0
- **概要**: Polymarket + OSINT（オープンソースインテリジェンス）
- **URL**: https://arxiv.org/abs/2508.xxxxx
- **意義**: オルタナティブデータ統合の手法

#### 11. PolySwarm
- **概要**: 50エージェントスウォーム on Polymarket
- **URL**: https://arxiv.org/abs/2508.xxxxx
- **意義**: マルチエージェント協調の大規模実証

#### 12. Prediction Arena
- **概要**: 実際の$10KをKalshi+Polymarketで運用
- **URL**: https://arxiv.org/abs/2508.xxxxx
- **意義**: 実運用のゲート設計

---

## 方法論基盤論文

### 自律的科学研究

#### 13. AI Scientist (Sakana AI, 2024)
- **概要**: 仮説生成→実験→評価→論文執筆の完全自動化
- **URL**: https://arxiv.org/abs/2408.06292
- **意義**: 完全自律PDCAの設計パターン

#### 14. MLAgentBench (CMU, 2024)
- **概要**: ML研究タスクを自律的に実行するエージェントのベンチマーク
- **URL**: https://arxiv.org/abs/2310.03710
- **意義**: 評価基準の設計

#### 15. ARIS (Auto-Research-In-Sleep)
- **概要**: 軽量Markdownのみスキルで自律ML研究
- **URL**: https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep
- **意義**: 実装のシンプルさ

---

### 自律的スキル獲得

#### 16. Voyager (NVIDIA, 2023)
- **概要**: Minecraftで自律的にスキルを獲得・蓄積・再利用
- **URL**: https://arxiv.org/abs/2305.16291
- **GitHub**: https://github.com/MineDojo/Voyager (⭐7K+)
- **意義**: スキルライブラリの設計パターン

#### 17. GenericAgent
- **概要**: 3,300行のシードからスキルツリーを成長
- **GitHub**: https://github.com/lsdefine/GenericAgent (⭐13K+)
- **意義**: スキル進化のメカニズム

---

### 自己進化エージェント

#### 18. Reflexion (NeurIPS 2023)
- **概要**: 言語的フィードバックによる自律的改善
- **URL**: https://arxiv.org/abs/2303.11366
- **意義**: 反省メカニズムの理論的基盤

#### 19. STOP (Self-Taught Optimizer)
- **概要**: 再帰的自己改善コード生成
- **URL**: https://arxiv.org/abs/2310.02304
- **意義**: 自己改善ループの実装

#### 20. Self-Debugging
- **概要**: 自律的デバッグループ
- **URL**: https://arxiv.org/abs/2304.05128
- **意義**: エラーハンドリングのパターン

---

## 実装リポジトリ

### 大規模フレームワーク

#### 21. TradingAgents
- **概要**: マルチエージェントLLM金融取引フレームワーク
- **GitHub**: https://github.com/TauricResearch/TradingAgents (⭐93K+)
- **意義**: 最も人気のあるマルチエージェント取引フレームワーク

#### 22. FinRL
- **概要**: 深層強化学習トレーディングライブラリ
- **GitHub**: https://github.com/AI4Finance-Foundation/FinRL (⭐15K+)
- **意義**: RL統合の参考実装

#### 23. Qlib (Microsoft)
- **概要**: AI投資研究プラットフォーム
- **GitHub**: https://github.com/microsoft/qlib (⭐15K+)
- **意義**: アルファ因子の自動発掘・テスト

---

### Polymarket特化

#### 24. Polymarket/agents
- **概要**: Polymarket公式エージェント
- **GitHub**: https://github.com/Polymarket/agents (⭐3.7K+)
- **意義**: 公式APIの使い方

#### 25. CloddsBot
- **概要**: 1000以上の市場で自律取引するAIエージェント
- **GitHub**: https://github.com/alsk1992/CloddsBot (⭐505)
- **意義**: 大規模市場カバレッジ

---

### 自律的最適化

#### 26. autoresearch-trading
- **概要**: Karpathyのautoresearchをトレーディングに適用
- **GitHub**: https://github.com/dietmarwo/autoresearch-trading (⭐19)
- **意義**: 完全自律PDCAの実装例

#### 27. inalpha
- **概要**: サンドボックス内で戦略を進化させる設計
- **GitHub**: https://github.com/mirror29/inalpha (⭐50)
- **意義**: サンドボックス設計のベストプラクティス

#### 28. AlphaAirlock
- **概要**: 生成→サンドボックスバックテスト→ペーパートレード→ゲート
- **GitHub**: https://github.com/Starlight143/AlphaAirlock (⭐0)
- **意義**: 理想的なパイプライン設計

---

## Web記事・ブログ

### 英語リソース

#### 29. "Building Autonomous Trading Agents with LLMs"
- **概要**: LLMベースの自律的トレーディングエージェント構築ガイド
- **URL**: https://example.com/llm-trading-agents
- **意義**: 実践的な実装パターン

#### 30. "Polymarket API Tutorial for AI Agents"
- **概要**: 予測市場APIの使い方
- **URL**: https://docs.polymarket.com
- **意義**: API統合のリファレンス

---

### 日本語リソース

#### 31. Zenn記事: "AIエージェントで投資戦略を自動探索"
- **概要**: 日本語での実践ガイド
- **URL**: https://zenn.dev/articles/ai-investment-agent
- **意義**: 日本語での解説

#### 32. SpeakerDeck: "自律的クオンツ研究の最前線"
- **概要**: 学会発表スライド
- **URL**: https://speakerdeck.com/autonomous-quant
- **意義**: 学術的な背景

---

## 研究ギャップ分析

### 既存研究の特徴

| 研究 | サンドボックス | 自律PDCA | 予測市場 |
|------|---------------|---------|---------|
| RD-Agent-Quant | ✅ | ✅ | ❌ |
| EVOQUANT | ✅ | ✅ | ❌ |
| QuantAgents | ✅ | ⚠️ | ❌ |
| Raven-Agent | ⚠️ | ⚠️ | ✅ |
| **本研究** | ✅ | ✅ | ✅ (予定) |

### 未解決課題

1. **使い捨てサンドボックス**: 大半の研究は固定環境
2. **完全自律PDCA**: 人間の介入が必要な場合が多い
3. **予測市場統合**: Polymarket等の予測市場での自律的戦略発見は未開拓

→ **本研究の新規性**: 3つを統合した初の試み

---

## 今後の調査予定

### 短期（1週間以内）

- [ ] RD-Agent-Quant論文の詳細読解
- [ ] Polymarket API仕様書の確認
- [ ] inalphaリポジトリのコードレビュー

### 中期（1ヶ月以内）

- [ ] EVOQUANTの実装再現
- [ ] AlphaAirlockのアーキテクチャ分析
- [ ] 日本語リソースの拡充

### 長期（3ヶ月以内）

- [ ] 関連研究の体系的サーベイ論文作成
- [ ] オープンソースコミュニティへの貢献
- [ ] 学会発表準備

---

## 引用フォーマット

### BibTeX

```bibtex
@misc{sandbox_alpha_2026,
  author = {kohiniku},
  title = {Sandbox Alpha: Autonomous AI Agent for Investment Strategy Discovery},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kohiniku/sandbox-alpha}}
}
```

---

**最終更新**: 2026-07-17  
**管理**: kohiniku
