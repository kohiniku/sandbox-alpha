# sandbox-alpha v2: Reproducible-by-design framework

## Motivation

v1 の固定ハーネス（`generate_signals(df)`・単一銘柄・OHLCV固定・Sharpe固定）は、
考案LLMが提案できる戦略の空間を「単一銘柄テクニカル」に限定していた。
定時収集している投資戦略論文の約2/3（クロスセクショナル・RL・オルタナ・オプション・
構造化データ・カスタム評価器）が枠組み自体の制約で再現不能な状態だった。

v2の目標は、**論文で提案されている手法を、データ問題を除けば枠組みが妨げない**
状態にすること。

## Target architecture

エージェント（LLM）は「戦略コード」ではなく「**戦略マニフェスト**」を書く:

```json
{
  "name": "cross_sectional_momentum_top20",
  "code_b64": "...",
  "data_sources": [
    {"type": "ohlcv", "universe": ["AAPL", "MSFT", ...], "start": "2020-01-01"},
    {"type": "news_sentiment", "universe": [...], "start": "..."}
  ],
  "model_artifacts": [{"name": "timesfm-base", "revision": "..."}],
  "compute": {"mode": "inference", "budget_seconds": 60, "gpu": false},
  "evaluator": {
    "type": "portfolio",
    "metrics": ["sharpe", "ir", "turnover", "cvar_95", "factor_exposure"],
    "benchmark": "SPY"
  }
}
```

Runner は**サンドボックス化された計算環境**として振る舞う:
- マニフェスト検証（宣言された data_sources と model_artifacts のみ read-only mount）
- 隔離コンテナ（既存の制約: cap_drop, no-new-privileges, network分離）
- 宣言済みメトリクスの計算・返却

## Non-negotiable invariants

- **信頼境界**: hermesが docker.sock に触らない。runnerのみが実行。
- **network遮断（推論時）**: モデル/データは事前フェッチ、read-only mount。
- **リソース上限**: mem/pids/timeout は既存同等以上。GPUモードは別endpoint。

## Phasing

### Phase 0 — 基盤（Week 1）
- PR-A: **Manifest schema** (Python module + JSON schema + validator, pure code)
- PR-B: **Runner /run_manifest endpoint** (backwards compat: /run, /run_code, /validate 温存)
- PR-C: **Multi-symbol OHLCV data adapter** (`data_sources.type=ohlcv, universe: [...]`)
- PR-D: **Portfolio evaluator plugin framework** (Sharpe/IR/turnover/CVaR/factor exposure)

### Phase 1 — エージェント統合（Week 1-2）
- PR-E: **Ideation v3** — manifest を emit するプロンプト、テンプレ拘束の解除
- PR-F: **Consumption loop v2** — /run_manifest 経由でマニフェスト実行、旧経路併存
- PR-G: **Migration** — 旧 knowledge スキーマの family/near_miss/error を新形式に持ち込み

### Phase 2 — オルタナデータ（Week 2-3）
- PR-H: **News/sentiment adapter** (arXiv集めた corpus を活用)
- PR-I: **SEC filings adapter** (13F, 10-K の公開API)
- PR-J: **Insider trading adapter** (SEC Form 4)
- PR-K: **Macro data adapter** (FRED公開API)

### Phase 3 — 学習と事前学習モデル（Week 3-4）
- PR-L: **Training mode container** (時間予算大、書き込み可能なscratch、checkpoint永続化)
- PR-M: **Model artifact store** (hermes側フェッチ → sandbox-data volume → runner RO mount)
- PR-N: **GPU access** (nvidia-runtime, `compute.gpu=true` 経路)

### Phase 4 — 高度な評価器（Week 4+）
- PR-O: **Custom objective functions** (recursive utility, tax-aware, personalized)
- PR-P: **Tail metrics** (VaR, CVaR, expected shortfall, drawdown distribution)
- PR-Q: **Factor decomposition** (Fama-French 5+momentum アトリビューション)

## Migration strategy

- 旧cron (69d74ba128df, 9d6c833bd3c5) は v2 が Phase 1 完了まで並行運転
- Phase 1 完了時点で **/run_manifest** に切替、旧 /run と /run_code は非推奨だが残置
- knowledge.json は v1/v2 両形式を読める後方互換ロード、書き込みは v2 のみに

## Ownership

- Runner 側変更 (PR-B, PR-L, PR-M, PR-N): Claude 直接（信頼境界コンポーネント）
- Repo 側変更 (PR-A, C-K, O-Q): hermes 委譲
