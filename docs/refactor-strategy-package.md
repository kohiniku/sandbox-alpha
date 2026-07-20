# 戦略パッケージ化リファクタ設計 (refactor/strategy-package-design)

挙動を一切変えずに、`backtest_engine.py` に内蔵されていた4戦略を独立パッケージへ
分離した場合の設計を、実コード＋機械検証つきで確認したブランチ。
設計の考え方は xlblueprint 調査メモ
(hermes-orchestration/docs/ideas-from-xlblueprint.md) の #1 (propose/apply+自己検証)・
#2 (正規化diff)・#5 (決定性) を適用したもの。

## Before / After

```
Before                              After
backtests/                          backtests/
├── backtest_engine.py              ├── backtest_engine.py   (CLI+run_backtest のみ, 戦略はimport)
│   ├── run_sma_crossover_strategy  ├── strategies/
│   ├── run_mean_reversion_strategy │   ├── __init__.py      (レジストリ+runner合成)
│   ├── run_momentum_strategy       │   ├── _pipeline.py     (共通: position→Returns/Strategy_Returns)
│   ├── run_rsi_strategy            │   ├── sma_crossover.py (compute_signal のみ)
│   ├── STRATEGIES dict             │   ├── mean_reversion.py
│   └── run_backtest + CLI          │   ├── momentum.py
├── metrics.py                      │   └── rsi.py
└── strategy_harness.py             ├── metrics.py
                                    └── strategy_harness.py
```

## 設計の要点

1. **シグナル計算と リターン計算の分離**: 各戦略モジュールは
   `compute_signal(df, **params) -> (df, position_col)` だけを持つ。
   `Returns` / `Strategy_Returns` の付与（1日ラグ）は `_pipeline.attach_returns` に
   一本化し、戦略側が共通規約から逸脱できない構造にする。
   これは LLM 生成戦略の契約（`generate_signals` がシグナルのみ返し、
   信頼済み harness がリターン・コスト・メトリクスを計算する）と同じ分離を
   ビルトイン戦略にも適用したもの。
2. **レジストリ**: `strategies/__init__.py` が `compute_signal` + 共通パイプラインから
   runner を合成して `STRATEGIES` dict を作る。戦略の追加 = モジュール1個 +
   `_MODULES` への1行。
3. **エンジンはファサード**: `backtest_engine` は従来の関数名
   (`run_*_strategy`, `STRATEGIES`) を再エクスポートする。外部契約
   (CLI・コンテナENTRYPOINT・テスト・autonomous_loop のsubprocess呼び出し) は不変。
   既存テストは**無改変**で通る。
4. **momentum の特殊性を明示化**: momentum のみ約定ポジションが `Position`
   (Signal の rolling 平均) で、コスト・trade数は生 `Signal` から計算される。
   従来は関数内に埋まっていた非対称性が、`position_col` の返り値として型に現れる。

## ハマりどころ (今後の実装者向け)

- **.gitignore の `strategies/`**: repoルートの実行時生成物 (LLM戦略ダンプ) 用の
  ignore が `backtests/strategies/` にもマッチする。`!backtests/strategies/` の
  negation を追加済み。
- **dual import**: コンテナは `backtest_engine.py` をスクリプト実行するため
  パッケージ相対importが使えない。metrics と同じ try/except パターンで
  `from .strategies import ...` / `from strategies import ...` を両対応。
  (fix/script-mode-imports PR #14 と同種の罠)
- **デプロイ時はイメージ再ビルドが必要**: Dockerfile の `COPY backtests/ /backtest/`
  に strategies/ が含まれるため、`sandbox-alpha-backtest:latest` の再ビルドを忘れると
  コンテナ内は旧コードのまま。

## 挙動不変の検証 (エビデンス)

検証ハーネス: `scripts/verify_strategy_equivalence.py`
(シード固定の合成OHLCV 2銘柄 × 4戦略 × 3モード(walkforward/カスタムparams/
full-sample+metrics-since) = 24ケースをエンジンCLI経由で実行し、
sorted-key正規化JSONで保存 → `diff -r` で突き合わせ)

2026-07-20 実施結果 (main 3d22290 vs 本ブランチ):

| チェック | 結果 |
|---|---|
| 正規化出力 diff (24ケース) | **0件 — 全ケースバイト単位で一致** |
| pytest (テストコード無改変) | 496 passed — main と同一 (既知の pre-existing fail 1件 `test_ideation_v3::test_prompt_has_expert_catalog` も main と同一) |
| import両コンテキスト | pytest=パッケージimport / CLI=スクリプトimport の両経路で実行済み |

## このブランチの扱い

設計確認が目的。本採用する場合は:
1. この差分をそのまま PR 化する (検証ハーネスごと)、または hermes に委譲して再実装
2. マージ後にバックテストイメージを再ビルド
3. 以後の戦略リファクタ・戦略追加時は同ハーネスで before/after を取るのを定型化する
   (「説明できない差分ゼロ」をゲート条件にする)
