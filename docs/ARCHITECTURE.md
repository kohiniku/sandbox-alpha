# アーキテクチャ設計

## システム全体図

```
┌──────────────────────────────────────────────────────────────────┐
│                        User / Scheduler                           │
│                    (cron job / manual trigger)                    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    autonomous_loop.py                             │
│                    (メインオーケストレーター)                       │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Hypothesis │  │  Backtest   │  │  Evaluator  │             │
│  │  Generator  │─▶│  Executor   │─▶│   & Judge   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         ▲                                       │                │
│         │                                       ▼                │
│  ┌─────────────┐                        ┌─────────────┐         │
│  │  Knowledge  │◀───────────────────────│   Result    │         │
│  │    Base     │                        │   Writer    │         │
│  └─────────────┘                        └─────────────┘         │
└──────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    backtest_engine.py                             │
│                    (サブプロセスで実行)                             │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Data     │  │  Strategy   │  │  Metrics    │             │
│  │  Fetcher    │─▶│   Engine    │─▶│ Calculator  │             │
│  │ (yfinance)  │  │             │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

---

## コンポーネント詳細

### 1. autonomous_loop.py（メインオーケストレーター）

**責務**: PDCAループの制御

**入力**:
- イテレーション数（コマンドライン引数）

**出力**:
- ターミナルログ（進捗表示）
- knowledge.json（ナレッジベース更新）
- results/*.json（各テスト結果）

**主要関数**:

```python
def load_knowledge() -> dict
    """ナレッジベースを読み込む"""

def save_knowledge(knowledge: dict) -> None
    """ナレッジベースを保存"""

def generate_hypothesis(knowledge: dict) -> dict
    """仮説を生成（ランダム or ナレッジ参照）"""

def run_backtest(hypothesis: dict) -> dict
    """subprocessでバックテストを実行"""

def evaluate_result(hypothesis: dict, result: dict) -> tuple[str, dict]
    """結果を評価し、採否を判定"""

def save_result(hypothesis: dict, result: dict, verdict: str, evaluation: dict) -> dict
    """結果をJSONファイルに保存"""

def run_loop(num_iterations: int) -> dict
    """メインPDCAループ"""
```

---

### 2. backtest_engine.py（バックテストエンジン）

**責務**: 単一戦略のバックテスト実行

**入力**:
- 戦略名（コマンドライン引数）
- 銘柄シンボル
- パラメータ

**出力**:
- JSON（stdout）: パフォーマンスメトリクス

**主要関数**:

```python
def fetch_data(symbol: str, period: str = "2y") -> pd.DataFrame
    """yfinanceからOHLCVデータを取得"""

def calculate_metrics(returns: pd.Series) -> dict
    """パフォーマンスメトリクスを計算"""

def run_sma_crossover_strategy(df: pd.DataFrame, fast_window: int, slow_window: int) -> dict
    """SMAクロスオーバー戦略を実行"""

def run_mean_reversion_strategy(df: pd.DataFrame, window: int, threshold: float) -> dict
    """平均回帰戦略を実行"""

def run_momentum_strategy(df: pd.DataFrame, lookback: int, hold_period: int) -> dict
    """モメンタム戦略を実行"""
```

**設計上の注意点**:
- stdoutにはJSONのみを出力（ログはstderr）
- タイムアウト120秒
- エラー時は `{"error": "..."}` を返す

---

### 3. knowledge.json（ナレッジベース）

**構造**:

```json
{
  "tested": ["hyp_1234", "hyp_5678", ...],
  "adopted": [
    {
      "hypothesis": {...},
      "backtest_result": {...},
      "evaluation": {...},
      "verdict": "adopted",
      "tested_at": "2026-07-17T16:38:12.345678"
    },
    ...
  ],
  "rejected": [...],
  "iterations": 10
}
```

**用途**:
- テスト済み仮説の重複防止
- 成功パターンの分析
- 次の仮説生成の参考

---

### 4. results/*.json（個別テスト結果）

**ファイル名**: `hyp_{timestamp}_{random}.json`

**構造**:

```json
{
  "hypothesis": {
    "id": "hyp_1784306292_3390",
    "strategy": "mean_reversion",
    "symbol": "NVDA",
    "params": {"window": 23, "threshold": 2.0},
    "description": "平均回帰 on NVDA",
    "generated_at": "2026-07-17T16:38:12.345678"
  },
  "backtest_result": {
    "total_return_pct": 26.4,
    "sharpe_ratio": 0.85,
    "max_drawdown_pct": -17.4,
    "num_trades": 500,
    "avg_daily_return_pct": 0.0526,
    "strategy": "mean_reversion",
    "params": {"window": 23, "threshold": 2.0},
    "symbol": "NVDA",
    "data_points": 502,
    "date_range": "2024-07-17 to 2026-07-17"
  },
  "evaluation": {
    "verdict": "adopted",
    "sharpe_ratio": 0.85,
    "total_return_pct": 26.4,
    "max_drawdown_pct": -17.4,
    "reasons": [
      "✅ Sharpe 0.85 >= 0.3",
      "✅ Return 26.4% >= 5.0%",
      "✅ Drawdown -17.4% >= -25.0%"
    ]
  },
  "verdict": "adopted",
  "tested_at": "2026-07-17T16:38:12.345678"
}
```

---

## データフロー

### 1イテレーションの流れ

```
1. load_knowledge()
   └─▶ knowledge.json を読み込む

2. generate_hypothesis(knowledge)
   ├─▶ 戦略テンプレートからランダム選択
   ├─▶ 銘柄プールからランダム選択
   ├─▶ パラメータ空間からランダムサンプリング
   └─▶ hypothesis dict を返す

3. run_backtest(hypothesis)
   ├─▶ subprocess を起動
   ├─▶ backtest_engine.py を実行
   ├─▶ stdout から JSON をパース
   └─▶ result dict を返す

4. evaluate_result(hypothesis, result)
   ├─▶ 閾値チェック（Sharpe, Return, Drawdown）
   ├─▶ 判定理由を生成
   └─▶ (verdict, evaluation) を返す

5. save_result(hypothesis, result, verdict, evaluation)
   └─▶ results/hyp_*.json に書き込む

6. knowledge を更新
   ├─▶ adopted or rejected リストに追加
   ├─▶ tested リストに追加
   └─▶ iterations をインクリメント

7. save_knowledge(knowledge)
   └─▶ knowledge.json に書き込む
```

---

## エラーハンドリング

### タイムアウト

```python
try:
    result = subprocess.run(cmd, timeout=120, capture_output=True)
except subprocess.TimeoutExpired:
    return {"error": "Timeout (120s)"}
```

### JSONパースエラー

```python
try:
    return json.loads(json_str)
except json.JSONDecodeError as e:
    return {"error": f"JSON parse error: {e}"}
```

### データ取得エラー

```python
if df.empty:
    return {"error": f"No data for {symbol}"}
```

---

## パフォーマンス最適化

### 現在のボトルネック

1. **データ取得**: yfinance API呼び出し（各銘柄2-3秒）
2. **直列実行**: 1イテレーションずつ順次実行

### 将来の最適化

#### 1. データキャッシュ

```python
# ローカルにOHLCVデータをキャッシュ
cache_dir = Path("./data_cache")
cache_file = cache_dir / f"{symbol}.parquet"

if cache_file.exists():
    df = pd.read_parquet(cache_file)
else:
    df = fetch_data(symbol)
    df.to_parquet(cache_file)
```

#### 2. 並列実行

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_backtest, hyp) for hyp in hypotheses]
    results = [f.result() for f in futures]
```

#### 3. サンドボックス隔離並列（Trusted Runner）

```python
# 各イテレーションを独立した隔離コンテナで実行
# エージェントは戦略パラメータ（データ）のみを信頼済み
# sandbox-runner にHTTP送信。Dockerアクセスは持たない。
export SANDBOX_RUNNER_URL="http://sandbox-runner:8080"
python3 autonomous_loop.py 10
# → run_backtest() が urllib.request で POST /run を呼び出す
```

---

## Future Work: Sandbox Isolation

### 設計原則

**DockerソケットをLLM駆動エージェントに一切渡さない。** Dockerソケットへのアクセス（raw でも docker-socket-proxy 経由でも）は、コンテナ作成ペイロードによるバインドマウント脱出（bind-mount escape）が可能であるため、実質的にホスト root 権限と同等である。エンドポイントレベルのプロキシではマウント指定を確実にフィルタリングできず、安全性を保証できない。

このため、アーキテクチャは以下の **Trusted Runner** パターンを採用する：

```
┌──────────────────────────────────────────────────┐
│               hermes コンテナ                      │
│                                                  │
│  autonomous_loop.py                              │
│       │                                          │
│       │ SANDBOX_RUNNER_URL 環境変数               │
│       │ HTTP POST (strategy, symbol, params)      │
│       │ JSON only — no socket, no image spec      │
│       ▼                                          │
└───────┬──────────────────────────────────────────┘
        │
        │ ネットワーク越しのHTTP API
        │
        ▼
┌──────────────────────────────────────────────────┐
│          sandbox-runner（信頼済みサービス）         │
│          ※ エージェントとは別プロセス・別権限       │
│                                                  │
│  - Docker アクセスを単独で保持                      │
│  - 受信するのは strategy / symbol / params のみ    │
│  - 固定イメージを起動（任意イメージの pull 禁止）   │
│  - 起動オプションはハードコード:                    │
│      --network=none                               │
│      --read-only                                  │
│      --cap-drop=ALL                               │
│      --memory=512m --cpus=1                       │
│      --tmpfs /tmp:size=64M,noexec                 │
│  - シークレット・ボリュームマウントなし               │
│  - 事前取得データは読み取り専用でマウント             │
│  - タイムアウト: 180秒                              │
│  - レスポンス: stdout JSON をそのまま返す            │
└──────────────────────────────────────────────────┘
```

### Runner API 契約

```
POST {SANDBOX_RUNNER_URL}/run
Content-Type: application/json

{
  "strategy": "sma_crossover|mean_reversion|momentum|rsi",
  "symbol": "AAPL",
  "params": {"fast_window": 10, "slow_window": 30}
}

→ 200: バックテスト結果 JSON（metrics with in_sample/out_of_sample）
→ 4xx: バリデーションエラー
→ 5xx: 実行時エラー
→ タイムアウト: 180秒
```

### クライアント側の実装（autonomous_loop.py）

`run_backtest()` は環境変数 `SANDBOX_RUNNER_URL` が設定されている場合、`urllib.request`（stdlibのみ、依存なし）で runner に POST し、レスポンス JSON を既存の subprocess パスと同一形状で返す。未設定時は従来の subprocess / venv パスをそのまま使用する。

エラーハンドリング：
- 接続エラー → `{"error": "Sandbox runner connection error: …"}`
- タイムアウト → `{"error": "Sandbox runner connection error: …"}`
- HTTP 4xx/5xx → `{"error": "Sandbox runner HTTP {code}: …"}`
- JSON パース失敗 → `{"error": "Sandbox runner JSON parse error: …"}`

### 現在のセキュリティ対策（サンドボックス未使用時）

- venv隔離（依存関係の衝突防止）
- subprocessタイムアウト（無限ループ防止、120秒）
- エラー時のグレースフルデグラデーション
- 固定テンプレートのみ実行（任意コード実行なし）

---

## スケーラビリティ

### 現在の制約

- 単一マシンでの実行
- 直列処理
- ローカルファイルベースのナレッジ

### 将来の拡張

#### 分散実行

```python
# Kubernetes Job で並列実行
kubectl create job alpha-search-1 --image=alpha-sandbox:v1 -- mean_reversion NVDA 23 2.0
```

#### 集中ナレッジベース

```python
# Redis or PostgreSQL でナレッジを共有
redis_client.hset("knowledge:adopted", hyp_id, json.dumps(record))
```

#### ストリーミング結果

```python
# WebSocket でリアルタイム進捗表示
websocket.send(json.dumps({"iteration": i, "verdict": verdict}))
```

---

## モニタリングとロギング

### 現在のログ

- ターミナル出力（進捗表示）
- JSONファイル（結果保存）

### 将来の拡張

- 構造化ログ（structlog）
- メトリクス収集（Prometheus）
- ダッシュボード（Grafana）
- アラート（Slack通知）

---

## テスト戦略

### ユニットテスト

```python
def test_sma_crossover_strategy():
    df = generate_mock_data()
    result = run_sma_crossover_strategy(df, fast_window=10, slow_window=30)
    assert "sharpe_ratio" in result
    assert "total_return_pct" in result

def test_evaluate_result():
    hypothesis = {...}
    result = {"sharpe_ratio": 0.5, "total_return_pct": 10.0, "max_drawdown_pct": -15.0}
    verdict, evaluation = evaluate_result(hypothesis, result)
    assert verdict == "adopted"
```

### 統合テスト

```python
def test_full_loop():
    knowledge = run_loop(num_iterations=3)
    assert knowledge["iterations"] == 3
    assert len(knowledge["tested"]) == 3
```

---

## デプロイメント

### ローカル実行

```bash
cd sandbox-alpha
source .venv/bin/activate
python3 autonomous_loop.py 10
```

### Docker実行（Trusted Runner経由）

```bash
# sandbox-runner が固定イメージを起動（--network=none, --read-only, --cap-drop=ALL）
export SANDBOX_RUNNER_URL="http://localhost:8080"
python3 autonomous_loop.py 5
```

### クラウド実行（将来）

```bash
# AWS Batch or GCP Cloud Run
gcloud run jobs execute alpha-search --region=asia-northeast1
```

---

## まとめ

本アーキテクチャは：

1. **モジュラー設計**: 各コンポーネントが独立
2. **拡張性**: 新しい戦略・メトリクスを簡単に追加可能
3. **堅牢性**: エラーハンドリングとタイムアウト
4. **スケーラブル**: 将来的に分散実行に対応可能

**次のステップ**:
- **Trusted Runner サンドボックス（実装済み）**: `SANDBOX_RUNNER_URL` 環境変数による隔離実行基盤。LLM生成コードの安全な実行を可能にする。
- Polymarket API統合
- LLM仮説生成（サンドボックス隔離実装後に導入）
