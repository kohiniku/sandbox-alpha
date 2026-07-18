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

#### 3. Docker並列

```python
# 各イテレーションを独立コンテナで実行
docker run --rm alpha-sandbox:v1 mean_reversion NVDA 23 2.0
```

---

## セキュリティ考慮事項

### 現在の対策

- venv隔離（依存関係の衝突防止）
- subprocessタイムアウト（無限ループ防止）
- エラー時のグレースフルデグラデーション

### 将来の対策

- Docker完全隔離（ネットワーク制限、リソース制限）
- サンドボックス内でのみ実行可能なコード検証
- 悪意のある戦略コードの検出（静的解析）

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

### Docker実行

```bash
docker build -t alpha-sandbox:v1 .
docker run --rm -v $(pwd)/results:/results alpha-sandbox:v1 10
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
- Polymarket API統合
- LLM仮説生成
- Docker完全隔離
