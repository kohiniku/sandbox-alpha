# sandbox-alpha: Cross-Sectional Strategy Expansion Plan

Status: DRAFT — Hermes planning deliverable. No code changes.

---

## 0. Executive Summary

Expand sandbox-alpha from ~5 single-name backtests to 100–500 name cross-sectional strategies. Statistical power scales roughly with sqrt(N) — moving from 10 names to 300 names increases the effective sample by ~5.5x, so edges that survive the validation gate are more likely real. This document covers scoping, design decisions, and PR sequencing. Execution remains gated behind an upcoming walk-forward CV / bootstrap gating plan (not in scope here).

---

## 1. Current-State Citation

### 1.1 Strategy Contract (Single-Symbol)

- **Built-in strategies**: `backtests/strategies/__init__.py:22-44`. Each module exposes `NAME` + `compute_signal(df) -> (df, position_col)`. The shared pipeline `backtests/strategies/_pipeline.py:10-18` (`attach_returns`) converts positions to `Strategy_Returns`.
- **LLM-generated code**: `backtests/strategy_harness.py:167-187` (`exec_and_extract`) expects `generate_signals(df) -> pd.Series` with values in {-1, 0, 1}.
- **Signal contract enforcement**: `backtests/strategy_harness.py:190-212` (`_call_signals`) validates return type, value domain, and index alignment.

### 1.2 Backtest Engine

- `backtests/backtest_engine.py:79-170` (`run_backtest`) — single symbol, 60/20/20 walk-forward split, metrics via `metrics.py`.
- `backtests/metrics.py:48-61` (`apply_trading_cost`) — single-symbol cost: `COST_BPS` per position change. This model assumes one position flips at a time — inadequate for portfolio turnover.

### 1.3 Runner Endpoints

- `POST /run` — `autonomous_loop.py:383-388`: `{strategy, symbol, params}` → returns `{in_sample, out_of_sample, holdout}`. Used by param-type entries.
- `POST /run_code` — `autonomous_loop.py:1174-1178`: `{code_b64, symbol}` → runs `strategy_harness.py`. Used by code-type entries.
- `POST /run_manifest` — `autonomous_loop.py:1138`: `{<full manifest spec>}` → runs `manifest_runner.py`. Already multi-symbol capable via OhlcvSource universe.
- `POST /validate` — referenced in `strategy_harness.py:289-344` (`run_preflight`). Single-symbol synthetic data only (line 318: `build_synthetic_df()`).

All runner endpoints are Claude-owned (trust boundary). The runner lives at `/home/kohiniku/hermes-secure/sandbox-runner`. Hermes code only POSTs to it.

### 1.4 Backtest Docker Image

- `Dockerfile:1-13`: copies `backtests/`, `manifest.py`, `manifest_runner.py`, `data_adapters/`, `evaluators/`. Available libraries: pandas, numpy, scipy, scikit-learn, statsmodels.
- Container runs as unprivileged `sandbox` user with `--network=none --read-only --cap-drop=ALL`.
- Entrypoint: `backtest_engine.py` (single-symbol path). For multi-symbol, `manifest_runner.py` is copied and callable.

### 1.5 Near-Miss / Deflation Infrastructure

- `autonomous_loop.py:690-764` (`_classify_near_miss`, `_record_near_miss`): records near-misses keyed by `_family_key(strategy, symbol)` at line 164. A panel-family near-miss would need a separate key space — otherwise cross-sectional failures inflate single-name deflation and vice versa.
- `autonomous_loop.py:73-77` (`compute_effective_min_sharpe`): `sqrt(2*log(N))` deflation uses N_family = count of same `(strategy, symbol)` combos (line 572-576). This formula still works for portfolio Sharpe but the N_family counting must be separated.

---

## 2. Universe Recommendation

### Recommendation: Russell 1000 (primary) + Russell 3000 as optional superset

**Statistical power**: 300–1000 cross-section per day vs. current 1 observation/day/symbol. For a 5-year window (~1250 trading days), Russell 1000 yields ~1,000,000 asset-day observations vs. current ~5,000. Even after accounting for cross-sectional correlation (effective N < nominal N), power increases dramatically.

**Thesis fit**: The user's research axis is "alternative data × Japanese market × statistical/ML methods." Given:
- Japanese market data sources are more constrained (J-Quants free tier is limited, Bloomberg is university-terminal only)
- US market has free, deep-data coverage (yfinance, Stooq, Tiingo)
- Cross-sectional methodology is transferable — validate on US, replicate on JP when data access expands

**Recommendation: Paired US + JP universe**:
- Primary: Russell 1000 constituents (~1000 names, free via Wikipedia/FTSE Russell public list)
- JP track: TOPIX 500 or Nikkei 225 via J-Quants free tier or Stooq (lower priority, feasibility TBD)
- Implement the US path first; JP path is PR 4f (parallel universe adapter)

**Primary data source**: yfinance via rate-limited bulk download script on hermes side (not runner — runner has no network). Cached as per-symbol CSV in the existing data_dir convention.

**Backup source**: Stooq (no auth required, daily CSV) for symbols yfinance fails on.

### Data Footprint Estimate

Assumptions: 1000 symbols × 1250 trading days (5y) × 5 columns (OHLCV) × 8 bytes (float64)

- Raw in-memory: 1000 × 1250 × 5 × 8 = 50 MB
- Panel with MultiIndex: ~100–200 MB overhead (Date × symbol index, wide-format alignment)
- Cached CSV: ~100–200 MB per universe (1000 symbols × 5y)
- Full pipeline: ~500 MB working set (returns + signals + weights intermediate arrays)
- **Recommendation**: 2 GB minimum for comfortable operation; 4 GB for Russell 3000 (3000 symbols).

---

## 3. PR Breakdown

### PR 4a: Universe Definition + Data Layer

**Goal**: Define a universe, fetch & cache OHLCV for all constituents, and expose a `load_universe_panel()` adapter.

**Files to create/modify**:
- `data_adapters/universe.py` — `UniverseProvider` class that loads constituent lists, fetches bulk CSVs
- `data_adapters/panel_loader.py` — `load_panel(universe, start, end, data_dir) -> pd.DataFrame` with MultiIndex (Date, symbol) or dict of DataFrames
- `scripts/fetch_universe.py` — rate-limited bulk downloader (hermes-side, cron runs this)
- `data_adapters/ohlcv.py:50-228` — extend `align_universe` with chunked loading, date-range filtering, forward-fill policy for panel data
- `tests/test_panel_loader.py`

**Key decisions**:
- Cache format: per-symbol CSV (existing convention), plus a `.universe_manifest.json` listing symbols + date ranges
- Rate limiting: yfinance allows ~2000 requests/hour unauthenticated. For 1000 symbols = 30 min fetch time. Script retries with exponential backoff.
- Survivorship tracking: universe manifest includes `as_of` date; delisted symbols preserved in separate `delisted/` subdir (see Risk 5.1)

**Runner impact**: None (data fetching is hermes-side). Runner reads cached CSVs via `data_dir` mount.

---

### PR 4b: Cross-Sectional Strategy Contract

**Goal**: Define the interface cross-sectional strategies must implement.

**Recommendation: Unified panel-first with single-symbol as degenerate case**

Two competing approaches:

| | Option A: New parallel interface | Option B: Panel-first unified |
|---|---|---|
| New function | `compute_cross_signal(panel) -> weights[date, symbol]` | Extend `compute_signal` to accept panel |
| Backward compat | Full — old interface untouched | Requires adapter for single-symbol |
| LLM simplicity | LLM must choose which interface | One interface, simpler prompt |
| Code duplication | Two signal contracts to maintain | Shared validation |
| Migration cost | Low initially, high over time | Higher once, low ongoing |

**Decision: Option B (panel-first)**. Reasoning: the single-symbol `generate_signals(df)` is already a special case of panel computation (universe size = 1). A panel-first interface with explicit universe size awareness eliminates duplication.

**Proposed interface**:

```python
def compute_cross_signal(
    panel: dict,                    # {symbol: pd.DataFrame} with aligned indices
    universe: list[str],
    extras: dict | None = None      # optional: benchmark returns, factor data
) -> pd.DataFrame:                  # index=date, columns=symbol, values=weights
```

Three return conventions (strategy declares which):
1. **weights** (continuous): DataFrame of portfolio weights, sum to 1 (long-only) or 0 (L/S)
2. **signals** (discrete): DataFrame of {-1, 0, 1}, aggregated to weights by engine
3. **scores** (ranking): DataFrame of raw scores, engine applies cross-sectional transform (z-score, quantile)

**Single-symbol backward compat**: `compute_signal(df) -> (df, position_col)` remains untouched. A thin adapter `_panel_from_single(compute_signal_fn, panel, universe) -> weights` wraps single-name strategies into the panel interface automatically.

**Files to create/modify**:
- `backtests/strategies/cross_sectional/__init__.py` — new package, `CROSS_SECTIONAL_STRATEGIES` registry
- `backtests/strategies/cross_sectional/_contract.py` — `validate_weights()`, `validate_signals()`, `validate_scores()`
- `backtests/strategies/__init__.py:22-44` — add import + auto-wrap for single-name strategies
- `tests/test_cross_sectional_contract.py`

**Runner impact**: REQUIRES CLAUDE — runner's `manifest_runner.py` already handles `generate_signals` and `generate_weights` (line 80-96: `_call_with_extras`). Need to add `generate_cross_signal` as recognized entrypoint alongside existing ones. The runner dispatch table (`manifest_runner.py:289-300` area) must check for this new function name.

---

### PR 4c: Cross-Sectional Backtest Engine

**Goal**: Portfolio construction, rebalancing, cost model, daily PnL aggregation, and portfolio-level Sharpe.

**Portfolio construction modes** (strategy declares in manifest):

| Mode | Input | Construction | Use case |
|---|---|---|---|
| `top_k` | scores/weights, k | Long top k equally; short bottom k (if L/S) | Factor tilts |
| `quintile_ls` | scores | Long top quintile, short bottom quintile, equal weight within | Academic factor replication |
| `zscore_continuous` | scores | `w = (zscore - threshold).clip(...)` normalized to sum 1 | Modern ML signals |
| `custom_weights` | weights DataFrame | Directly use returned weights (engine only validates) | RL/optimization strategies |

**Rebalance cadence**: Configurable (daily, weekly, monthly). Default: monthly to keep turnover manageable. Daily rebalancing on 500 names = ~125,000 rebalance events/year.

**Transaction cost model** (REQUIRES CLAUDE — runner must implement):
```
C_t = sum_i |w_{i,t} - w_{i,t-1}| × cost_bps_i / 10000
```
Where `cost_bps_i` is per-symbol (e.g. 10 bps for small-cap, 3 bps for large-cap). Default: uniform 5 bps.

For 500-name daily rebalance with mean absolute weight change of 0.002/symbol: turnover = 500 × 0.002 × 2 = 2.0 (200% daily). At 5 bps: cost = 10 bps/day = 25%/year — this MUST be modeled or all strategies will look terrible. **Mitigation**: default to monthly rebalance, introduce turnover constraint in evaluation gate.

**Portfolio Sharpe** (hermes-side computation):
```python
portfolio_returns = (weights.shift(1) * asset_returns).sum(axis=1)
portfolio_returns_net = portfolio_returns - cost_series
sharpe = portfolio_returns_net.mean() / portfolio_returns_net.std() * sqrt(252)
```

**Files to create/modify**:
- `backtests/cross_sectional_engine.py` — `run_cross_sectional_backtest(panel, strategy_fn, config) -> result_dict`
- `backtests/cross_metrics.py` — portfolio-level: Sharpe, IR, turnover, CVaR, max drawdown, hit rate per period
- `tests/test_cross_engine.py`
- `tests/test_cross_metrics.py`

**Runner impact**: REQUIRES CLAUDE — new endpoint `POST /run_cross_sectional` or extend `/run_manifest` to recognize cross-sectional evaluator type. The runner already has `_signals_to_weights` in `manifest_runner.py:110-130` — this is the foundation for the aggregation layer. Add:
1. Cost model with per-symbol bps
2. Rebalance calendar logic
3. Portfolio-level aggregation
4. Return shape: `{train_metrics, val_metrics, holdout_metrics}` with portfolio Sharpe + turnover as primary metrics

---

### PR 4d: Registry & Strategy Package Migration

**Goal**: New package layout for cross-sectional families + migration policy for existing built-ins.

**Package layout**:
```
backtests/strategies/
├── __init__.py              # STRATEGIES + CROSS_SECTIONAL_STRATEGIES registries
├── _pipeline.py             # unchanged: attach_returns (single-name)
├── _single_name/            # RENAMED: existing built-ins moved here
│   ├── __init__.py
│   ├── sma_crossover.py
│   ├── mean_reversion.py
│   ├── momentum.py
│   └── rsi.py
├── cross_sectional/         # NEW: panel-family strategies
│   ├── __init__.py          # CROSS_SECTIONAL_STRATEGIES registry
│   ├── _contract.py         # validation functions
│   ├── xs_value.py          # cross-sectional value (B/P, E/P)
│   ├── xs_momentum.py       # cross-sectional momentum (12-1)
│   ├── xs_low_vol.py        # cross-sectional low volatility
│   └── xs_quality.py        # cross-sectional quality (ROE, accruals)
└── _panel_adapter.py        # wraps single-name -> panel
```

**Migration policy**: Dual-run. Existing strategies continue to work from `_single_name/` with backward-compatible imports. `__init__.py` re-exports old names. The adapter `_panel_adapter.py` automatically makes any `compute_signal` strategy available as a panel strategy (runs per-symbol then stacks). After 2 release cycles, deprecate direct single-name access but keep adapter.

**Files to modify**:
- `backtests/strategies/__init__.py:22-44` — restructure imports
- Move 4 existing strategy files into `_single_name/`
- Create `_panel_adapter.py`
- `autonomous_loop.py:34-64` (`STRATEGY_TEMPLATES`) — add cross-sectional templates
- `autonomous_loop.py:164` (`_family_key`) — extend to include family type discriminator

**Family key change**: `_family_key` must become `_family_key(strategy, symbol, family_type)` with `family_type in {"single", "cross"}`. This prevents near-miss leakage between single-name and cross-sectional families.

---

### PR 4e: Ideation-v2 Prompt Upgrade

**Goal**: LLM ideation pipeline must understand both single-name and cross-sectional interfaces, when each applies, and the available expert catalog.

**Concrete edits to `strategy_ideation.py`**:

1. **`_build_prompt()` (line 407-495)**: Add cross-sectional context section after the single-name contract:
   - Cross-sectional interface contract (panel input, weights/signals/scores output)
   - When to use each: cross-sectional when thesis is about relative ranking; single-name when thesis is absolute timing
   - Universe context: "you have access to Russell 1000 constituents; think in terms of cross-sectional factors"

2. **`STRATEGY_TEMPLATES` in `autonomous_loop.py:34-64`**: Add cross-sectional template block:
```python
"cross_sectional": {
    "description": "Cross-sectional factor (panel-based)",
    "sub_families": ["xs_value", "xs_momentum", "xs_low_vol", "xs_quality", "xs_size"],
    "param_space": {
        "top_k": [10, 20, 50, 100],
        "rebalance": ["monthly", "weekly"],
        "weighting": ["equal", "value_weighted"]
    }
}
```

3. **Expert catalog expansion** (`_EXPERT_MODE_CATALOG`, line 382-404): Add:
   - Cross-sectional regression (Fama-MacBeth via `statsmodels`)
   - Portfolio sorting (univariate, bivariate, conditional)
   - Risk-based (minimum variance, risk parity via `scipy.optimize`)
   - Factor timing (dynamic allocation across style factors)

4. **`_PROPOSAL_JSON_SCHEMA` (line 306-332)**: Add `"type": "cross_sectional"` proposal variant specifying universe, factor family, construction method.

5. **Backlog compatibility**: `backlog.py` must accept `type: "cross_sectional"` entries alongside existing `param`, `code`, `manifest`.

**Files to modify**:
- `strategy_ideation.py:306-495` — prompt construction
- `autonomous_loop.py:34-64` — STRATEGY_TEMPLATES
- `autonomous_loop.py:382-404` — _EXPERT_MODE_CATALOG
- `backlog.py` — accept cross_sectional type (if backlog schema is enforced)
- `tests/test_ideation.py`, `tests/test_ideation_v3.py` — add cross-sectional proposal tests

**Runner impact**: None (ideation is hermes-side only).

---

### PR 4f: Preflight — Synthetic Panel Input (REQUIRES CLAUDE)

**Goal**: `/validate` endpoint must support cross-sectional strategies with synthetic panel data.

**Current state**: `strategy_harness.py:289-344` (`run_preflight`) uses `build_synthetic_df(n_days=250)` — single 250-row DataFrame. `manifest_runner.py` likely has its own preflight or delegates to harness.

**Required new schema**:

```json
{
  "validate_type": "cross_sectional",
  "code_b64": "<base64 strategy code>",
  "panel_shape": {
    "n_symbols": 100,
    "n_days": 250,
    "seed": 42
  }
}
```

Synthetic panel generation (REQUIRES CLAUDE — in runner):
- Generate `n_symbols` correlated random-walk price series (factor model: common factor + idiosyncratic noise)
- Default: 30% common factor correlation (realistic for US equities)
- Columns: Open, High, Low, Close, Volume per symbol
- Output: MultiIndex DataFrame (Date, Symbol) or dict of DataFrames

**Return shape**: Same as current preflight — `{valid: true/false, n_signals: N}` — but `n_signals` is now total non-zero weight entries across all dates × symbols.

**Runner endpoint**: Extend `POST /validate` to accept `validate_type` discriminator. When `validate_type == "cross_sectional"`, route to new synthetic panel path.

**Hermes-side**: `autonomous_loop.py` code-preflight path must pass `validate_type` when the backlog entry is cross-sectional.

---

## 4. Backward Compatibility

| Concern | Mitigation |
|---|---|
| Single-name strategies stop working | Adapter in `_panel_adapter.py` wraps any `compute_signal` into panel interface. Old import paths preserved via `__init__.py` re-exports. |
| knowledge.json schema breaks | Add `"family_type"` field to entries. Old entries default to `"single"`. New entries set explicitly. Loader migration is additive. |
| Existing cron jobs break | No cron changes until Phase 1 complete. New endpoints additive; old `/run` and `/run_code` unchanged. |
| Deflation counter leakage | `_family_key` extended to `(strategy, symbol, family_type)`. Cross-sectional near-misses tracked in separate `near_misses_cross` list in knowledge. `compute_effective_min_sharpe` counts per-family-type N. |

### Deflation Counter Separation

`knowledge.json` gains:
```json
{
  "families": {
    "sma_crossover|AAPL": {"family_type": "single", "n_trials": 5, ...},
    "xs_momentum|universe:abc12345": {"family_type": "cross", "n_trials": 3, ...}
  },
  "near_misses": [...],
  "near_misses_cross": [...]
}
```

`compute_effective_min_sharpe` already takes `N_family` as parameter — caller must pass the correct count for the family type. No formula change needed; only the counting must be partitioned.

---

## 5. Impact Analysis

### 5.1 Preflight Synthetic Data Generator

- Single-symbol: unchanged (`build_synthetic_df` in `strategy_harness.py:267-286`)
- Cross-sectional: new synthetic panel generator (REQUIRES CLAUDE, see PR 4f)

### 5.2 Evaluation Gate

Portfolio Sharpe is the natural primary metric for cross-sectional strategies. The existing gate structure (`_eval_val_gate` + `_eval_holdout_gate` in `autonomous_loop.py:497-539`) translates directly:
- `val_sharpe` → portfolio Sharpe on validation split
- `holdout_sharpe` → portfolio Sharpe on holdout split
- Additional gate: `max_turnover_daily` (e.g., < 1.0) to catch pathological daily-rebalance cost

The CV/bootstrap gating plan (separate dispatch) should consume portfolio Sharpe as the test statistic.

### 5.3 OOS Monitor (`oos_monitor.py`)

Current code (line 77-133, `run_oos_check`) runs `run_backtest(hyp, metrics_since=...)` — single-symbol only. For cross-sectional, the OOS flow must:
1. Re-fetch universe for the date range (hermes-side)
2. Call runner with updated data and `metrics_since`
3. Read portfolio-level `since_metrics`
4. Record `oos_portfolio_sharpe` instead of `oos_sharpe`

This requires a new runner endpoint or parameter (`POST /run_cross_sectional` with `metrics_since`). REQUIRES CLAUDE.

### 5.4 Cron Capacity Impact

| Dimension | Current (single) | Cross-sectional (500 names) | Factor |
|---|---|---|---|
| Data fetch per iteration | ~3s (1 symbol, cached) | ~30 min (initial fetch) / ~5s (cached) | 1–600x |
| Backtest compute per iteration | ~2s (single symbol, walk-forward) | ~30–120s (500 names, monthly rebalance, cost model) | 15–60x |
| Memory per iteration | ~50 MB | ~2 GB | 40x |
| Runner timeout needed | 180s | 300s | 1.7x |
| Iterations per hour (1 runner) | ~15–20 | ~2–5 | 4–10x fewer |

**Mitigation**: Cross-sectional iterations are heavier but each carries more statistical power. The autonomous loop should interleave single-name and cross-sectional iterations (e.g., 3 cross-sectional + 7 single-name per hour). Universe data is cached, so incremental updates are cheap after initial fetch.

The cron job should run universe-refresh as a separate pre-step, not inline with backtest iterations.

---

## 6. Risks & Required Mitigations

### 6.1 Survivorship Bias

**Risk**: Russell 1000 constituents as-of-today excludes names that delisted during the backtest window. Strategies that look good in backtests systematically exclude the worst performers.

**Mitigation**:
- Maintain `universe_manifest.json` with `as_of` dates. Each backtest run constrains universe to constituents that were in the index at that date.
- CRSP/Compustat delisted returns data is ideal but requires institutional access. Free alternative: archive.org snapshots of Russell constituent lists, or Wikipedia page history.
- Minimum: flag the bias explicitly in evaluation output. Track "universe date" in knowledge entries.

### 6.2 Corporate Actions

**Risk**: yfinance adjusted close handles splits/dividends but the engine uses raw OHLCV columns. Price discontinuities from splits will corrupt signal calculations.

**Mitigation**:
- Add `Adj Close` or `Close_adjusted` column to cached data. Use adjusted close for all signal computation, raw close only for position sizing.
- `data_adapters/ohlcv.py:47-228` already has `_canonicalize_columns` — extend to preserve/promote adjusted close.

### 6.3 Transaction Costs & Market Impact

**Risk**: 500-name daily rebalance with 5 bps/side = 500 × 5 × 2 × 250 / 10000 = 125% annual cost drag. Strategies that ignore costs will shine in backtest and fail in reality.

**Mitigation**:
- Default rebalance: monthly (not daily).
- Per-symbol cost tiers: large-cap 3 bps, mid-cap 7 bps, small-cap 15 bps (mapped from market cap decile).
- Turnover cap gate: strategies exceeding X% daily turnover fail validation.
- Market impact model (Phase 2, deferred): sqrt impact function based on daily volume participation.

### 6.4 Universe Drift

**Risk**: Russell 1000 reconstitutes annually (June). A 5-year backtest spans 5+ reconstitutions. Constituents enter and exit, changing the investable set.

**Mitigation**:
- `universe_manifest.json` tracks `valid_from` / `valid_until` per symbol. Backtest engine filters panel to symbols valid at each rebalance date.
- Gaps: symbols that exit mid-year are held until next rebalance or delisting (configurable).

### 6.5 Data Source Resilience

**Risk**: yfinance is unofficial, rate-limited, and occasionally broken. Depending on it as sole source for 1000 symbols is fragile.

**Mitigation**:
- Primary: yfinance (rate-limited bulk script with retries)
- Secondary: Stooq (no auth, daily CSV, `https://stooq.com/q/d/l/?s={symbol}&i=d`)
- Tertiary (paid, if needed): Tiingo free tier (1000 symbols/day, API key)
- Fallback: skip symbols unavailable from all sources; log and track coverage ratio
- J-Quants for JP: requires registration, ~500 symbols/day free tier. Feasibility TBD.

---

## 7. Open Questions for User/Claude Decision

1. **Rebalance frequency default**: Monthly is safer for costs but reduces signal frequency. Weekly is standard in academic factor literature. What should the default be? Recommend configurable with `monthly` as default.

2. **Benchmark for IR**: Cross-sectional strategies should be evaluated against equal-weight universe or cap-weight index (e.g., IWB for Russell 1000). Which benchmark should the engine compute IR against?

3. **JP universe priority**: The user's thesis is JP market-focused. Should PP 4a include TOPIX 500 data pipeline from the start, or defer to a follow-up PR? Deferring is pragmatic (J-Quants access TBD) but means all early cross-sectional validation is US-only.

4. **Runner capacity**: Multiple cross-sectional backtests in parallel will saturate the single sandbox-runner. Should we:
   - (a) Add a cross-sectional timeout budget separate from single-name?
   - (b) Implement backtest result caching (same universe + code = cached result)?
   - (c) Scale runner horizontally (multiple containers)?

5. **Knowledge file size**: 500-name × many iterations × full signal matrices stored in knowledge.json will explode file size. Should per-iteration signals be stored in a separate results directory (already done: `results/hyp_*.json`) and knowledge.json keep only summary statistics? Currently knowledge.json stores full entries — confirm this pattern.

6. **Existing manifest runner `/run_manifest` vs. new `/run_cross_sectional`**: The manifest runner already handles multi-symbol OHLCV (via `OhlcvSource.universe`). Should the cross-sectional backtest engine be:
   - (a) A new evaluator type `"cross_sectional"` inside the manifest runner?
   - (b) A separate endpoint `POST /run_cross_sectional`?
   - Recommendation: (a) — extend `evaluator.type` with `"cross_sectional"` that the runner routes to the new engine code. Fewer endpoints = simpler runner.

7. **Dual-listings**: Russell 1000 includes ADRs and dual-listed companies. Should the universe filter to US-primary-exchange only, or include all?

8. **Preflight compute budget**: Synthetic panel of 100 symbols × 250 days at preflight time. Is 30s a reasonable budget? Strategy code that needs >30s on synthetic data is suspicious anyway.

---

## 8. Runner-Side Change Summary (REQUIRES CLAUDE)

Every runner change enumerated for hand-implementation:

| # | Change | Endpoint affected | PR |
|---|---|---|---|
| 1 | Add `generate_cross_signal` as recognized entrypoint in manifest_runner dispatch | `/run_manifest` | 4b |
| 2 | Add portfolio construction engine (top-k, quintile L/S, zscore continuous, custom weights) | new: `/run_cross_sectional` or extend `/run_manifest` | 4c |
| 3 | Add portfolio-level transaction cost model with per-symbol bps tiers | same as above | 4c |
| 4 | Add rebalance calendar logic (daily/weekly/monthly) | same as above | 4c |
| 5 | Add portfolio PnL aggregation: `(weights.shift(1) × returns).sum(axis=1)` | same as above | 4c |
| 6 | Extend `/validate` to accept `validate_type: "cross_sectional"` with synthetic panel generation | `/validate` | 4f |
| 7 | Add `metrics_since` support to cross-sectional endpoint for OOS monitoring | `/run_cross_sectional` | 5.3 |
| 8 | Add evaluator type `"cross_sectional"` to manifest runner evaluator dispatch | `/run_manifest` | 4c |
| 9 | New return shape: `{train_metrics: {sharpe, ir, turnover, ...}, val_metrics: {...}, holdout_metrics: {...}, n_days: N}` | all cross-sectional endpoints | 4c |

All runner changes must follow existing trust boundaries: no network access in container, read-only data mount, sandbox user, cap-drop=ALL.

---

## 9. Suggested Sequencing

```
PR 4a (Data Layer) ──────┐
                          ├──► PR 4b (Contract) ──► PR 4c (Engine) ──► PR 4d (Registry)
                          │                                              │
                          └──► PR 4f (Preflight, REQUIRES CLAUDE) ◄─────┘
                                                                         │
                                                         PR 4e (Ideation v2)
```

Rationale:
- 4a and 4b can be developed in parallel (data layer and interface contract are independent)
- 4c depends on both 4a (needs real data to test) and 4b (needs contract to build against)
- 4d (registry migration) depends on 4c (engine must exist to register strategies)
- 4f (preflight) is Claude-owned and can run parallel to 4a-4d on the hermes side
- 4e (ideation) is last — LLM prompts should be updated after the infrastructure is stable

---

## 10. Verification Checklist

Before considering this plan complete, confirm:

- [ ] Plan reviewed against current codebase (all file:line citations validated)
- [ ] Runner-side changes enumerated explicitly (Claude handover ready)
- [ ] Backward compatibility path defined for all 4 existing built-in strategies
- [ ] Deflation counter separation specified (near_misses don't cross-contaminate)
- [ ] Data source resilience plan includes primary + backup
- [ ] Survivorship bias mitigation specified
- [ ] Transaction cost model specified for 500-name portfolios
- [ ] Cron capacity impact quantified
- [ ] Open questions explicitly listed for user decision
- [ ] No code changes made (planning deliverable only)
