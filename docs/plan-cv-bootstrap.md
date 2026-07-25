# sandbox-alpha Gate Refactor: Walk-Forward CV + Bootstrap LCB

> **PR plan, not code.** 実装前に設計判断を固めるための文書。
> Target: `github.com/kohiniku/sandbox-alpha`

---

## 1. Current Gating Implementation (Exact Citations)

### 1.1 Core gating functions — `autonomous_loop.py`

**Deflation formula:**
```python
# autonomous_loop.py:73-76
def compute_effective_min_sharpe(N_family, T_val):
    """Deflation formula: threshold rises with more trials to penalize data snooping."""
    N = max(N_family, 2)
    return max(MIN_SHARPE_BASE, math.sqrt(2 * math.log(N)) * math.sqrt(252.0 / max(T_val, 1)))
```

**Validation gate:**
```python
# autonomous_loop.py:497-518
def _eval_val_gate(val_sharpe, val_return, val_max_dd, effective_min_sharpe,
                   N_family, T_val):
    """Validation gate: returns (passed: bool, reasons: list[str])."""
    passed = (
        val_sharpe >= effective_min_sharpe
        and val_return > 0
        and val_max_dd >= MAX_DRAWDOWN_LIMIT
    )
```
`MAX_DRAWDOWN_LIMIT = -25.0` (`autonomous_loop.py:68`).
`MIN_SHARPE_BASE = 0.5` (`autonomous_loop.py:67`).

**Holdout gate:**
```python
# autonomous_loop.py:521-539
def _eval_holdout_gate(holdout_sharpe, holdout_return, val_sharpe):
    """Holdout confirmation gate: returns (passed: bool, reasons: list[str])."""
    holdout_threshold = min(0.5, 0.5 * val_sharpe)
    passed = (holdout_sharpe >= holdout_threshold) and (holdout_return > 0)
```

**Orchestration (param path):**
```python
# autonomous_loop.py:547-683  (evaluate_result)
# Gate order: (a) validation → (b) deflation [embedded in (a)] → (c) holdout → (d) cluster dedup
# Lines 579-581: val gate uses val_metrics from result["out_of_sample"]
# Lines 603-607: holdout gate uses result["holdout"]
```

**Orchestration (manifest path):**
```python
# autonomous_loop.py:840-937  (_evaluate_manifest_result)
# Same gate sequence applied to manifest runner results
```

### 1.2 Walk-forward split — `backtests/metrics.py`

```python
# backtests/metrics.py:36-41
def split_walkforward(df, train_ratio=0.6, val_ratio=0.2, holdout_ratio=0.2):
    """Split data into train (in-sample), validation, and holdout segments chronologically."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
```

No embargo/purge gap between segments. Pure chronological split on 5y daily OHLCV (~1260 rows).

### 1.3 Manifest runner split — `manifest_runner.py`

```python
# manifest_runner.py:138-148
def _walk_forward_split(
    index: pd.DatetimeIndex, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.Timestamp, pd.Timestamp]:
    n = len(index)
    train_end_idx = int(n * train_frac) - 1
    val_end_idx = int(n * (train_frac + val_frac)) - 1
    return index[train_end_idx], index[val_end_idx]
```

Same 60/20/20 scheme, no embargo.

### 1.4 Near-miss classification — `autonomous_loop.py`

```python
# autonomous_loop.py:690-744
def _classify_near_miss(hypothesis, evaluation):
    # (a) val_sharpe >= 90% of deflated threshold but validation failed → "val_sharpe_90pct"
    # (b) validation passed but holdout failed → "holdout"
    # (c) drawdown/return gate failure despite Sharpe passing → "max_drawdown" / "val_return"
```

Capped at 30 entries (`autonomous_loop.py:763-764`).

### 1.5 OOS monitor — `oos_monitor.py`

```python
# oos_monitor.py:77-133 (run_oos_check)
# Runs backtest with metrics_since=adoption_date
# Records oos_sharpe as raw point estimate — NO gate, pure monitoring
```

No gate applied in OOS monitor. It uses `run_backtest()` which returns point-estimate Sharpe from the engine.

---

## 2. PR Breakdown (2–4 PRs, Each Independently Mergeable)

### PR #1: Walk-Forward CV Splitter + Embargo (`backtests/splitter.py`)
**What:** Replace single-shot `split_walkforward()` with an expanding-window walk-forward CV splitter that yields K folds and enforces an embargo gap.

**Independent because:** Pure utility module. No gate changes. Existing `split_walkforward()` retained as a compat wrapper for one cycle. Tests pass on synthetic data. Mergeable without touching any gate or knowledge.json schema.

**Files:**
- **NEW:** `backtests/splitter.py` — `WalkForwardCV` class
- **MODIFY:** `backtests/metrics.py` — keep `split_walkforward` as compat shim, add `from .splitter import WalkForwardCV`
- **NEW:** `tests/test_splitter.py`

### PR #2: Bootstrap LCB Utility + Gate Refactor (`autonomous_loop.py`)
**What:** Add `BootstrapLCB.compute(sharpe_series, block_len, n_resample, alpha)` utility class. Replace `_eval_val_gate` and `_eval_holdout_gate` point-estimate comparisons with LCB-based comparisons. New functions: `_eval_val_gate_cv()` and `_eval_holdout_gate_cv()`.

**Independent because:** Builds on PR #1's splitter. Gate changes are self-contained in `autonomous_loop.py`. Existing gates remain as fallback via feature flag.

**Files:**
- **NEW:** `backtests/bootstrap.py` — `BootstrapLCB` class
- **MODIFY:** `autonomous_loop.py` — add `_eval_val_gate_cv()`, `_eval_holdout_gate_cv()`, modify `evaluate_result()` to route via feature flag
- **MODIFY:** `loop_constants.py` — add `CV_FOLDS`, `EMBARGO_DAYS`, `BOOTSTRAP_ALPHA`, `BOOTSTRAP_N_RESAMPLE`, `BLOCK_LEN_HEURISTIC` constants
- **NEW:** `tests/test_bootstrap.py`
- **MODIFY:** `tests/test_overfitting_guards.py` — add CV+Bootstrap variants of existing tests

### PR #3: Characterization Pass + Rollout (feature flag + parallel reporting)
**What:** Run all current `near_misses` through the new gate on the same data. Report false-negative-reversal count. Add `SANDBOX_GATE_V2=1` env-var feature flag with parallel gate reporting (both old and new verdicts logged side-by-side for N iterations before cutover).

**Independent because:** No behavior change without env var. Mergeable with zero risk.

**Files:**
- **MODIFY:** `autonomous_loop.py` — feature flag routing in `evaluate_result()`, parallel `_classify_near_miss_cv()` for dual reporting
- **NEW:** `scripts/characterize_gate_v2.py` — replay script
- **MODIFY:** `knowledge.json` — optional new fields `gate_v2_verdict`, `gate_v2_lcb` (write only when feature flag on, backward-compat read)

### PR #4: OOS Monitor + knowledge.json Schema Migration
**What:** Update `oos_monitor.py` to use CV+Bootstrap for re-measurement. Add `gate_version` field to knowledge.json entries for forward-only schema migration (old verdicts stay `"v1"`, new ones get `"v2"`).

**Files:**
- **MODIFY:** `oos_monitor.py` — adopt new gate for re-measurement
- **MODIFY:** `autonomous_loop.py` — add `gate_version: "v2"` to adopted/rejected record schemas
- **MODIFY:** `tests/test_holdout_oos.py` — update mock data to include CV+Bootstrap-shaped results

---

## 3. Design Decisions (Per PR)

### 3.1 PR #1 — Fold Strategy

| Axis | Recommendation | Rationale |
|---|---|---|
| **Window type** | **Expanding-window** (anchored at earliest data) | Sliding window loses statistical power on short series. With 5y daily (~1260 rows), a sliding window of fixed 252d per fold would produce only ~4 folds with severe overlap. Expanding keeps train size growing, mirrors live deployment (retrain-on-all-history), and gives 3–5 folds with ~200–250 val days each. |
| **Fold count (K)** | **3 folds** (anchored expanding) | With 5y data: fold 1 = train[0:756], val[757:1008] (with embargo); fold 2 = train[0:882], val[883:1134]; fold 3 = train[0:1008], val[1009:1260]. Each val is ~252 days. More folds → shrinking train, not helpful. |
| **Embargo/purge** | **21 trading days** between train/val boundary | Motivation: daily autocorrelation of returns is negligible, but **strategy signals** (SMA crossover, RSI, momentum) have lookback windows up to 100 days. Without an embargo, a 100-day SMA on train[-1] leaks information about val[0] through the moving average's trailing window. 21 days covers typical signal decay and is conservative without burning excessive data. |
| **Holdout after CV** | Keep final 20% as true holdout | Same as current: holdout is never used in CV. CV operates only on the train+val portion (first 80% of data). |

```
Data layout for 5y (~1260 days, 252d/year):
|------ Train (expanding) ------|--Embargo--|--Val--|--Holdout--|
Fold 1: [0:756]                  [757:777]   [778:1029]          |
Fold 2: [0:882]                  [883:903]   [904:1155]          |
Fold 3: [0:1008]                 [1009:1029] [1030:1260]         |
Final holdout:                                                [1260:]  ← same as current, never touched by CV
Wait — recalibrating. If holdout still gets last 20%, folds share the first 80%.
Fold 1: train[0:605], gap[606:625], val[626:756]    (val ~130 days → tighter but workable)
Fold 2: train[0:706], gap[707:726], val[727:857]
Fold 3: train[0:807], gap[808:827], val[828:958]
Holdout: [958:1260] (~302 days)
```

**Revised: 3 folds within the 60+20 portion, holdout untouched at last 20%.** Each val fold is ~130 trading days (~0.5 yr). This is intentionally short — CV's power comes from averaging across folds, not individual fold precision.

### 3.2 PR #2 — Bootstrap Design

| Axis | Recommendation | Rationale |
|---|---|---|
| **Block length** | **`max(21, int(sqrt(T_fold)))`** with floor at fixed 21-day embargo | `sqrt(T)` heuristic (Politis & White 2004). For T≈130: `sqrt(130)≈11`, floored to 21 (embargo). This matches the embargo gap and is conservative for daily returns where autocorrelation in the return series itself is near zero. Using empirical autocorrelation adds complexity for marginal gain. |
| **Resample count (B)** | **B = 2000** | Standard for 5% LCB precision. More is better but 2000 gives stable LCB estimates with <0.01 Sharpe SD. For 3 folds × 2000 resamples = 6000 total, computation is sub-second with numpy. |
| **Confidence level (α)** | **α = 0.05 → 5% LCB** | Standard practice. LCB = `mean(bootstrap_sharpes) - z_{1-α} × SE_bootstrap`. 5% LCB means "we are 95% confident true Sharpe exceeds this value." |
| **Fold aggregation** | **Sharpe of concatenated fold returns** then bootstrap that single series | Alternative: median of per-fold LCBs. Concatenation is preferred because (a) it preserves temporal ordering across folds, (b) Sharpe is not additive so median of Sharpes is not a Sharpe, (c) concat → single Sharpe → single bootstrap is computationally simpler and a well-defined estimator (the "pooled Sharpe" across all OOS periods). |
| **Gate function signature** | `_eval_val_gate_cv(per_fold_returns: list[pd.Series], val_return, val_max_dd, N_family)` → `(passed, lcb_sharpe, reasons)` | Returns LCB Sharpe instead of point estimate. `effective_min_sharpe` from deflation formula still applied but compared against LCB, not point estimate. |

### 3.3 PR #2 — Deflated-Sharpe Interaction

**Recommendation: Keep the DSR term, apply it to the LCB, not the point estimate.**

```
Gate check:
  LCB_sharpe >= max(MIN_SHARPE_BASE, sqrt(2*ln(N_family)) * sqrt(252/T_val_cv))

where T_val_cv = total OOS days across all folds (≈ 3 × 130 = 390)
```

**Trade-off argument:**
- **Keep DSR:** The multiple-testing penalty is orthogonal to estimation uncertainty. DSR penalizes for searching across many (strategy, symbol) pairs; bootstrap penalizes for estimation noise on a single pair. They compound naturally: `(true Sharpe - data-snooping bias - estimation noise) > 0`.
- **Drop DSR:** Bootstrap already captures "worst-case across folds" → some argue this subsumes the deflation penalty. But bootstrap captures *within-family* variance, not *across-family* multiple comparisons. A strategy that looks good by chance because you tried 100 families is not captured by bootstrap.
- **Fold into LCB level:** Setting α smaller (1% instead of 5%) as a proxy for deflation. Attractive simplicity but conflates two distinct error sources and makes the α choice ad-hoc rather than principled.

**Decision: Keep DSR applied to LCB.** The DSR formula uses `T_val_cv` = total OOS days (not per-fold) because the LCB is computed on concatenated returns.

### 3.4 PR #3 — Backward Compatibility / Rollout

| Axis | Decision |
|---|---|
| **Feature flag** | `SANDBOX_GATE_V2=1` env var. When unset or `0`, current behavior unchanged. |
| **Parallel reporting** | When flag is on, `evaluate_result()` computes BOTH old and new verdicts. Old verdict drives knowledge.json (adopted/rejected). New verdict is written to `evaluation["gate_v2"]` as `{verdict, lcb_sharpe, effective_threshold}`. Cron report reads both and prints a diff line. |
| **Parallel window** | Run for N iterations (recommend N=50–100, gated by `SANDBOX_GATE_V2_PARALLEL=N` env var). After N iterations with parallel reporting, flip to full v2. |
| **knowledge.json schema** | Add optional `"gate_version": "v2"` field to evaluation dicts. Old entries (no field) default to `"v1"`. Near-misses schema adds optional `"gate_v2_would_adopt": bool` field. `effective_min_sharpe` field gets companion `effective_min_sharpe_v2` when v2 gate is active. |
| **backlog verdicts** | Backlog entries store `result.verdict` as a string. The `"rejected"` / `"adopted"` strings don't change. Add `gate_version` to the `result` dict written by `bl.mark()`. Filter on `gate_version` for analysis; don't migrate old entries. |

**Key principle: forward-only.** Old verdicts are never re-interpreted. They were correct under v1. Analysis comparing adoption rates should filter by `gate_version`.

---

## 4. Impact Analysis (Concrete File-Level Impact)

### 4.1 `knowledge.json` fields

| Field | Impact |
|---|---|
| `evaluation.sharpe_ratio` | Stays as point-estimate val Sharpe for backward compat. Add `evaluation.lcb_sharpe` (new). |
| `evaluation.effective_min_sharpe` | Stays. Add `evaluation.effective_min_sharpe_v2` (recomputed with `T_val_cv` total OOS days). |
| `evaluation.gate_results` | Stays. Add `evaluation.gate_results_v2` when v2 active. |
| `evaluation.gate_version` | NEW: `"v1"` or `"v2"`. |
| `near_misses[].val_sharpe` | Stays. Add `near_misses[].lcb_sharpe` (may be null for v1 entries). |
| `near_misses[].gate_v2_would_adopt` | NEW: boolean. |
| `families[].best_val_sharpe` | Stays as best point-estimate. Add `families[].best_lcb_sharpe`. |
| `families[].gate_failures` | Stays. Failure counting logic in `_apply_entry_to_family()` reads `gate_results` — add v2-aware path. |
| `backlog.json` entries' `result` dict | Add optional `gate_version` string. |

### 4.2 Preflight path (`/validate`)

**No changes needed.** Confirmed: there is no `/validate` endpoint in sandbox-alpha. The only preflight is `backtests/strategy_harness.py:run_preflight()` which checks code contract, not strategy performance. It uses synthetic data (250 rows) and would fail to produce meaningful CV folds anyway. Do not touch.

### 4.3 OOS Monitor (`oos_monitor.py`)

**Recommendation: Adopt the new gate for re-measurement.**

Rationale: OOS monitor checks "was this adoption actually good on data that didn't exist?" If we think CV+Bootstrap is a more honest estimator, we should use it here too. However, the OOS monitor already has a unique advantage (true future data), so the bootstrap CI on a single OOS period may be too wide to be useful until many months have passed.

**Compromise:** Continue recording raw OOS Sharpe (current behavior). Add a `use_gate_v2` parameter (default `False` for now) that also computes and records LCB. Gate v2 adoption in OOS monitor is deferred to a future PR — mark this item **REQUIRES CLAUDE** to decide timing.

### 4.4 Ideation v2/v3 Prompt (`strategy_ideation.py`)

**Should the LLM be told about the new gate?**

Yes, but only after PR #3 (characterization pass confirms the change is worth it). The LLM currently sees `near_misses` in its context. If near-misses start including `gate_v2_would_adopt: true`, the LLM should understand that borderline proposals now have a higher chance. The prompt change is minimal:

```
The evaluation gate has been upgraded: it now uses walk-forward cross-validation
with a bootstrap lower-confidence-bound instead of a single point-estimate.
Proposals that were previously near-misses may now pass.
```

Add this to `strategy_ideation.py`'s prompt template in the context-gathering section (`_summarise_rejects` or the system prompt). This is a 3-line change.

---

## 5. Test Plan

### 5.1 Characterization Pass (PR #3 pre-merge gate)

**Script:** `scripts/characterize_gate_v2.py`

**What it does:**
1. Load `knowledge.json` → extract all `near_misses` (up to 30).
2. For each near-miss, reconstruct the hypothesis and re-run the backtest (or read the stored `backtest_result` from `results/<hyp_id>.json`).
3. Apply the v2 gate (CV split + bootstrap LCB) to the same OHLCV data.
4. Report: `{strategy}/{symbol}` → `v1_verdict: rejected`, `v1_val_sharpe: X`, `v2_lcb: Y`, `v2_would_adopt: true/false`.
5. Summary line: `CHARACTERIZATION: N near-misses tested, M would-be-adopted (M/N%)`.

**Success criterion:** If ≥ 3 near-misses flip to "would adopt" under v2, the change is empirically justified. If 0 flips, the hypothesis that the current gate is Type-II dominated is wrong, and we should re-examine before merging.

**Implementation note:** This script must not mutate knowledge.json. It reads only.

### 5.2 Unit Tests

**PR #1 — `tests/test_splitter.py`:**

| Test | What it verifies |
|---|---|
| `test_num_folds` | 3 folds returned for 5y data |
| `test_chronological_order` | Train < Gap < Val within each fold |
| `test_embargo_days_exact` | Gap = 21 trading days |
| `test_holdout_unchanged` | Holdout is last 20% of data, identical to current `split_walkforward` holdout |
| `test_fold_non_overlapping_val` | Val periods of consecutive folds do not overlap (accounting for gap) |
| `test_expanding_train` | Fold K+1 train ⊃ Fold K train |
| `test_short_series_graceful` | Data < 200 rows → raises informative error, doesn't silently produce garbage folds |

**PR #2 — `tests/test_bootstrap.py`:**

| Test | What it verifies |
|---|---|
| `test_coverage_synthetic_normal` | Generate 1000 synthetic Sharpe processes with known true Sharpe=0.5, block-len=21, B=2000. 5% LCB should cover true Sharpe in ~95% of trials. |
| `test_coverage_synthetic_zero` | True Sharpe=0.0. LCB should be negative in >95% of trials (i.e., correctly identifies zero-skill). |
| `test_block_len_invariant` | Doubling block length lowers LCB (more conservative). |
| `test_n_resample_stability` | B=500 vs B=2000: LCB difference < 0.02 Sharpe. |
| `test_concatenation_aggregation` | Sharpe of concatenated fold returns ≈ mean of per-fold Sharpes (when folds have equal days; sanity check, not equality). |
| `test_empty_input` | Empty list or zero-length series → raises ValueError. |

**PR #3 — Integration tests (in `tests/test_overfitting_guards.py`):**

Adapt existing `_make_synthetic_result()` to produce CV-shaped results (dict with `folds` key containing per-fold metrics). Test that:
- A strategy that passes v1 but fails v2 is rejected when `SANDBOX_GATE_V2=1`.
- A strategy that fails v1 but passes v2 is adopted when `SANDBOX_GATE_V2=1`.
- `SANDBOX_GATE_V2=0` preserves exact v1 behavior (regression).

---

## 6. Open Questions for User / Claude Decision

1. **Fold count: 3 folds vs 5 folds?** 3 folds gives ~130 val days each (tight but CV benefits from averaging). 5 folds gives ~78 val days each (Sharpe SE ≈ 1.4, very noisy even with bootstrap). Recommendation is 3. Objections?

2. **Embargo: 21 days vs adaptive (max param lookback + buffer)?** 21 is simple and covers SMA-20/RSI-14/momentum-20. But if future strategies use 200-day lookbacks, 21 is insufficient. An adaptive embargo = `max(param_values) + 5` would dynamically match the strategy. More complex but more correct. Which?

3. **Deflated-Sharpe on LCB: keep, drop, or fold into α?** Plan recommends "keep, applied to LCB." If you think this is overly conservative (LCB already penalizes twice), we can drop DSR entirely and rely on bootstrap for both estimation uncertainty and multiple-comparison penalty. This is a philosophical choice — do we treat "tried 100 families" as separable from "this one family has noisy Sharpe"?

4. **OOS monitor gate upgrade timing?** Plan recommends deferring (record both, decide later). If you want the OOS monitor to immediately demote strategies whose LCB goes negative, that's a separate design question.

5. **Cutover strategy: parallel for N iterations then hard flip, or gradual?** Plan recommends N=50–100 iterations of parallel reporting, then env-var flip. If you want a gradual rollout (10% → 50% → 100% over weeks), that's more infra work but safer. Preference?

6. **REQUIRES CLAUDE — sandbox-runner interface change:** The CV split happens in the agent process (Python, `backtests/splitter.py`), not in the sandbox runner. The runner still gets single-train, single-val periods per invocation. The agent calls the runner K times (once per fold). This means the runner interface does NOT change. Confirm this is acceptable vs. having the runner do CV internally?

7. **Block length heuristic: `sqrt(T)` vs empirical autocorrelation?** `sqrt(21)≈4.6`, floored to 21. For daily returns this is generous (actual autocorrelation is near zero). Using empirical ACF to set block length would give shorter blocks and narrower CIs. Do you want the simpler `sqrt(T)` or the data-adaptive approach?

---

## Appendix A: Non-Authoritative Code Sketches

### A.1 WalkForwardCV (conceptual shape)

```python
# backtests/splitter.py (non-authoritative sketch)
class WalkForwardCV:
    def __init__(self, n_folds=3, embargo_days=21, train_frac=0.6, val_frac=0.2):
        ...
    def split(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Returns list of (train, val, holdout) tuples. Holdout is identical across folds."""
        ...
```

### A.2 BootstrapLCB (conceptual shape)

```python
# backtests/bootstrap.py (non-authoritative sketch)
class BootstrapLCB:
    @staticmethod
    def compute(returns: pd.Series, block_len: int = 21,
                n_resample: int = 2000, alpha: float = 0.05) -> float:
        """Circular block bootstrap → Sharpe distribution → lower α-quantile."""
        ...
```

### A.3 Gate V2 orchestration (conceptual shape)

```python
# autonomous_loop.py — inserted into evaluate_result() (non-authoritative sketch)
if os.environ.get("SANDBOX_GATE_V2") == "1":
    val_pass, lcb_sharpe, reasons = _eval_val_gate_cv(
        per_fold_val_returns, val_return, val_max_dd, N_family
    )
    effective_threshold = compute_effective_min_sharpe(N_family, total_oos_days)
    gate_results["validation"] = lcb_sharpe >= effective_threshold
    gate_results["_v2_lcb"] = lcb_sharpe
    ...
```

---

## Appendix B: Files Not Touched

These files require **zero changes** under this plan:
- `backtests/backtest_engine.py` — still produces per-segment metrics as before
- `backtests/strategies/*.py` — strategy code is unchanged
- `evaluators/*.py` — metric computation is unchanged
- `manifest_runner.py` — split happens in agent, not runner
- `manifest.py` — schema is unchanged
- `data_adapters/*.py` — data loading is unchanged
- `backlog.py` — only `result` dict gets optional field, no schema change
- `strategy_ideation.py` — prompt tweak only (1 section)
