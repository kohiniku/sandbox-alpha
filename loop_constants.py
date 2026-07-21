"""
Shared constants for the sandbox-alpha autonomous loop.

Stdlib-only — no imports of other project modules to avoid import-time
side effects and circular dependencies.  Safe to import from anywhere,
including modules that autonomous_loop itself depends on.
"""

# --- Sentinel for missing/absent metrics ---
MISSING_METRIC = -999  # flows into gates so missing data always fails


# --- Verdict and backlog-status string constants ---
class Verdict:
    """Evaluation verdict tags — must match JSON/Cron contract exactly."""
    ADOPTED = 'adopted'
    REJECTED = 'rejected'
    ERROR = 'error'
    CODE_ERROR = 'code_error'


class BacklogStatus:
    """Backlog entry lifecycle states — must match Backlog class contract."""
    PENDING = 'pending'
    TESTING = 'testing'
    DONE_ADOPTED = 'done_adopted'
    DONE_REJECTED = 'done_rejected'
    DONE_ERROR = 'done_error'


# --- Gate v2 (CV + bootstrap LCB) ---
# Frozen defaults from docs/plan-cv-bootstrap.md §3.2.
# These are the canonical values for walk-forward CV and bootstrap LCB
# gate functions added in PR #2.  They are not used by any v1 code path.

CV_FOLDS = 3                          # number of expanding-window CV folds
EMBARGO_DAYS = 21                     # trading-day gap between train and val
BOOTSTRAP_ALPHA = 0.05                # significance level for LCB (5%)
BOOTSTRAP_N_RESAMPLE = 2000           # bootstrap resamples (B)
BLOCK_LEN_MIN = 21                    # floor for block-length heuristic
