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
