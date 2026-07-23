#!/usr/bin/env python3
"""
Manual operator CLI for managing strategy family lifecycle states.

Usage:
    python3 scripts/family_admin.py --list [--lifecycle candidate|refining|killed]
    python3 scripts/family_admin.py --kill <family_key> --reason "..."
    python3 scripts/family_admin.py --revive <family_key>

Uses the same flock pattern as backlog.py on knowledge.json.
KNOWLEDGE_PATH env respected; defaults to ./knowledge.json.
stdlib-only.
"""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path


def _get_knowledge_path():
    return Path(os.environ.get("KNOWLEDGE_PATH", Path(__file__).resolve().parent.parent / "knowledge.json"))


def _locked_load(path):
    fd = open(path, "r+")
    fcntl.flock(fd, fcntl.LOCK_EX)
    fd.seek(0)
    raw = fd.read() or "{}"
    return json.loads(raw), fd


def _locked_save(data, fd):
    fd.seek(0)
    fd.truncate()
    fd.write(json.dumps(data, indent=2, default=str))
    fd.flush()
    fcntl.flock(fd, fcntl.LOCK_UN)
    fd.close()


def cmd_list(lifecycle_filter=None):
    path = _get_knowledge_path()
    if not path.exists():
        print("knowledge.json not found", file=sys.stderr)
        sys.exit(1)

    data, fd = _locked_load(path)
    fcntl.flock(fd, fcntl.LOCK_UN)
    fd.close()

    families = data.get("families", {})
    if not families:
        print("(no families)")
        return

    for key in sorted(families.keys()):
        fam = families[key]
        lc = fam.get("lifecycle", "candidate")
        if lifecycle_filter and lc != lifecycle_filter:
            continue
        print(
            f"{key}  lifecycle={lc}  n_trials={fam.get('n_trials', 0)}  "
            f"best_val_sharpe={fam.get('best_val_sharpe', -999):.2f}  "
            f"refine_count={fam.get('refine_count', 0)}  "
            f"kill_reason={fam.get('kill_reason', '')!r}"
        )


def cmd_kill(family_key, reason):
    path = _get_knowledge_path()
    if not path.exists():
        print("knowledge.json not found", file=sys.stderr)
        sys.exit(1)

    data, fd = _locked_load(path)
    families = data.setdefault("families", {})

    if family_key not in families:
        print(f"Unknown family key: {family_key}", file=sys.stderr)
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        sys.exit(1)

    fam = families[family_key]
    if fam.get("lifecycle") == "killed":
        print(f"Family {family_key} is already killed", file=sys.stderr)
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        sys.exit(1)

    fam["lifecycle"] = "killed"
    fam["kill_reason"] = reason
    _locked_save(data, fd)
    print(f"Killed family {family_key}: {reason}")


def cmd_revive(family_key):
    path = _get_knowledge_path()
    if not path.exists():
        print("knowledge.json not found", file=sys.stderr)
        sys.exit(1)

    data, fd = _locked_load(path)
    families = data.setdefault("families", {})

    if family_key not in families:
        print(f"Unknown family key: {family_key}", file=sys.stderr)
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        sys.exit(1)

    fam = families[family_key]
    fam["lifecycle"] = "candidate"
    fam["kill_reason"] = ""
    _locked_save(data, fd)
    print(f"Revived family {family_key}")


def main():
    parser = argparse.ArgumentParser(description="Family lifecycle administration")
    parser.add_argument("--list", action="store_true", help="List families")
    parser.add_argument("--lifecycle", choices=["candidate", "refining", "killed"],
                        help="Filter by lifecycle state (used with --list)")
    parser.add_argument("--kill", metavar="FAMILY_KEY", help="Kill a family")
    parser.add_argument("--reason", default="", help="Reason for killing (used with --kill)")
    parser.add_argument("--revive", metavar="FAMILY_KEY", help="Revive a killed family")
    args = parser.parse_args()

    if args.list:
        cmd_list(lifecycle_filter=args.lifecycle)
    elif args.kill:
        if not args.reason:
            print("--reason is required with --kill", file=sys.stderr)
            sys.exit(1)
        cmd_kill(args.kill, args.reason)
    elif args.revive:
        cmd_revive(args.revive)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
