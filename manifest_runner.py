#!/usr/bin/env python3
"""Scaffold manifest runner (Phase 0 PR-B).

Executed inside the disposable backtest container when the sandbox runner
receives a /run_manifest request. Reads a base64-encoded manifest, validates
it via manifest.py, and returns a JSON response.

For now, the executor is a scaffold: it validates the manifest structure and
reports the declared universe. Full manifest execution (code load, portfolio
returns, evaluator dispatch) lands in a follow-up PR once PR-C (OHLCV adapter)
and PR-D (evaluators) are available.
"""
import argparse
import base64
import json
import sys
import traceback

from manifest import StrategyManifest, ManifestValidationError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-b64", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    try:
        raw = base64.b64decode(args.manifest_b64, validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error_type": "infra",
            "error": f"failed to decode manifest: {e}",
        }))
        return 0

    try:
        manifest = StrategyManifest.from_dict(payload)
    except ManifestValidationError as e:
        print(json.dumps({
            "status": "error",
            "error_type": "manifest",
            "error": str(e),
        }))
        return 0
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error_type": "infra",
            "error": f"unexpected error parsing manifest: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }))
        return 0

    violations = manifest.validate()
    if violations:
        print(json.dumps({
            "status": "error",
            "error_type": "manifest",
            "error": "manifest validation failed",
            "violations": violations,
        }))
        return 0

    universe = []
    for ds in manifest.data_sources:
        if getattr(ds, "type", None) == "ohlcv":
            universe.extend(getattr(ds, "universe", []))

    print(json.dumps({
        "status": "scaffold",
        "manifest_name": manifest.name,
        "universe": universe,
        "evaluator_type": manifest.evaluator.type,
        "requested_metrics": list(manifest.evaluator.metrics),
        "note": "manifest execution stub — full execution lands after PR-C/D",
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
