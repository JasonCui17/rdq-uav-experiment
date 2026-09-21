#!/usr/bin/env python3
"""Summarize P4 measurements and enforce threshold preregistration.

Projection measurement generation is intentionally separate: it may only use
the subsequently frozen and verified LiDAR-to-left-camera calibration.  The
input JSON is a list of measurement records documented by
``multimodal_v1.geometry_audit``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from rdq_uav.multimodal_v1.geometry_audit import (
    evaluate_geometry_gate,
    summarize_geometry_measurements,
    validate_threshold_contract,
)


def _verify_committed_contract(path: Path, contract: dict) -> None:
    validate_threshold_contract(contract)
    repository = Path(__file__).resolve().parents[1]
    relative = path.resolve().relative_to(repository)
    committed = subprocess.run(
        ["git", "show", f"{contract['threshold_commit']}:{relative.as_posix()}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if committed != path.read_text():
        raise RuntimeError(
            "threshold contract differs from threshold_commit; formal audit refused"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--threshold-contract", type=Path)
    parser.add_argument("--mode", choices=("dry-run", "formal"), default="dry-run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.measurements.read_text())
    if not isinstance(records, list):
        raise ValueError("measurements must be a JSON list")
    summary = summarize_geometry_measurements(records)
    report = {"mode": args.mode, "metrics": summary, "gate": {"status": "UNREGISTERED"}}
    if args.mode == "formal":
        if args.threshold_contract is None:
            raise ValueError("formal audit requires --threshold-contract")
        contract = yaml.safe_load(args.threshold_contract.read_text())
        _verify_committed_contract(args.threshold_contract, contract)
        report["gate"] = evaluate_geometry_gate(summary, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
