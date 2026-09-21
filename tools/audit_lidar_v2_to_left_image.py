#!/usr/bin/env python3
"""Summarize P4 measurements and enforce threshold preregistration.

Projection measurement generation is intentionally separate. The input JSON is
one split's measurement list produced by
``tools/generate_lidar_v2_geometry_measurements.py``.
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


def _verify_committed_contract(path: Path, contract: dict) -> str:
    """Return the commit that last changed this exact frozen contract.

    The old design stored that same commit SHA inside the YAML, which is
    self-referential: inserting a commit hash changes the file and therefore
    changes the commit hash. Instead, require the worktree file to be clean and
    byte-identical to the last committed version, then record that commit in the
    audit report.
    """

    validate_threshold_contract(contract)
    repository = Path(__file__).resolve().parents[1]
    relative = path.resolve().relative_to(repository)

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", relative.as_posix()],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "threshold contract has uncommitted changes; formal audit refused"
        )

    threshold_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative.as_posix()],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(threshold_commit) < 7:
        raise RuntimeError("threshold contract has no committed history")

    committed = subprocess.run(
        ["git", "show", f"{threshold_commit}:{relative.as_posix()}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if committed != path.read_text():
        raise RuntimeError(
            "threshold contract differs from its last committed version; formal audit refused"
        )
    return threshold_commit


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
        threshold_commit = _verify_committed_contract(args.threshold_contract, contract)
        report["gate"] = evaluate_geometry_gate(
            summary,
            contract,
            threshold_commit=threshold_commit,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
