#!/usr/bin/env python3
"""Validate and freeze reviewed annotation/split inputs for one pipeline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return f"UNAVAILABLE: {exc}"
    return (result.stdout or result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--nextflow-version", required=True)
    parser.add_argument("--test-train-sequence")
    parser.add_argument("--test-val-sequence")
    args = parser.parse_args()

    base_config = yaml.safe_load(args.base_config.read_text())
    if os.environ.get("RDQ_DATA_ROOT"):
        base_config["data"]["root"] = os.environ["RDQ_DATA_ROOT"]
    split = json.loads(args.split.read_text())
    split_keys = (
        base_config["data"]["train_split"],
        base_config["data"]["val_split"],
        "heldout_test_sub",
    )
    groups = {key: list(split.get(key, [])) for key in split_keys}
    for left_index, left in enumerate(split_keys):
        for right in split_keys[left_index + 1 :]:
            overlap = set(groups[left]) & set(groups[right])
            if overlap:
                raise RuntimeError(f"sequence leakage between {left} and {right}: {sorted(overlap)}")

    records = []
    for line_number, line in enumerate(args.manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = {"sequence_id", "query_time", "image_path"} - set(record)
        if missing:
            raise RuntimeError(f"manifest line {line_number} lacks {sorted(missing)}")
        records.append(record)

    test_mode = bool(args.test_train_sequence or args.test_val_sequence)
    if test_mode:
        if not args.test_train_sequence or not args.test_val_sequence:
            raise ValueError("test profile requires both train and validation sequences")
        if args.test_train_sequence == args.test_val_sequence:
            raise ValueError("test train/validation sequences must differ")
        if args.test_train_sequence not in groups[split_keys[0]]:
            raise ValueError(f"unknown test train sequence {args.test_train_sequence}")
        if args.test_val_sequence not in groups[split_keys[1]]:
            raise ValueError(f"unknown test validation sequence {args.test_val_sequence}")
        split[split_keys[0]] = [args.test_train_sequence]
        split[split_keys[1]] = [args.test_val_sequence]
        split["heldout_test_sub"] = []
        allowed = {args.test_train_sequence, args.test_val_sequence}
        records = [record for record in records if record["sequence_id"] in allowed]

    train_sequences = set(split[split_keys[0]])
    val_sequences = set(split[split_keys[1]])
    if train_sequences & val_sequences:
        raise RuntimeError("prepared train/validation split is not sequence-isolated")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared_manifest = output / "manifest.jsonl"
    prepared_split = output / "split.json"
    prepared_config = output / "base_config.yaml"
    prepared_manifest.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records))
    prepared_split.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n")
    prepared_config.write_text(yaml.safe_dump(base_config, sort_keys=False, allow_unicode=True))

    dataset_root = Path(base_config["data"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = args.project_root / dataset_root
    environment = output / "environment.txt"
    environment.write_text(
        "\n".join(
            (
                f"python={sys.version}",
                f"platform={platform.platform()}",
                f"conda_prefix={Path(sys.prefix).resolve()}",
                f"dataset_root={dataset_root.resolve()}",
                f"nextflow={args.nextflow_version}",
                f"nvidia_smi={command_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])}",
                "",
                "[pip-freeze]",
                command_output([sys.executable, "-m", "pip", "freeze"]),
            )
        )
        + "\n"
    )

    summary = {
        "test_mode": test_mode,
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(val_sequences),
        "manifest_records": len(records),
        "train_manifest_records": sum(item["sequence_id"] in train_sequences for item in records),
        "validation_manifest_records": sum(item["sequence_id"] in val_sequences for item in records),
        "sequence_overlap": [],
    }
    (output / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "git_commit": args.git_commit,
        "source_fingerprint": args.source_fingerprint,
        "git_status": command_output(["git", "-C", str(args.project_root), "status", "--short"]),
        "data_version": args.data_version,
        "nextflow_version": args.nextflow_version,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_prefix": str(Path(sys.prefix).resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "inputs": {
            "base_config": {"path": str(args.base_config.resolve()), "sha256": sha256(args.base_config)},
            "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
            "split": {"path": str(args.split.resolve()), "sha256": sha256(args.split)},
        },
        "prepared": {
            "manifest_sha256": sha256(prepared_manifest),
            "split_sha256": sha256(prepared_split),
            "environment_sha256": sha256(environment),
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
