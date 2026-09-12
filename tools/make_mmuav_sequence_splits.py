#!/usr/bin/env python3
"""Create and freeze deterministic sequence-level MMUAV reproduction splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("/home/jasoncui/datasets/MMAUD/official/train")
DEFAULT_OUTPUT = Path(
    "/home/jasoncui/projects/rdq-uav-experiment/outputs/"
    "mmuav_paper_reproduction/splits/splits.json"
)


def discover_sequences(root: Path) -> list[str]:
    sequences = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "lidar_360").is_dir() and (path / "ground_truth").is_dir():
            if any((path / "lidar_360").glob("*.npy")) and any(
                (path / "ground_truth").glob("*.npy")
            ):
                sequences.append(path.name)
    return sequences


def make_splits(sequences: list[str], seed: int = 42) -> dict[str, object]:
    if len(sequences) < 3:
        raise ValueError("At least three GT-bearing sequences are required")
    shuffled = np.asarray(sorted(sequences), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    # 102 sequences -> 72/15/15, closest integer allocation to 70/15/15.
    n_val = int(round(len(shuffled) * 0.15))
    n_heldout = int(round(len(shuffled) * 0.15))
    n_train = len(shuffled) - n_val - n_heldout
    result = {
        "schema_version": 1,
        "seed": seed,
        "source_root": str(DEFAULT_ROOT),
        "allocation": "sequence_level_random_approximately_70_15_15",
        "num_sequences": len(shuffled),
        "train_sub": sorted(shuffled[:n_train].tolist()),
        "validation_sub": sorted(shuffled[n_train:n_train + n_val].tolist()),
        "heldout_test_sub": sorted(shuffled[n_train + n_val:].tolist()),
    }
    joined = "\n".join(
        f"{split}:{sequence}"
        for split in ("train_sub", "validation_sub", "heldout_test_sub")
        for sequence in result[split]
    )
    result["assignment_sha256"] = hashlib.sha256(joined.encode()).hexdigest()
    return result


def validate_disjoint(payload: dict[str, object]) -> None:
    groups = [set(payload[name]) for name in (
        "train_sub", "validation_sub", "heldout_test_sub"
    )]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Sequence leakage detected between splits")
    if len(set.union(*groups)) != int(payload["num_sequences"]):
        raise RuntimeError("Split union does not match num_sequences")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sequences = discover_sequences(args.data_root)
    payload = make_splits(sequences, args.seed)
    payload["source_root"] = str(args.data_root.resolve())
    validate_disjoint(payload)
    text = json.dumps(payload, indent=2) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"Frozen split exists with different content: {args.output}"
            )
        print(f"frozen split already matches: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote frozen split: {args.output}")
    for name in ("train_sub", "validation_sub", "heldout_test_sub"):
        print(f"{name}: {len(payload[name])} sequences")
    print(f"assignment_sha256: {payload['assignment_sha256']}")


if __name__ == "__main__":
    main()
