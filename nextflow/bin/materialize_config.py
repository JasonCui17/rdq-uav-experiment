#!/usr/bin/env python3
"""Materialize an immutable training config from staged pipeline inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulate", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulate) < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch/accumulate must be positive and workers non-negative")
    config = yaml.safe_load(args.base_config.read_text())
    config["data"]["annotation_manifest"] = str(args.manifest.resolve())
    config["data"]["split_file"] = str(args.split.resolve())
    config["data"]["num_workers"] = args.num_workers
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["accumulate"] = args.accumulate
    config["experiment"]["output_dir"] = args.run_output
    config["experiment"]["name"] = f"{config['experiment']['name']}_nextflow"
    args.output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
