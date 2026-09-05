#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdq_uav.config import load_config  # noqa: E402
from rdq_uav.experiment import make_dataset, make_loader  # noqa: E402
from rdq_uav.models import build_model  # noqa: E402
from rdq_uav.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real-batch forward/backward smoke test")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/smoke.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    dataset = make_dataset(config, "val", limit_per_class=1)
    loader = make_loader(config, dataset, "val", batch_size=5)
    batch = next(iter(loader))
    expected_views = 1 if config["data"].get("image_mode") == "oracle_left" else 2
    expected_height, expected_width = config["data"]["image_size"]
    expected_points = int(config["data"]["radar"]["max_points"])
    assert batch["image"].shape == (
        5,
        expected_views,
        3,
        expected_height,
        expected_width,
    )
    assert batch["radar"].shape == (5, expected_points, 3)
    assert batch["radar_mask"].shape == (5, expected_points)
    print(
        "real_batch",
        {key: tuple(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
    )

    for variant in ("rgb", "radar", "concat", "learned_query", "rdq"):
        model_cfg = copy.deepcopy(config["model"])
        model_cfg["variant"] = variant
        model = build_model(model_cfg)
        output = model(
            batch["image"], batch["radar"], batch["radar_mask"], return_attention=True
        )
        loss = torch.nn.functional.cross_entropy(output["logits"], batch["label"])
        loss.backward()
        attention_shape = (
            None if output["attention"] is None else tuple(output["attention"].shape)
        )
        print(
            variant,
            "logits",
            tuple(output["logits"].shape),
            "attention",
            attention_shape,
            "loss",
            float(loss),
        )
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
