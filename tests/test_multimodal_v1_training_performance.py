from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.train_multimodal_v1_full import (
    expand_projection_context,
    image_tensor,
    load_left_rgb,
    prepare_e5_image,
    resize_wh,
)
from rdq_uav.multimodal_v1.projection import load_left_projection_context


def test_worker_image_preparation_is_numerically_identical(tmp_path: Path) -> None:
    height, width = 48, 80
    values = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
    image = (values % 256).astype(np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(image).save(path)

    source_wh = (64, 40)
    source = load_left_rgb(path, source_wh)
    view_wh = resize_wh(source.size, short_edge=32, max_size=64)
    legacy = image_tensor(
        source.resize(view_wh, Image.Resampling.BILINEAR), torch.device("cpu")
    )
    prepared = prepare_e5_image(path, source_wh, short_edge=32, max_size=64)

    assert prepared["image_uint8"].dtype == torch.uint8
    assert torch.equal(legacy, prepared["image_uint8"].float())
    assert tuple(prepared["image_source_wh"].tolist()) == source_wh
    assert tuple(prepared["image_view_wh"].tolist()) == view_wh


def test_cached_projection_matches_reparsed_projection() -> None:
    root = Path(__file__).resolve().parents[1]
    camera = root / "configs/calibration/mmaud_v1_omni.yaml"
    geometry = root / "calibration/official_left_p4_current_geometry.json"
    scales = torch.tensor(((0.5, 0.75), (1.0, 1.0)), dtype=torch.float32)
    base = load_left_projection_context(
        camera, geometry, image_scale_xy=torch.ones((1, 2)), device="cpu"
    )
    cached = expand_projection_context(base, scales)
    reference = load_left_projection_context(
        camera, geometry, image_scale_xy=scales, device="cpu"
    )

    for name in (
        "rotation_camera_from_radar",
        "translation_camera_from_radar_m",
        "intrinsics",
        "distortion",
        "image_size_wh",
        "image_scale_xy",
    ):
        assert torch.equal(getattr(cached, name), getattr(reference, name))


def test_loader_rng_state_reproduces_resumed_epoch_order() -> None:
    dataset = TensorDataset(torch.arange(31))

    def loader(generator: torch.Generator) -> DataLoader:
        return DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)

    uninterrupted_generator = torch.Generator().manual_seed(42)
    list(loader(uninterrupted_generator))  # completed epoch 1
    saved = uninterrupted_generator.get_state()
    expected = torch.cat([batch[0] for batch in loader(uninterrupted_generator)])

    resumed_generator = torch.Generator()
    resumed_generator.set_state(saved)
    actual = torch.cat([batch[0] for batch in loader(resumed_generator)])
    assert torch.equal(expected, actual)
