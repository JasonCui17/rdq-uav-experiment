from __future__ import annotations

from typing import Mapping, Sequence

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


class DualFisheyeTransform:
    """Split a 2560x960 panorama into two cameras and transform consistently."""

    def __init__(
        self,
        image_size: Sequence[int],
        training: bool,
        color_jitter: float = 0.0,
        image_mode: str = "dual_full",
        center_mask_fraction: float = 0.0,
        bbox_mode: str = "full",
        bbox_context_scale: float = 1.0,
    ) -> None:
        if len(image_size) != 2:
            raise ValueError("image_size must be [height, width]")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.training = training
        if image_mode not in {"dual_full", "oracle_left"}:
            raise ValueError(f"Unsupported image_mode: {image_mode}")
        self.image_mode = image_mode
        self.center_mask_fraction = float(center_mask_fraction)
        if not 0.0 <= self.center_mask_fraction < 1.0:
            raise ValueError("center_mask_fraction must be in [0, 1)")
        if bbox_mode not in {"full", "erase", "foreground_only", "crop"}:
            raise ValueError(f"Unsupported bbox_mode: {bbox_mode}")
        self.bbox_mode = bbox_mode
        self.bbox_context_scale = float(bbox_context_scale)
        if self.bbox_context_scale < 1.0:
            raise ValueError("bbox_context_scale must be >= 1")
        if self.center_mask_fraction > 0 and self.bbox_mode != "full":
            raise ValueError("center_mask_fraction and bbox_mode cannot be enabled together")
        self.jitter = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(color_jitter / 3.0, 0.5),
            )
            if color_jitter > 0
            else None
        )
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, image: Image.Image, row: Mapping[str, str] | None = None) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = image.size
        if self.image_mode == "dual_full":
            if self.training and self.jitter is not None:
                # Apply once to the panorama so both synchronized cameras
                # receive the same photometric perturbation.
                image = self.jitter(image)
            if width % 2 != 0:
                raise ValueError(f"Dual-fisheye image width must be even, got {image.size}")
            midpoint = width // 2
            cameras = [
                image.crop((0, 0, midpoint, height)),
                image.crop((midpoint, 0, width, height)),
            ]
        else:
            if row is None:
                raise ValueError("oracle_left mode requires manifest ROI columns")
            required = ("roi_x1", "roi_y1", "roi_x2", "roi_y2")
            if any(not row.get(key, "") for key in required):
                raise ValueError(f"Missing Oracle ROI columns: {required}")
            box = tuple(int(float(row[key])) for key in required)
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width // 2 and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid left ROI {box} for image size {image.size}")
            camera = image.crop(box)
            if self.training and self.jitter is not None:
                camera = self.jitter(camera)
            if self.bbox_mode != "full":
                required_bbox = (
                    "official_bbox_x1",
                    "official_bbox_y1",
                    "official_bbox_x2",
                    "official_bbox_y2",
                )
                if any(not row.get(key, "") for key in required_bbox):
                    raise ValueError(f"Missing official bbox columns: {required_bbox}")
                bx1, by1, bx2, by2 = [float(row[key]) for key in required_bbox]
                if not (bx1 < bx2 and by1 < by2):
                    raise ValueError(f"Invalid official bbox: {(bx1, by1, bx2, by2)}")
                center_x = (bx1 + bx2) / 2.0 - x1
                center_y = (by1 + by2) / 2.0 - y1
                box_width = (bx2 - bx1) * self.bbox_context_scale
                box_height = (by2 - by1) * self.bbox_context_scale
                camera_width, camera_height = camera.size
                local_x1 = max(0, round(center_x - box_width / 2.0))
                local_y1 = max(0, round(center_y - box_height / 2.0))
                local_x2 = min(camera_width, round(center_x + box_width / 2.0))
                local_y2 = min(camera_height, round(center_y + box_height / 2.0))
                if local_x1 >= local_x2 or local_y1 >= local_y2:
                    raise ValueError(
                        "Official bbox does not overlap Oracle ROI: "
                        f"bbox={(bx1, by1, bx2, by2)}, roi={box}"
                    )
                original = camera
                if self.bbox_mode == "erase":
                    camera = camera.copy()
                    camera.paste(
                        (124, 116, 104),
                        (local_x1, local_y1, local_x2, local_y2),
                    )
                elif self.bbox_mode == "foreground_only":
                    foreground = original.crop((local_x1, local_y1, local_x2, local_y2))
                    camera = Image.new("RGB", original.size, (124, 116, 104))
                    camera.paste(foreground, (local_x1, local_y1))
                else:
                    # Mirrors the detection-then-crop classification pipeline
                    # used by the CVPR 2024 UG2 winning solution.
                    camera = original.crop((local_x1, local_y1, local_x2, local_y2))
            if self.center_mask_fraction > 0:
                if not row.get("roi_center_u", "") or not row.get("roi_center_v", ""):
                    raise ValueError("Center-mask diagnostic requires roi_center_u/roi_center_v")
                camera = camera.copy()
                camera_width, camera_height = camera.size
                mask_width = round(camera_width * self.center_mask_fraction)
                mask_height = round(camera_height * self.center_mask_fraction)
                target_x = float(row["roi_center_u"]) - x1
                target_y = float(row["roi_center_v"]) - y1
                mask_x1 = round(target_x - mask_width / 2)
                mask_y1 = round(target_y - mask_height / 2)
                mask_x1 = min(max(mask_x1, 0), camera_width - mask_width)
                mask_y1 = min(max(mask_y1, 0), camera_height - mask_height)
                # ImageNet mean color becomes approximately zero after the
                # standard normalization and does not introduce a class cue.
                camera.paste(
                    (124, 116, 104),
                    (mask_x1, mask_y1, mask_x1 + mask_width, mask_y1 + mask_height),
                )
            cameras = [camera]
        tensors = []
        for camera in cameras:
            camera = TF.resize(camera, self.image_size, antialias=True)
            tensor = TF.pil_to_tensor(camera).float().div_(255.0)
            tensor = TF.normalize(tensor, self.mean, self.std)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)
