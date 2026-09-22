"""Stage adapter over the frozen LiDAR V2 spatial candidate detector."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from rdq_uav.lidar_v2.model import LiDARUAVDetector

from ..contracts import LiDARPyramidContext
from ..registry import COMPONENTS


@COMPONENTS.register("lidar_v2_pyramid")
class LiDARV2PyramidAdapter(nn.Module):
    """Expose LiDAR V2 pyramid seams without changing its mathematics.

    The wrapped detector is the single owner of all learnable modules.  The
    adapter delegates to those exact module instances and retains the exact
    hierarchy object created by :meth:`prepare` through CandidateHead.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any] | None = None,
        *,
        detector: LiDARUAVDetector | None = None,
    ) -> None:
        super().__init__()
        if detector is None:
            if cfg is None:
                raise ValueError("cfg is required when detector is not provided")
            detector = LiDARUAVDetector(dict(cfg))
        elif cfg is not None:
            raise ValueError("provide either cfg or detector, not both")
        self.detector = detector

    def prepare(self, batch: Mapping[str, Any]) -> LiDARPyramidContext:
        hierarchy = self.detector.hierarchy(
            batch["points"], batch["point_batch_index"]
        )
        level0, level1, level2 = hierarchy.levels
        r0_pre = self.detector.voxel_embed(batch, hierarchy)
        return LiDARPyramidContext(
            batch=batch,
            hierarchy=hierarchy,
            level0=level0,
            level1=level1,
            level2=level2,
            r0_pre=r0_pre,
        )

    def run_stage0(
        self, r0_pre: torch.Tensor, ctx: LiDARPyramidContext
    ) -> torch.Tensor:
        return self.detector.encoder0(r0_pre, ctx.level0)

    def merge01(
        self, r0_post: torch.Tensor, ctx: LiDARPyramidContext
    ) -> torch.Tensor:
        return self.detector.merge01(
            r0_post,
            ctx.level0,
            ctx.level1,
            ctx.hierarchy.parent_l0_to_l1,
        )

    def run_stage1(
        self, r1_pre: torch.Tensor, ctx: LiDARPyramidContext
    ) -> torch.Tensor:
        return self.detector.encoder1(r1_pre, ctx.level1)

    def merge12(
        self, r1_post: torch.Tensor, ctx: LiDARPyramidContext
    ) -> torch.Tensor:
        return self.detector.merge12(
            r1_post,
            ctx.level1,
            ctx.level2,
            ctx.hierarchy.parent_l1_to_l2,
        )

    def run_stage2(
        self, r2_pre: torch.Tensor, ctx: LiDARPyramidContext
    ) -> torch.Tensor:
        return self.detector.encoder2(r2_pre, ctx.level2)

    def decode_to_fine(
        self,
        r0_post: torch.Tensor,
        r1_post: torch.Tensor,
        r2_post: torch.Tensor,
        ctx: LiDARPyramidContext,
    ) -> torch.Tensor:
        d1 = self.detector.up21(
            r1_post, r2_post, ctx.hierarchy.parent_l1_to_l2
        )
        return self.detector.final_norm(
            self.detector.up10(r0_post, d1, ctx.hierarchy.parent_l0_to_l1)
        )

    def candidate_head(
        self, fine_features: torch.Tensor, ctx: LiDARPyramidContext
    ) -> dict[str, Any]:
        logits, residual_xyz, pred_xyz = self.detector.head(
            fine_features, ctx.level0.centers
        )
        return {
            "logits": logits,
            "residual_xyz": residual_xyz,
            "pred_xyz": pred_xyz,
            "fine_features": fine_features,
            "voxel_centers": ctx.level0.centers,
            "source_token_id": torch.arange(
                len(fine_features), device=fine_features.device
            ),
            "batch_index": ctx.level0.batch_index,
            "layouts": ctx.hierarchy,
            "aux_stats": {
                "token_counts": [len(level.coords) for level in ctx.hierarchy.levels],
                "num_samples": int(ctx.batch["num_samples"]),
                "attention_backend": (
                    "explicit_pytorch_scaled_dot_product_with_additive_axis_bias"
                ),
            },
        }

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        ctx = self.prepare(batch)
        r0_post = self.run_stage0(ctx.r0_pre, ctx)
        r1_post = self.run_stage1(self.merge01(r0_post, ctx), ctx)
        r2_post = self.run_stage2(self.merge12(r1_post, ctx), ctx)
        fine = self.decode_to_fine(r0_post, r1_post, r2_post, ctx)
        return self.candidate_head(fine, ctx)
