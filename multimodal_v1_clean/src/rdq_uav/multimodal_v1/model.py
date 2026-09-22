"""P5 backbone integration plus the frozen P6/P7 Multimodal V1 model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import torch
from torch import nn
from .contracts import InteractionContext
from .interaction import GeometryBiHCIStack
from .interaction.geometry_local import project_omni_radtan
from .radar import LiDARV2PyramidAdapter
from .registry import COMPONENTS
from .vision import DINOAdapter, SwinPyramidAdapter
from .vision.swin_adapter import SwinPyramidOutput, SwinStageInput
from .candidate import RadarCandidateBuilder, RGBCandidateBuilder, HypothesisSet, associate_candidates
from .candidate.reliability_gate import ReliabilityGate
from .candidate.shared_query import TypedSharedQuery
from .decoder import FusionTransformerDecoder, FusionPredictionHeads, joint_suppression_indices
from .loss import FusionLoss, FusionTargets


@dataclass(frozen=True)
class P5BackboneOutput:
    """Outputs after three Pre-Stage HCI blocks and both original backbones."""
    radar: dict[str, Any]
    vision: SwinPyramidOutput
    hci_aux: tuple[dict[str, Any] | None, ...]
    # Interface extension only: exact post-stage tensors already computed by P5.
    # Exposing them prevents P7 from rerunning the Radar backbone.
    radar_stages: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class FusionOutput:
    fused_score: torch.Tensor
    box_xyxy_px: torch.Tensor
    xyz: torch.Tensor
    c_2d: torch.Tensor
    c_3d: torch.Tensor
    batch_index: torch.Tensor
    hypothesis_type: torch.Tensor
    gate_weights: torch.Tensor
    hypotheses: HypothesisSet
    losses: dict[str, torch.Tensor] | None = None
    aux: dict[str, Any] | None = None


@COMPONENTS.register("p5_multimodal_backbone")
class P5MultimodalBackbone(nn.Module):
    """Interleave HCI0--2 with the original Radar/Swin stages."""
    def __init__(self, radar: LiDARV2PyramidAdapter, vision: SwinPyramidAdapter, interaction: GeometryBiHCIStack) -> None:
        super().__init__(); self.radar=radar; self.vision=vision; self.interaction=interaction

    def forward(self,lidar_batch:Mapping[str,Any],images:torch.Tensor,context:InteractionContext,*,return_aux:bool=False)->P5BackboneOutput:
        if images.ndim!=4: raise ValueError(f"images must be [B,C,H,W], got {tuple(images.shape)}")
        batch_size=int(images.shape[0])
        if int(lidar_batch["num_samples"])!=batch_size: raise ValueError("LiDAR and vision batch sizes must match")
        context.validate(batch_size)
        radar_ctx=self.radar.prepare(lidar_batch); current=self.vision.prepare(images)
        levels=(radar_ctx.level0,radar_ctx.level1,radar_ctx.level2); radar_pre=radar_ctx.r0_pre
        radar_posts=[]; vision_outputs=[]; hci_aux=[]
        for stage_index in range(3):
            if current.index!=stage_index: raise RuntimeError(f"expected Swin pre-stage {stage_index}, got {current.index}")
            level=levels[stage_index]
            inter=self.interaction.forward_stage(stage_index,radar_pre,level.centers,level.batch_index,current.tokens,
                height=current.height,width=current.width,context=context,return_aux=return_aux)
            hci_aux.append(inter.aux)
            visual_stage=self.vision.run_stage(SwinStageInput(stage_index,inter.vision_tokens,current.height,current.width)); vision_outputs.append(visual_stage)
            if stage_index==0:
                radar_post=self.radar.run_stage0(inter.radar_features,radar_ctx); radar_pre=self.radar.merge01(radar_post,radar_ctx)
            elif stage_index==1:
                radar_post=self.radar.run_stage1(inter.radar_features,radar_ctx); radar_pre=self.radar.merge12(radar_post,radar_ctx)
            else: radar_post=self.radar.run_stage2(inter.radar_features,radar_ctx)
            radar_posts.append(radar_post)
            if visual_stage.next_input is None: raise RuntimeError(f"Swin stage {stage_index} did not expose next_input")
            current=visual_stage.next_input
        if current.index!=3: raise RuntimeError(f"expected Swin pre-stage 3, got {current.index}")
        stage3=self.vision.run_stage(current); vision_outputs.append(stage3)
        dino_features={f"p{index}":vision_outputs[index].feature for index in self.vision.backbone.out_indices}
        vision_pyramid=SwinPyramidOutput(tuple(vision_outputs),dino_features)
        fine=self.radar.decode_to_fine(radar_posts[0],radar_posts[1],radar_posts[2],radar_ctx)
        radar_output=self.radar.candidate_head(fine,radar_ctx)
        return P5BackboneOutput(radar_output,vision_pyramid,tuple(hci_aux),tuple(radar_posts))


@COMPONENTS.register("full_multimodal_v1")
class FullMultimodalV1(nn.Module):
    """Frozen P5 -> P6 -> P7 chain. File IO and calibration parsing stay outside forward."""
    def __init__(self, *, backbone:P5MultimodalBackbone, dino:DINOAdapter, radar_selector:Any,
                 radar_topk:int=50, vision_pre_topk:int=100, vision_topk:int=50, vision_nms_iou:float=.7,
                 geometry_gate_px:float=16., xyz_mean, xyz_std,
                 fusion_loss: FusionLoss | None=None) -> None:
        super().__init__()
        if backbone.vision is not dino.swin: raise ValueError("FullMultimodalV1 requires the exact shared DINO Swin adapter used by P5")
        # Selector must come from an independent multimodal config. Never mutate E0's frozen selector.
        if hasattr(radar_selector,'final') and int(radar_selector.final)!=int(radar_topk):
            raise ValueError(f'multimodal Radar selector final_topk must equal {radar_topk}; got {radar_selector.final}')
        self.backbone=backbone; self.dino=dino
        self.radar_candidates=RadarCandidateBuilder(radar_selector)
        self.rgb_candidates=RGBCandidateBuilder(pre_topk=vision_pre_topk,final_topk=vision_topk,nms_iou=vision_nms_iou)
        self.geometry_gate_px=float(geometry_gate_px)
        self.reliability_gate=ReliabilityGate(); self.shared_query=TypedSharedQuery()
        self.decoder=FusionTransformerDecoder(); self.heads=FusionPredictionHeads(xyz_mean=xyz_mean,xyz_std=xyz_std)
        self.fusion_loss=fusion_loss or FusionLoss()

    def forward(self,lidar_batch:Mapping[str,Any],images:torch.Tensor,context:InteractionContext,*,targets:FusionTargets|None=None,return_aux:bool=False)->FusionOutput:
        p5=self.backbone(lidar_batch,images,context,return_aux=return_aux)
        image_masks=images.new_zeros(images.shape[0],images.shape[-2],images.shape[-1])
        dino_output=self.dino.forward_from_pyramid(p5.vision,image_masks,allow_training_candidate_path=True)
        rset=self.radar_candidates(p5.radar)
        vset=self.rgb_candidates(dino_output,context.projection.image_size_wh)
        hypotheses=associate_candidates(rset,vset,geometry_gate_px=self.geometry_gate_px,projection=context.projection)
        gate=self.reliability_gate(hypotheses); query=self.shared_query(hypotheses)
        level2=p5.radar["layouts"].levels[2]
        decoded,decoder_aux=self.decoder(query,hypotheses.batch_index,p5.radar_stages[2],level2.batch_index,p5.vision.features[2],
            sample_m_R=context.m_R,sample_m_V=context.m_V,return_aux=return_aux)
        projected=query.new_zeros((hypotheses.n,2))
        rid=torch.nonzero(hypotheses.m_R).flatten()
        if len(rid):
            pixels,_=project_omni_radtan(hypotheses.radar_xyz[rid],hypotheses.batch_index[rid],context.projection)
            projected=projected.index_copy(0,rid,pixels.to(projected.dtype))
        pred=self.heads(decoded,hypotheses,projected,context.projection.image_size_wh)
        losses=None
        if targets is not None:
            losses=self.fusion_loss(fused_score=gate.fused_score,pred_box=pred.box_xyxy_px,pred_xyz=pred.xyz,
                c2d_logit=pred.c2d_logit,c3d_logit=pred.c3d_logit,hypotheses=hypotheses,targets=targets,
                source_image_size_wh=context.projection.image_size_wh)
        aux=None
        if return_aux:
            aux={"p5":p5,"dino":dino_output,"radar_candidates":rset,"rgb_candidates":vset,
                 "hypotheses_pre_postprocess":hypotheses,"decoder":decoder_aux,
                 "joint_score":gate.joint_score,"joint_logit":gate.joint_logit}
        fused_score=gate.fused_score; box=pred.box_xyxy_px; xyz=pred.xyz
        c2d=torch.sigmoid(pred.c2d_logit); c3d=torch.sigmoid(pred.c3d_logit); weights=gate.weights; out_h=hypotheses
        if not self.training and hypotheses.n:
            keep=joint_suppression_indices(fused_score,box,xyz,hypotheses.batch_index,iou_threshold=.7,radius_m=1.0,final_topk=50)
            fused_score=fused_score[keep]; box=box[keep]; xyz=xyz[keep]; c2d=c2d[keep]; c3d=c3d[keep]; weights=weights[keep]; out_h=hypotheses.index_select(keep)
            if aux is not None: aux["postprocess_keep_indices"]=keep
        return FusionOutput(fused_score,box,xyz,c2d,c3d,out_h.batch_index,out_h.hypothesis_type,weights,out_h,losses,aux)
