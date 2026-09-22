from __future__ import annotations

from dataclasses import dataclass

import torch

HYP_RV = 0
HYP_R = 1
HYP_V = 2


@dataclass(frozen=True)
class HypothesisSet:
    radar_feature: torch.Tensor
    vision_feature: torch.Tensor
    radar_score: torch.Tensor
    vision_score: torch.Tensor
    radar_xyz: torch.Tensor
    radar_xyz_valid: torch.Tensor
    vision_box_xyxy_px: torch.Tensor
    vision_box_valid: torch.Tensor
    association_info: torch.Tensor
    m_R: torch.Tensor
    m_V: torch.Tensor
    hypothesis_type: torch.Tensor
    batch_index: torch.Tensor
    radar_source_index: torch.Tensor
    vision_source_index: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.batch_index.shape[0])
        expected = {
            'radar_feature': (n,128),
            'vision_feature': (n,128),
            'radar_score': (n,),
            'vision_score': (n,),
            'radar_xyz': (n,3),
            'radar_xyz_valid': (n,),
            'vision_box_xyxy_px': (n,4),
            'vision_box_valid': (n,),
            'association_info': (n,4),
            'm_R': (n,),
            'm_V': (n,),
            'hypothesis_type': (n,),
            'batch_index': (n,),
            'radar_source_index': (n,),
            'vision_source_index': (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}, got {tuple(value.shape)}')
        for name in ('radar_xyz_valid','vision_box_valid','m_R','m_V'):
            if getattr(self,name).dtype != torch.bool:
                raise TypeError(f'{name} must be bool')
        for name in ('hypothesis_type','batch_index','radar_source_index','vision_source_index'):
            if getattr(self,name).dtype != torch.long:
                raise TypeError(f'{name} must be long')
        if n:
            allowed = (self.hypothesis_type >= 0) & (self.hypothesis_type <= 2)
            if not bool(allowed.all()):
                raise ValueError('hypothesis_type must be RV/R/V')
            rv = self.hypothesis_type == HYP_RV
            rr = self.hypothesis_type == HYP_R
            vv = self.hypothesis_type == HYP_V
            if not bool((self.m_R == (rv|rr)).all()) or not bool((self.m_V == (rv|vv)).all()):
                raise ValueError('presence mask disagrees with hypothesis type')

    @property
    def n(self) -> int:
        return int(self.batch_index.shape[0])


    def index_select(self, index: torch.Tensor) -> "HypothesisSet":
        if index.dtype != torch.long or index.ndim != 1:
            raise TypeError("index must be 1D long")
        return HypothesisSet(
            self.radar_feature[index], self.vision_feature[index],
            self.radar_score[index], self.vision_score[index],
            self.radar_xyz[index], self.radar_xyz_valid[index],
            self.vision_box_xyxy_px[index], self.vision_box_valid[index],
            self.association_info[index], self.m_R[index], self.m_V[index],
            self.hypothesis_type[index], self.batch_index[index],
            self.radar_source_index[index], self.vision_source_index[index],
        )

    @classmethod
    def empty(cls, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> 'HypothesisSet':
        zf=lambda *s: torch.empty(s,device=device,dtype=dtype)
        zb=lambda *s: torch.empty(s,device=device,dtype=torch.bool)
        zl=lambda *s: torch.empty(s,device=device,dtype=torch.long)
        return cls(zf(0,128),zf(0,128),zf(0),zf(0),zf(0,3),zb(0),zf(0,4),zb(0),zf(0,4),zb(0),zb(0),zl(0),zl(0),zl(0),zl(0))
