from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Source = Literal['radar', 'rgb']


@dataclass(frozen=True)
class CandidateSet:
    score: torch.Tensor
    feature: torch.Tensor
    xyz: torch.Tensor
    xyz_valid: torch.Tensor
    box_xyxy_px: torch.Tensor
    box_valid: torch.Tensor
    batch_index: torch.Tensor
    source: Source
    source_index: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.score.shape[0])
        expected = {
            'score': (n,),
            'feature': (n, 128),
            'xyz': (n, 3),
            'xyz_valid': (n,),
            'box_xyxy_px': (n, 4),
            'box_valid': (n,),
            'batch_index': (n,),
            'source_index': (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}, got {tuple(value.shape)}')
        if self.feature.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError('feature must be floating point')
        if self.xyz.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError('xyz must be floating point')
        if self.box_xyxy_px.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError('box_xyxy_px must be floating point')
        if self.xyz_valid.dtype != torch.bool or self.box_valid.dtype != torch.bool:
            raise TypeError('valid flags must be bool')
        if self.batch_index.dtype != torch.long or self.source_index.dtype != torch.long:
            raise TypeError('batch_index/source_index must be long')
        if self.source not in ('radar', 'rgb'):
            raise ValueError(f'unsupported source {self.source!r}')
        if n:
            if not bool(torch.isfinite(self.score.float()).all()):
                raise ValueError('score must be finite')
            if not bool(torch.isfinite(self.feature.float()).all()):
                raise ValueError('feature must be finite')
            if self.source == 'radar':
                if not bool(self.xyz_valid.all()) or bool(self.box_valid.any()):
                    raise ValueError('radar candidates require xyz_valid=True and box_valid=False')
            else:
                if bool(self.xyz_valid.any()) or not bool(self.box_valid.all()):
                    raise ValueError('rgb candidates require xyz_valid=False and box_valid=True')

    @property
    def n(self) -> int:
        return int(self.score.shape[0])

    @classmethod
    def empty(cls, *, source: Source, device: torch.device | str, dtype: torch.dtype = torch.float32) -> 'CandidateSet':
        return cls(
            score=torch.empty(0, device=device, dtype=dtype),
            feature=torch.empty((0,128), device=device, dtype=dtype),
            xyz=torch.empty((0,3), device=device, dtype=dtype),
            xyz_valid=torch.empty(0, device=device, dtype=torch.bool),
            box_xyxy_px=torch.empty((0,4), device=device, dtype=dtype),
            box_valid=torch.empty(0, device=device, dtype=torch.bool),
            batch_index=torch.empty(0, device=device, dtype=torch.long),
            source=source,
            source_index=torch.empty(0, device=device, dtype=torch.long),
        )
