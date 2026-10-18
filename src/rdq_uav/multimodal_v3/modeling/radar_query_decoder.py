"""Small single-target set decoder over ALL per-sample L0 features."""
from __future__ import annotations

import torch
from torch import nn


class RadarQueryDecoder(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.k = int(config.get('num_queries', 32))
        self.scale = float(config.get('coordinate_scale_m', 50.))
        if self.k < 1 or self.scale <= 0:
            raise ValueError('num_queries and coordinate_scale_m must be positive')
        self.queries = nn.Embedding(self.k, dim)
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerDecoderLayer(dim, int(config.get('heads', 4)), dim_feedforward=dim*2,
                                           dropout=0., batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, int(config.get('layers', 1)), norm=nn.LayerNorm(dim))
        self.cls = nn.Linear(dim, 2)
        self.xyz = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        # Small, distinct offsets. No sigmoid or GT-derived coordinates.
        nn.init.normal_(self.xyz[-1].weight, std=.001)
        nn.init.zeros_(self.xyz[-1].bias)

    def forward(self, dense):
        features, centers, batch = dense['fine_features'], dense['voxel_centers'], dense['batch_index']
        chunks = []
        for b in range(int(dense['aux_stats']['num_samples'])):
            ids = torch.nonzero(batch == b, as_tuple=False).flatten()
            if not len(ids):
                continue
            # Ragged per-sample attention prevents padding and cross-sample leakage.
            origin = centers[ids].float().mean(0)
            position = self.position(((centers[ids].float() - origin) / self.scale).to(features.dtype))
            memory = (features[ids] + position)[None]
            q = self.decoder(self.queries.weight[None], memory)[0]
            class_logits = self.cls(q)
            xyz = origin + self.xyz(q).float() * self.scale
            chunks.append((b, q, class_logits, xyz))
        if chunks:
            q = torch.cat([x[1] for x in chunks])
            logits = torch.cat([x[2] for x in chunks])
            xyz = torch.cat([x[3] for x in chunks])
            batches = torch.cat([torch.full((self.k,), x[0], device=batch.device, dtype=torch.long) for x in chunks])
        else:
            q = features.new_empty((0, features.shape[1])); logits = features.new_empty((0, 2))
            xyz = centers.new_empty((0, 3)); batches = batch[:0]
        # sigmoid(logit_fg - logit_bg) == softmax(two-class logits)[...,0]
        return dict(logits=logits[:, 0] - logits[:, 1], class_logits=logits, pred_xyz=xyz,
                    fine_features=q, batch_index=batches, prediction_kind='query',
                    source_token_id=torch.arange(len(q), device=batch.device), aux_stats=dense['aux_stats'])
