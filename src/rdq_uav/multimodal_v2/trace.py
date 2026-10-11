"""Opt-in, read-only capture of the existing V2 forward path."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import torch

from .radar_model import SpatialBlock

_MISSING = object()


class TraceRecorder:
    """Detach tensors as each stage runs; remove every observer on exit."""

    def __init__(self, model, directory: Path, *, focus_voxel: int = 0,
                 max_tensor_mb: int = 256):
        self.model, self.directory = model, Path(directory)
        self.focus_voxel = int(focus_voxel)
        self.max_bytes = int(max_tensor_mb) * 1024 * 1024
        self.tensors = {}
        self.omitted = []
        self.events = {}
        self.handles = []
        self.restores = []
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "tensors").mkdir(exist_ok=True)

    def save(self, name: str, value):
        if value is None or not torch.is_tensor(value):
            return
        shape = list(value.shape)
        count = value.numel()
        metadata = {"shape": shape, "dtype": str(value.dtype), "elements": count}
        if count * value.element_size() > self.max_bytes:
            metadata["status"] = "omitted_limit"
            self.omitted.append(name)
            self.tensors[name] = metadata
            return
        array = value.detach().cpu().numpy()
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        path = self.directory / "tensors" / f"{safe}.npy"
        np.save(path, array, allow_pickle=False)
        metadata.update(status="saved", file=f"tensors/{safe}.npy",
                        nonfinite=int(np.count_nonzero(~np.isfinite(array))) if np.issubdtype(array.dtype, np.number) else 0)
        if count and np.issubdtype(array.dtype, np.number) and np.isfinite(array).any():
            finite = array[np.isfinite(array)].astype(np.float64)
            metadata.update(min=float(finite.min()), max=float(finite.max()),
                            mean=float(finite.mean()), std=float(finite.std()))
        self.tensors[name] = metadata

    def save_many(self, prefix: str, values: dict):
        for name, value in values.items():
            self.save(f"{prefix}.{name}", value)

    def event(self, name: str, value):
        self.events[name] = value

    def _set_observer(self, obj, name, callback):
        previous = getattr(obj, name, _MISSING)
        self.restores.append((obj, name, previous))
        setattr(obj, name, callback)

    def _hook(self, name, module):
        def capture(_, args, result):
            for i, value in enumerate(args):
                if torch.is_tensor(value):
                    self.save(f"{name}.input_{i}", value)
            if torch.is_tensor(result):
                self.save(f"{name}.output", result)
            elif isinstance(result, tuple):
                for i, value in enumerate(result):
                    if torch.is_tensor(value): self.save(f"{name}.output_{i}", value)
        self.handles.append(module.register_forward_hook(capture))

    def __enter__(self):
        self._set_observer(self.model, "trace_projection_observer",
                           lambda values: self.save_many("projection", values))
        lidar = getattr(self.model, "lidar", None)
        detector = getattr(lidar, "detector", None)
        if detector is not None:
            self._set_observer(detector, "trace_observer", lambda hierarchy: self._focus_hierarchy(detector, hierarchy))
            if hasattr(detector.voxel_embed, "trace_observer"):
                self._set_observer(detector.voxel_embed, "trace_observer",
                                   lambda values: self.save_many("frontend", values))
            if hasattr(detector.voxel_embed, "vqsa"):
                self._hook("frontend.slot_attention", detector.voxel_embed.vqsa)
            for name in ("merge01", "merge12", "up21", "up10"):
                self._set_observer(getattr(detector, name), "trace_observer",
                                   lambda values, name=name: self.save_many(f"radar.{name}.internal", values))
            for name in ("voxel_embed", "merge01", "merge12", "up21", "up10", "final_norm", "head"):
                self._hook(f"radar.{name}", getattr(detector, name))
            for name, module in (("linear0", detector.head.cls[0]),
                                 ("gelu", detector.head.cls[1]),
                                 ("linear1", detector.head.cls[2])):
                self._hook(f"radar.head.cls.{name}", module)
            for level in range(3):
                encoder = getattr(detector, f"encoder{level}")
                self._set_observer(encoder, "trace_observer",
                                   lambda block, groups, shift, level=level: self._groups_event(level, block, groups, shift))
                self._hook(f"radar.encoder{level}.position", encoder.position)
                for block_index, block in enumerate(encoder.blocks):
                    prefix = f"radar.encoder{level}.block{block_index}"
                    self._hook(prefix, block)
                    if isinstance(block, SpatialBlock):
                        self._set_observer(block, "trace_focus_token", self.focus_voxel)
                        self._set_observer(block, "trace_observer",
                                           lambda values, prefix=prefix: self.save_many(f"{prefix}.window", values))
            selector = lidar.builder.selector
            self._set_observer(selector, "trace_observer", self._selector_event)
        interaction = self.model.interaction
        self._set_observer(interaction, "trace_visual_observer",
                           lambda level, values: self.save_many(f"visual_read.level{level}", values))
        self._hook("interaction.radar_reads_vision", interaction.radar_reads_vision)
        self._hook("interaction.vision_reads_radar", interaction.vision_reads_radar)
        self._set_observer(self.model.scoring, "trace_observer",
                           lambda value: self.event("association", value))
        self._set_observer(self.model.scoring.score_head, "trace_observer", self._score_event)
        return self

    def _groups_event(self, level, block, groups, shift):
        prefix = f"radar.encoder{level}.block{block}.groups"
        ids = torch.cat(groups) if groups else torch.empty(0, dtype=torch.long)
        offsets = torch.tensor([0] + [len(group) for group in groups], dtype=torch.long).cumsum(0)
        self.save(f"{prefix}.token_ids", ids)
        self.save(f"{prefix}.offsets", offsets)
        self.event(prefix, {"shift": shift, "count": len(groups)})

    def _focus_hierarchy(self, detector, hierarchy):
        l0 = self.focus_voxel
        if l0 < 0 or l0 >= len(hierarchy.levels[0].coords):
            raise ValueError(f"focus voxel {l0} outside L0 [0,{len(hierarchy.levels[0].coords)})")
        l1 = int(hierarchy.parent_l0_to_l1[l0])
        l2 = int(hierarchy.parent_l1_to_l2[l1])
        for level, token in enumerate((l0, l1, l2)):
            for block in getattr(detector, f"encoder{level}").blocks:
                block.trace_focus_token = token
        self.event("focus", {"l0": l0, "l1": l1, "l2": l2})

    def _selector_event(self, value):
        prefix = f"selector.batch{value['batch_index']}"
        for key in ("order", "raw", "pool", "final", "suppressed"):
            self.save(f"{prefix}.{key}", value[key])
        self.event(prefix, {"kept_local": value["kept_local"],
                            "radius_m": value["radius_m"], "final_topk": value["final_topk"],
                            "evaluated_count": value["evaluated_count"]})

    def _score_event(self, value):
        source = value["source"]
        self.save_many(f"scoring.{source}", {k: v for k, v in value.items() if k != "source"})

    def __exit__(self, exc_type, exc, traceback):
        for handle in reversed(self.handles): handle.remove()
        for obj, name, previous in reversed(self.restores):
            if previous is _MISSING: delattr(obj, name)
            else: setattr(obj, name, previous)
        self.handles.clear(); self.restores.clear()
        (self.directory / "tensor_index.json").write_text(json.dumps(self.tensors, indent=2))
        (self.directory / "events.json").write_text(json.dumps(self.events, indent=2))
        return False
