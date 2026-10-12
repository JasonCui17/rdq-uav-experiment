"""Epoch wall-clock accounting. GPU sync only at epoch/validation boundaries."""
from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback
from .radar_lora import parameter_report


class EfficiencyRecorder(Callback):
    def __init__(self, output):
        self.output = Path(output)
        self.start = None
        self.validation_start = None
        self.validation_s = 0.
        self.samples = self.batches = 0

    @staticmethod
    def sync(module):
        if module.device.type == 'cuda': torch.cuda.synchronize(module.device)

    def on_fit_start(self, trainer, module):
        self.output.mkdir(parents=True, exist_ok=True)
        detector = module.runtime.lidar_detector
        report = parameter_report(detector)
        report.update(train_mode=module.runtime.initialization.get('radar_train_mode'),
                      torch_version=torch.__version__, device=str(module.device),
                      gpu_name=torch.cuda.get_device_name(module.device) if module.device.type=='cuda' else None,
                      precision=str(trainer.precision), batch_size=module.config['training']['batch_size'],
                      accumulate=trainer.accumulate_grad_batches, requested_max_steps=trainer.max_steps)
        (self.output/'parameters.json').write_text(json.dumps(report,indent=2))

    def on_train_epoch_start(self, trainer, module):
        self.sync(module)
        self.start = time.perf_counter()
        self.step_start = trainer.global_step
        self.validation_s = 0.
        self.samples = self.batches = 0
        if module.device.type=='cuda': torch.cuda.reset_peak_memory_stats(module.device)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.samples += len(batch['m_R'])
        self.batches += 1

    def on_validation_start(self, trainer, module):
        if trainer.sanity_checking or self.start is None: return
        self.sync(module)
        self.validation_start=time.perf_counter()

    def on_validation_end(self, trainer, module):
        if self.validation_start is None: return
        self.sync(module)
        self.validation_s += time.perf_counter()-self.validation_start
        self.validation_start=None

    def on_train_epoch_end(self, trainer, module):
        if self.start is None: return
        self.sync(module)
        wall=time.perf_counter()-self.start
        train=max(0.,wall-self.validation_s)
        steps=trainer.global_step-self.step_start
        row=dict(epoch=trainer.current_epoch,global_step=trainer.global_step,optimizer_updates=steps,
                 batches=self.batches,samples=self.samples,epoch_wall_s=wall,
                 train_wall_s=train,validation_wall_s=self.validation_s,
                 train_s_per_update=train/steps if steps else None,
                 train_samples_per_s=self.samples/train if train else None,
                 epoch_peak_allocated_mib=torch.cuda.max_memory_allocated(module.device)/2**20 if module.device.type=='cuda' else None,
                 note='Wall includes data, model, training metrics/logging; epoch peak includes validation. Checkpoint callback overhead may be excluded.')
        with (self.output/'timing.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        self.start=None
