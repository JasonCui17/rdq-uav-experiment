# Multimodal V1 training-engineering audit

Scope: `tools/train_multimodal_v1_full.py` and its current E5 call chain. These
are engineering findings; they do not change the frozen model or experiment.

## Findings addressed by the Lightning path

1. The reference entry point combines dataset construction, model
   initialization, device transfer, optimization, validation, logging, and
   checkpoint persistence in one file. The new path separates Trainer,
   LightningModule, DataModule, and callbacks while reusing the scientific
   functions unchanged.
2. AMP, accumulation, clipping, optimizer stepping, and loop state were
   maintained manually. Lightning now owns these mechanics. The configured
   training set sizes must remain divisible by the accumulation factor, or an
   explicit final-partial-accumulation equivalence test is required.
3. Checkpoint restoration previously depended on custom RNG and optimizer
   bookkeeping. Native `.ckpt` files now contain module, optimizer, scheduler,
   callback, loop, hyperparameter, and DataModule generator state. Production
   last checkpoints are committed at epoch boundaries.
4. Optimizer parameter identity is already stable in the reference helper.
   The Lightning checkpoint retains `param_names` and validates them after
   restoration.
5. Formal validation must remain FP32 even under FP16 training. The Lightning
   validation step opens an inner no-autocast context and preserves the
   no-output denominators from the reference implementation.
6. T1/T2/T3 trainability is applied at `on_train_epoch_start`; the four named
   optimizer groups and their base learning rates remain unchanged.
7. The configured train/validation sequence sets are checked for overlap before
   any model is initialized. The current annotated-20 split has zero overlap.
8. The dominant expected runtime costs remain image decode/resize, DINO/Swin,
   HCI, and sparse LiDAR processing. Framework migration alone is not claimed
   to accelerate those kernels. CUDA profiling is required on the RTX 3070.

## Deliberately unresolved

- The current checkout contains a verified E5 definition but no complete,
  frozen E0-E4 build/loss specifications. The variant field is explicit and
  unsupported variants fail fast rather than silently running E5 mathematics.
- CUDA is unavailable in the current WSL execution environment, so real FP16
  stability, one full epoch, peak VRAM, and old/new GPU throughput remain
  hardware gates. CPU toy timings are not reported as GPU evidence.
- Multi-worker mid-epoch sample-exact resume is not claimed. `last.ckpt` is
  saved at completed epoch boundaries; resuming it is deterministic and avoids
  ambiguous prefetched iterator state.
