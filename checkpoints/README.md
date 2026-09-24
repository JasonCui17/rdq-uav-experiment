# Runtime checkpoints

Checkpoint binaries are intentionally excluded from Git. Before training,
provide these paths as files/directories or local symbolic links:

- `dino_swin_t/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth`
- `lidar_v2/best_spatial.pt`

Run `python tools/check_assets.py` to verify them.

External locations can be selected without editing YAML:

```bash
export RDQ_LIDAR_CHECKPOINT=/path/to/best_spatial.pt
export RDQ_DINO_CHECKPOINT=/path/to/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth
```
