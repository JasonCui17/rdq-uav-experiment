# Runtime checkpoints

Checkpoint binaries are intentionally excluded from Git. Before training,
provide these paths as files/directories or local symbolic links:

- `dino_swin_t/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth`
- `lidar_v2/best_spatial.pt`

Run `python tools/check_assets.py` to verify them.
