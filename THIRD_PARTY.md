# Third-party sources and design provenance

This project keeps third-party systems behind small adapters instead of copying
their training frameworks.

- **MMAUD / Multi-Modal-UAV** (`/home/jasoncui/projects/open_source/Multi-Modal-UAV`):
  MIT licensed. Its timeline-driven multimodal loading pattern informed the
  manifest design. Its Challenge-specific dataset code was not copied because
  it assumes `seqXXXX` directories and contains dataset-specific filename fixes.
- **RaCFormer** (`/home/jasoncui/projects/RaCFormer`): MIT licensed. The project
  uses its research principle that radar measurements can condition queries.
  MMDetection3D, voxelization and custom CUDA code were not copied.
- **timm**: maintained image-backbone provider, used through
  `timm.create_model(..., features_only=True)`.
- **PyTorch**: maintained `nn.MultiheadAttention`, optimization, AMP and data
  loading implementations.
- **torchvision**: maintained image transforms.
- **DETR** (`facebookresearch/detr`, Apache-2.0): design source for the compact
  2D sine/cosine positional encoding in `models/position.py`.

The project-specific integration, split logic, radar masking and controlled
ablation interface are implemented locally for MMAUD V1.
