import bisect
import json
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RDQ_DATA_ROOT", ROOT / "data/mmaud_official_train")).expanduser()
if not DATA.is_absolute():
    DATA = ROOT / DATA

TRAIN = """
seq0001 seq0019 seq0020 seq0021 seq0022
seq0023 seq0024 seq0026 seq0027 seq0028
seq0037 seq0046 seq0049 seq0050 seq0055
seq0061 seq0079 seq0089 seq0101 seq0102
""".split()

old_split = json.loads(
    (ROOT / "splits/mmaud_splits.json").read_text()
)
train_set = set(TRAIN)

new_split = {
    "schema_version": 1,
    "seed": 42,
    "allocation": "annotated_over_50_percent_20260923",
    "num_sequences": old_split["num_sequences"],
    "train_sub": TRAIN,
    "validation_sub": [
        s for s in old_split["validation_sub"]
        if s not in train_set
    ],
    "heldout_test_sub": [
        s for s in old_split["heldout_test_sub"]
        if s not in train_set
    ],
}

# 确保三个集合在 Sequence 层面严格互斥。
groups = [
    set(new_split[k])
    for k in ("train_sub", "validation_sub", "heldout_test_sub")
]
assert all(
    groups[i].isdisjoint(groups[j])
    for i in range(3)
    for j in range(i + 1, 3)
)

split_path = ROOT / "splits/mmaud_annotated20.json"
split_path.write_text(
    json.dumps(new_split, indent=2, ensure_ascii=False) + "\n"
)

# 使用与 E5 训练器相同的最近图像匹配阈值。
MAX_IMAGE_GAP = 0.04
WH = (1280, 960)

records = []
matched_images = set()
total_positive = 0

for seq in TRAIN:
    seq_root = DATA / seq
    images = sorted(
        (seq_root / "Image").glob("*.png"),
        key=lambda p: float(p.stem),
    )
    gt_files = sorted(
        (seq_root / "ground_truth").glob("*.npy"),
        key=lambda p: float(p.stem),
    )

    if not images or not gt_files:
        raise RuntimeError(f"{seq}: missing images or GT")

    image_times = [float(p.stem) for p in images]
    total_positive += sum(
        bool(p.read_text().strip())
        for p in (seq_root / "2d_detect").glob("*.txt")
    )

    for uid, gt_path in enumerate(gt_files):
        gt_time = float(gt_path.stem)
        pos = bisect.bisect_left(image_times, gt_time)
        candidates = [
            i for i in (pos - 1, pos)
            if 0 <= i < len(images)
        ]
        if not candidates:
            continue

        idx = min(
            candidates,
            key=lambda i: (abs(image_times[i] - gt_time), i),
        )
        image = images[idx]

        if abs(image_times[idx] - gt_time) > MAX_IMAGE_GAP:
            continue

        label = seq_root / "2d_detect" / (image.stem + ".txt")
        if not label.exists():
            continue

        lines = [
            line.split()
            for line in label.read_text().splitlines()
            if line.strip()
        ]
        if not lines:
            continue

        # 当前 E5 每张图仅支持一个监督框。
        if len(lines) != 1 or len(lines[0]) != 5:
            raise RuntimeError(f"Unsupported YOLO label: {label}")

        cls, cx, cy, w, h = map(float, lines[0])
        if int(cls) != 0 or not all(
            np.isfinite(v) for v in (cx, cy, w, h)
        ):
            raise RuntimeError(f"Invalid label: {label}")

        W, H = WH
        box = [
            (cx - w / 2) * W,
            (cy - h / 2) * H,
            (cx + w / 2) * W,
            (cy + h / 2) * H,
        ]

        # 修正 YOLO 浮点精度造成的微小越界
        eps = 0.01

        if not (
            -eps <= box[0] < box[2] <= W + eps
            and -eps <= box[1] < box[3] <= H + eps
        ):
            raise RuntimeError(f"Invalid bbox: {label}, box={box}")

        box = [
            max(0.0, min(float(W), box[0])),
            max(0.0, min(float(H), box[1])),
            max(0.0, min(float(W), box[2])),
            max(0.0, min(float(H), box[3])),
        ]

        if box[2] <= box[0] or box[3] <= box[1]:
            raise RuntimeError(f"Degenerate bbox: {label}")

        xyz = np.load(gt_path, allow_pickle=False).reshape(-1)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise RuntimeError(f"Invalid GT: {gt_path}")

        matched_images.add((seq, image.name))
        records.append({
            "sequence_id": seq,
            "query_uid": uid,
            "query_time": gt_time,
            # Dataset-relative provenance remains valid after repository/data
            # migration. Training binds images by sequence and basename.
            "image_path": str(image.relative_to(DATA)),
            "role": "labeled_train",
            "gt_xyz_m": xyz.tolist(),
            "range_m": float(np.linalg.norm(xyz)),
            "box_xyxy_px": box,
            "gt_2d_valid": True,
            "annotation_source": "reviewed_yolo_gt_shift",
        })

# 保留原始 Manifest 中属于新验证集的少量人工标注。
# 训练器根据 sequence_id 自动隔离 Train / Validation。
old_manifest = (
    ROOT / "manifests/multimodal_v1/vision_manual161_train.jsonl"
)
val_set = set(new_split["validation_sub"])

for line in old_manifest.read_text().splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    if item["sequence_id"] in val_set:
        records.append(item)

manifest_path = (
    ROOT / "manifests/multimodal_v1/"
    "vision_annotated20.jsonl"
)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(
    "".join(
        json.dumps(r, ensure_ascii=False) + "\n"
        for r in records
    )
)

# 复制原配置，不影响正在运行或已完成的旧实验。
old_config = yaml.safe_load(
    (ROOT / "configs/multimodal_v1/e5_full_v1.yaml").read_text()
)
old_config["experiment"]["name"] = "e5_annotated20"
old_config["experiment"]["output_dir"] = (
    "outputs/own_multimodal_research/"
    "multimodal_v1/e5_annotated20_seed42"
)
old_config["data"]["split_file"] = (
    "splits/mmaud_annotated20.json"
)
old_config["data"]["annotation_manifest"] = (
    "manifests/multimodal_v1/vision_annotated20.jsonl"
)

config_path = (
    ROOT / "configs/multimodal_v1/e5_annotated20.yaml"
)
config_path.write_text(
    yaml.safe_dump(
        old_config, allow_unicode=True, sort_keys=False
    )
)

train_records = sum(
    r["sequence_id"] in train_set for r in records
)
val_records = sum(
    r["sequence_id"] in val_set for r in records
)

print(f"Train sequences: {len(TRAIN)}")
print(f"Validation sequences: {len(val_set)}")
print(f"Test sequences: {len(new_split['heldout_test_sub'])}")
print(f"Positive YOLO files: {total_positive}")
print(f"Matched image labels: {len(matched_images)}")
print(f"Train manifest records: {train_records}")
print(f"Validation manifest records: {val_records}")
print(f"Split: {split_path}")
print(f"Manifest: {manifest_path}")
print(f"Config: {config_path}")
