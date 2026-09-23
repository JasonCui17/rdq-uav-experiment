#!/usr/bin/env python3

import json
from pathlib import Path

import gradio as gr
import numpy as np
import yaml
from PIL import Image, ImageDraw

from rdq_uav.calibration.omni import OmniRadtanCamera, transform_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path("/home/jasoncui/datasets/MMAUD/official/train")

CAMERA_CONFIG = PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
CAMERA_CALIB = PROJECT_ROOT / "calibration/official_left_fitted_calibration.json"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/multimodal_v1/manual_calibration"

LEFT_W = 1280
LEFT_H = 960

# 仅用于提示，不参与最终标定
SUGGEST_HALF_W = 20
SUGGEST_HALF_H = 20


# ============================================================
# Camera calibration
# ============================================================

camera_cfg = yaml.safe_load(CAMERA_CONFIG.read_text())
calib = json.loads(CAMERA_CALIB.read_text())

CAMERA = OmniRadtanCamera.from_config(
    camera_cfg["cameras"]["left"]
)

left_calib = calib["cameras"]["left"]

ROTATION = np.asarray(
    left_calib["rotation_camera_from_gt"],
    dtype=np.float64,
).reshape(3, 3)


TRANSLATION = np.asarray(
    left_calib["translation_camera_from_gt_m"],
    dtype=np.float64,
).reshape(3)

TIME_OFFSET_S = float(calib["time_offset_s"])


# ============================================================
# Sequences
# ============================================================

SEQUENCES = sorted(
    [
        p.name
        for p in DATA_ROOT.iterdir()
        if p.is_dir()
        and (p / "Image").exists()
        and (p / "ground_truth").exists()
    ]
)

if not SEQUENCES:
    raise RuntimeError(f"No sequences found in {DATA_ROOT}")

CACHE = {}
ANNOTATION_CACHE = {}


def load_sequence(seq):
    if seq in CACHE:
        return CACHE[seq]

    seq_dir = DATA_ROOT / seq

    images = sorted(
        (seq_dir / "Image").glob("*.png"),
        key=lambda p: float(p.stem),
    )

    gt_paths = sorted(
        (seq_dir / "ground_truth").glob("*.npy"),
        key=lambda p: float(p.stem),
    )

    gt_times = np.asarray(
        [float(p.stem) for p in gt_paths],
        dtype=np.float64,
    )

    if gt_paths:
        gt_xyz = np.stack(
            [
                np.asarray(
                    np.load(p, allow_pickle=False),
                    dtype=np.float64,
                ).reshape(3)
                for p in gt_paths
            ],
            axis=0,
        )
    else:
        gt_xyz = np.empty((0, 3), dtype=np.float64)

    data = {
        "images": images,
        "image_map": {p.name: p for p in images},
        "gt_times": gt_times,
        "gt_xyz": gt_xyz,
    }

    CACHE[seq] = data
    return data


# ============================================================
# Annotations
# ============================================================

def annotation_file(seq):
    out_dir = OUTPUT_ROOT / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "bbox_annotations.json"


def load_annotations(seq):
    if seq in ANNOTATION_CACHE:
        return ANNOTATION_CACHE[seq]

    path = annotation_file(seq)

    if path.exists():
        records = json.loads(path.read_text())
        annotations = {
            r["image_name"]: r
            for r in records
        }
    else:
        annotations = {}

    ANNOTATION_CACHE[seq] = annotations
    return annotations


def save_annotations(seq):
    annotations = load_annotations(seq)

    records = sorted(
        annotations.values(),
        key=lambda x: x["image_time"],
    )

    path = annotation_file(seq)
    tmp = path.with_suffix(".tmp")

    tmp.write_text(
        json.dumps(records, indent=2)
    )

    tmp.replace(path)


# ============================================================
# GT interpolation
# ============================================================

def interpolate_gt(seq, image_time):
    data = load_sequence(seq)

    gt_times = data["gt_times"]
    gt_xyz = data["gt_xyz"]

    if len(gt_times) == 0:
        return None

    # 注意：
    # 当前 Δt 仅作为粗标定初值，不是最终固定参数
    query_time = float(image_time) + TIME_OFFSET_S

    i = int(
        np.searchsorted(gt_times, query_time)
    )

    if i <= 0:
        return {
            "xyz": gt_xyz[0],
            "gt_time": float(gt_times[0]),
            "query_time": query_time,
            "mode": "nearest",
        }

    if i >= len(gt_times):
        return {
            "xyz": gt_xyz[-1],
            "gt_time": float(gt_times[-1]),
            "query_time": query_time,
            "mode": "nearest",
        }

    t0 = gt_times[i - 1]
    t1 = gt_times[i]

    p0 = gt_xyz[i - 1]
    p1 = gt_xyz[i]

    if t1 <= t0:
        xyz = p0
    else:
        alpha = (query_time - t0) / (t1 - t0)
        xyz = (1.0 - alpha) * p0 + alpha * p1

    return {
        "xyz": xyz,
        "gt_time": query_time,
        "query_time": query_time,
        "mode": "interpolated",
    }


# ============================================================
# Projection
# ============================================================

def project_gt(seq, image_time):
    record = interpolate_gt(seq, image_time)

    if record is None:
        return None

    xyz = record["xyz"]

    point_camera = transform_points(
        xyz.reshape(1, 3),
        ROTATION,
        TRANSLATION,
    )

    pixels, valid = CAMERA.project(
        point_camera,
        require_in_image=True,
    )

    if not bool(valid[0]):
        return None

    u, v = map(float, pixels[0])

    return {
        "u": u,
        "v": v,
        "xyz": xyz.tolist(),
        "gt_time": float(record["gt_time"]),
        "query_time": float(record["query_time"]),
        "mode": record["mode"],
    }


def suggestion_bbox(projection):
    if projection is None:
        return None

    u = projection["u"]
    v = projection["v"]

    return [
        max(0.0, u - SUGGEST_HALF_W),
        max(0.0, v - SUGGEST_HALF_H),
        min(LEFT_W - 1.0, u + SUGGEST_HALF_W),
        min(LEFT_H - 1.0, v + SUGGEST_HALF_H),
    ]


# ============================================================
# Manual bbox
# ============================================================

def get_bbox(clicks):
    (xa, ya), (xb, yb) = clicks

    x1, x2 = sorted(
        [float(xa), float(xb)]
    )

    y1, y2 = sorted(
        [float(ya), float(yb)]
    )

    return (
        max(0.0, min(LEFT_W - 1.0, x1)),
        max(0.0, min(LEFT_H - 1.0, y1)),
        max(0.0, min(LEFT_W - 1.0, x2)),
        max(0.0, min(LEFT_H - 1.0, y2)),
    )


# ============================================================
# Render
# ============================================================

def render(seq, image_name, clicks=None):
    data = load_sequence(seq)

    if not image_name:
        return None

    p = data["image_map"][image_name]
    image_time = float(p.stem)

    with Image.open(p) as raw:
        im = (
            raw.convert("RGB")
            .crop((0, 0, LEFT_W, LEFT_H))
        )

    draw = ImageDraw.Draw(im)

    # --------------------------------------------------------
    # 粗标定：
    # 只画空心橙色框
    # 不再画中心十字
    # --------------------------------------------------------

    projection = project_gt(
        seq,
        image_time,
    )

    sb = suggestion_bbox(projection)

    if sb is not None:
        draw.rectangle(
            tuple(sb),
            outline="orange",
            width=2,
        )

        draw.text(
            (
                sb[0],
                max(0, sb[1] - 16),
            ),
            "COARSE",
            fill="orange",
        )

    # --------------------------------------------------------
    # 已保存人工框
    # --------------------------------------------------------

    annotations = load_annotations(seq)

    old = annotations.get(image_name)

    if old is not None:
        b = old["bbox_xyxy"]

        draw.rectangle(
            tuple(b),
            outline="lime",
            width=2,
        )

        draw.text(
            (
                b[0],
                max(0, b[1] - 16),
            ),
            "SAVED",
            fill="lime",
        )

    # --------------------------------------------------------
    # 当前人工操作
    # --------------------------------------------------------

    clicks = clicks or []

    # 点击点只画极小点，不画十字
    for x, y in clicks:
        r = 2

        draw.ellipse(
            (
                x - r,
                y - r,
                x + r,
                y + r,
            ),
            fill="yellow",
        )

    if len(clicks) == 2:
        x1, y1, x2, y2 = get_bbox(clicks)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=2,
        )

    return im


# ============================================================
# Status
# ============================================================

def frame_status(seq, image_name, extra=""):
    data = load_sequence(seq)

    images = data["images"]

    if not image_name:
        return "No image"

    names = [p.name for p in images]

    idx = names.index(image_name)

    image_time = float(
        Path(image_name).stem
    )

    projection = project_gt(
        seq,
        image_time,
    )

    annotations = load_annotations(seq)

    lines = [
        f"Sequence: {seq}",
        f"Frame: {idx + 1}/{len(images)}",
        f"image_time: {image_time:.6f}",
        f"initial Δt: {TIME_OFFSET_S:+.6f} s",
    ]

    if projection is None:
        lines.append(
            "coarse projection: INVALID"
        )
    else:
        lines.append(
            "coarse projection: "
            f"({projection['u']:.1f}, "
            f"{projection['v']:.1f})"
        )

    lines.append(
        "manual saved: "
        + (
            "YES"
            if image_name in annotations
            else "NO"
        )
    )

    lines.append(
        f"sequence annotations: "
        f"{len(annotations)}"
    )

    if extra:
        lines.append(extra)

    return "\n".join(lines)


# ============================================================
# UI callbacks
# ============================================================

def on_sequence_change(seq):
    data = load_sequence(seq)

    choices = [
        p.name
        for p in data["images"]
    ]

    if not choices:
        return (
            gr.Dropdown(
                choices=[],
                value=None,
            ),
            None,
            [],
            f"{seq}: no images",
        )

    first = choices[0]

    return (
        gr.Dropdown(
            choices=choices,
            value=first,
        ),
        render(seq, first, []),
        [],
        frame_status(seq, first),
    )


def on_image_change(seq, image_name):
    if not image_name:
        return (
            None,
            [],
            "No image",
        )

    return (
        render(seq, image_name, []),
        [],
        frame_status(
            seq,
            image_name,
        ),
    )


def on_select(
    seq,
    image_name,
    clicks,
    evt: gr.SelectData,
):
    if not image_name:
        return None, [], "No image"

    if evt.index is None:
        return (
            render(
                seq,
                image_name,
                clicks,
            ),
            clicks,
            frame_status(
                seq,
                image_name,
            ),
        )

    x, y = evt.index

    x = float(x)
    y = float(y)

    clicks = list(clicks or [])

    if len(clicks) >= 2:
        clicks = []

    clicks.append([x, y])

    if len(clicks) == 1:
        msg = (
            f"第1点=({x:.1f},{y:.1f})，"
            "再点 bbox 对角点"
        )
    else:
        x1, y1, x2, y2 = get_bbox(
            clicks
        )

        msg = (
            f"bbox=({x1:.1f},{y1:.1f})"
            f"→({x2:.1f},{y2:.1f})"
        )

    return (
        render(
            seq,
            image_name,
            clicks,
        ),
        clicks,
        frame_status(
            seq,
            image_name,
            msg,
        ),
    )


def move_image(
    seq,
    image_name,
    offset,
):
    data = load_sequence(seq)

    names = [
        p.name
        for p in data["images"]
    ]

    if not names:
        return None

    if image_name not in names:
        return names[0]

    idx = names.index(image_name)

    new_idx = max(
        0,
        min(
            len(names) - 1,
            idx + offset,
        ),
    )

    return names[new_idx]


def previous_image(seq, image_name):
    target = move_image(
        seq,
        image_name,
        -1,
    )

    return (
        target,
        render(seq, target, []),
        [],
        frame_status(seq, target),
    )


def next_image(seq, image_name):
    target = move_image(
        seq,
        image_name,
        +1,
    )

    return (
        target,
        render(seq, target, []),
        [],
        frame_status(seq, target),
    )


def save_and_next(
    seq,
    image_name,
    clicks,
):
    clicks = list(clicks or [])

    if len(clicks) != 2:
        return (
            image_name,
            render(
                seq,
                image_name,
                clicks,
            ),
            clicks,
            frame_status(
                seq,
                image_name,
                "需要先点击 bbox 两个对角点",
            ),
        )

    x1, y1, x2, y2 = get_bbox(
        clicks
    )

    if x2 - x1 < 1 or y2 - y1 < 1:
        return (
            image_name,
            render(
                seq,
                image_name,
                clicks,
            ),
            clicks,
            frame_status(
                seq,
                image_name,
                "bbox 太小",
            ),
        )

    p = load_sequence(seq)[
        "image_map"
    ][image_name]

    image_time = float(p.stem)

    projection = project_gt(
        seq,
        image_time,
    )

    record = {
        "sequence": seq,
        "image_name": image_name,
        "image_path": str(p),
        "image_time": image_time,

        "bbox_xyxy": [
            x1,
            y1,
            x2,
            y2,
        ],

        "center_u":
            (x1 + x2) / 2.0,

        "center_v":
            (y1 + y2) / 2.0,

        "width_px":
            x2 - x1,

        "height_px":
            y2 - y1,

        "image_width": LEFT_W,
        "image_height": LEFT_H,

        "initial_projection":
            projection,

        "annotation_source":
            "manual_bbox",
    }

    annotations = load_annotations(seq)

    annotations[image_name] = record

    save_annotations(seq)

    target = move_image(
        seq,
        image_name,
        +1,
    )

    return (
        target,
        render(seq, target, []),
        [],
        frame_status(
            seq,
            target,
            f"已保存；当前 {seq} "
            f"累计 {len(annotations)} 个框",
        ),
    )


# ============================================================
# Initial state
# ============================================================

INITIAL_SEQ = SEQUENCES[0]

INITIAL_IMAGES = [
    p.name
    for p in load_sequence(
        INITIAL_SEQ
    )["images"]
]

INITIAL_IMAGE = (
    INITIAL_IMAGES[0]
    if INITIAL_IMAGES
    else None
)


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="MMAUD Interactive Calibration V3"
) as demo:

    gr.Markdown(
        """
### MMAUD 交互式标定 V3

- **橙色空心框**：当前粗标定预测区域
- **红框**：当前人工 bbox
- **绿框**：已经保存的人工 bbox
- 中心不再显示十字，不遮挡远距离小无人机
"""
    )

    click_state = gr.State([])

    with gr.Row():

        sequence_dropdown = gr.Dropdown(
            choices=SEQUENCES,
            value=INITIAL_SEQ,
            label="Sequence",
            interactive=True,
        )

        image_dropdown = gr.Dropdown(
            choices=INITIAL_IMAGES,
            value=INITIAL_IMAGE,
            label="Image / Timestamp",
            interactive=True,
            filterable=True,
        )

    image = gr.Image(
        value=render(
            INITIAL_SEQ,
            INITIAL_IMAGE,
            [],
        ),
        type="pil",
        interactive=True,
        show_label=False,
        height=720,
    )

    status = gr.Textbox(
        value=frame_status(
            INITIAL_SEQ,
            INITIAL_IMAGE,
        ),
        label="状态",
        interactive=False,
        lines=7,
    )

    with gr.Row():

        prev_btn = gr.Button(
            "上一张"
        )

        next_btn = gr.Button(
            "下一张"
        )

        skip_btn = gr.Button(
            "跳过"
        )

        save_btn = gr.Button(
            "保存人工框并下一张",
            variant="primary",
        )

    sequence_dropdown.change(
        on_sequence_change,
        inputs=[
            sequence_dropdown,
        ],
        outputs=[
            image_dropdown,
            image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    image_dropdown.change(
        on_image_change,
        inputs=[
            sequence_dropdown,
            image_dropdown,
        ],
        outputs=[
            image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    image.select(
        on_select,
        inputs=[
            sequence_dropdown,
            image_dropdown,
            click_state,
        ],
        outputs=[
            image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    prev_btn.click(
        previous_image,
        inputs=[
            sequence_dropdown,
            image_dropdown,
        ],
        outputs=[
            image_dropdown,
            image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    next_btn.click(
        next_image,
        inputs=[
            sequence_dropdown,
            image_dropdown,
        ],
        outputs=[
            image_dropdown,
            image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    skip_btn.click(
        next_image,
        inputs=[
            sequence_dropdown,
            image_dropdown,
        ],
        outputs=[
            image_dropdown,
            image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    save_btn.click(
        save_and_next,
        inputs=[
            sequence_dropdown,
            image_dropdown,
            click_state,
        ],
        outputs=[
            image_dropdown,
            image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )


if __name__ == "__main__":

    print(
        "Sequences:",
        len(SEQUENCES),
    )

    print(
        "First sequence:",
        INITIAL_SEQ,
    )

    print(
        "First sequence images:",
        len(INITIAL_IMAGES),
    )

    print(
        "Initial time offset:",
        TIME_OFFSET_S,
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
