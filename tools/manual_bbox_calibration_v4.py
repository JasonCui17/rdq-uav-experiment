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

# 粗投影提示框：原图像素
SUGGEST_HALF_W = 20
SUGGEST_HALF_H = 20

# 局部标定窗口：160x160 原始像素，放大 4 倍显示为 640x640
ZOOM_CROP_SIZE = 320
ZOOM_SCALE = 2


# ============================================================
# Calibration
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
# Dataset
# ============================================================

SEQUENCES = sorted(
    p.name
    for p in DATA_ROOT.iterdir()
    if p.is_dir()
    and (p / "Image").exists()
    and (p / "ground_truth").exists()
)

if not SEQUENCES:
    raise RuntimeError(f"No sequences found: {DATA_ROOT}")

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
            ]
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
# Annotation storage
# ============================================================

def annotation_file(seq):
    d = OUTPUT_ROOT / seq
    d.mkdir(parents=True, exist_ok=True)
    return d / "bbox_annotations.json"


def load_annotations(seq):
    if seq in ANNOTATION_CACHE:
        return ANNOTATION_CACHE[seq]

    p = annotation_file(seq)

    if p.exists():
        records = json.loads(p.read_text())
        annotations = {
            r["image_name"]: r
            for r in records
        }
    else:
        annotations = {}

    ANNOTATION_CACHE[seq] = annotations
    return annotations


def save_annotations(seq):
    records = sorted(
        load_annotations(seq).values(),
        key=lambda r: r["image_time"],
    )

    p = annotation_file(seq)
    tmp = p.with_suffix(".tmp")

    tmp.write_text(
        json.dumps(records, indent=2) + "\n"
    )
    tmp.replace(p)


# ============================================================
# GT -> coarse projection
# ============================================================

def interpolate_gt(seq, image_time):
    data = load_sequence(seq)

    times = data["gt_times"]
    xyz = data["gt_xyz"]

    if len(times) == 0:
        return None

    # 这里只是粗初始化，最终 Δt 后续重新拟合
    query_time = float(image_time) + TIME_OFFSET_S

    i = int(np.searchsorted(times, query_time))

    if i <= 0:
        return xyz[0], float(times[0]), query_time, "nearest"

    if i >= len(times):
        return xyz[-1], float(times[-1]), query_time, "nearest"

    t0 = times[i - 1]
    t1 = times[i]

    p0 = xyz[i - 1]
    p1 = xyz[i]

    alpha = (query_time - t0) / (t1 - t0)

    point = (1.0 - alpha) * p0 + alpha * p1

    return point, query_time, query_time, "interpolated"


def project_gt(seq, image_time):
    result = interpolate_gt(seq, image_time)

    if result is None:
        return None

    xyz, gt_time, query_time, mode = result

    camera_xyz = transform_points(
        xyz.reshape(1, 3),
        ROTATION,
        TRANSLATION,
    )

    pixels, valid = CAMERA.project(
        camera_xyz,
        require_in_image=True,
    )

    if not bool(valid[0]):
        return None

    u, v = map(float, pixels[0])

    return {
        "u": u,
        "v": v,
        "xyz": xyz.tolist(),
        "gt_time": float(gt_time),
        "query_time": float(query_time),
        "mode": mode,
    }


def coarse_bbox(projection):
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
# Zoom geometry
# ============================================================

def zoom_bounds(seq, image_name):
    image_time = float(Path(image_name).stem)
    projection = project_gt(seq, image_time)

    if projection is None:
        cx = LEFT_W / 2
        cy = LEFT_H / 2
    else:
        cx = projection["u"]
        cy = projection["v"]

    half = ZOOM_CROP_SIZE / 2

    x0 = int(round(cx - half))
    y0 = int(round(cy - half))

    x0 = max(0, min(LEFT_W - ZOOM_CROP_SIZE, x0))
    y0 = max(0, min(LEFT_H - ZOOM_CROP_SIZE, y0))

    x1 = x0 + ZOOM_CROP_SIZE
    y1 = y0 + ZOOM_CROP_SIZE

    return x0, y0, x1, y1


def original_to_zoom(x, y, bounds):
    x0, y0, _, _ = bounds

    return (
        (float(x) - x0) * ZOOM_SCALE,
        (float(y) - y0) * ZOOM_SCALE,
    )


def zoom_to_original(x, y, bounds):
    x0, y0, _, _ = bounds

    ox = x0 + float(x) / ZOOM_SCALE
    oy = y0 + float(y) / ZOOM_SCALE

    ox = max(0.0, min(LEFT_W - 1.0, ox))
    oy = max(0.0, min(LEFT_H - 1.0, oy))

    return ox, oy


def get_bbox(clicks):
    (xa, ya), (xb, yb) = clicks

    x1, x2 = sorted([float(xa), float(xb)])
    y1, y2 = sorted([float(ya), float(yb)])

    return (
        max(0.0, min(LEFT_W - 1.0, x1)),
        max(0.0, min(LEFT_H - 1.0, y1)),
        max(0.0, min(LEFT_W - 1.0, x2)),
        max(0.0, min(LEFT_H - 1.0, y2)),
    )


# ============================================================
# Rendering
# ============================================================

def load_left_image(seq, image_name):
    p = load_sequence(seq)["image_map"][image_name]

    with Image.open(p) as raw:
        return raw.convert("RGB").crop(
            (0, 0, LEFT_W, LEFT_H)
        )


def render_full(seq, image_name, clicks=None):
    if not image_name:
        return None

    im = load_left_image(seq, image_name)
    draw = ImageDraw.Draw(im)

    image_time = float(Path(image_name).stem)

    # 粗投影：橙色空心框
    projection = project_gt(seq, image_time)
    b = coarse_bbox(projection)

    if b is not None:
        draw.rectangle(
            tuple(b),
            outline="orange",
            width=2,
        )

    # 已保存人工框：绿色
    old = load_annotations(seq).get(image_name)

    if old is not None:
        draw.rectangle(
            tuple(old["bbox_xyxy"]),
            outline="lime",
            width=2,
        )

    # 当前人工框：红色
    clicks = clicks or []

    if len(clicks) == 2:
        draw.rectangle(
            get_bbox(clicks),
            outline="red",
            width=2,
        )

    return im


def render_zoom(seq, image_name, clicks=None):
    if not image_name:
        return None

    full = load_left_image(seq, image_name)
    bounds = zoom_bounds(seq, image_name)

    x0, y0, x1, y1 = bounds

    crop = full.crop(
        (x0, y0, x1, y1)
    )

    crop = crop.resize(
        (
            ZOOM_CROP_SIZE * ZOOM_SCALE,
            ZOOM_CROP_SIZE * ZOOM_SCALE,
        ),
        Image.Resampling.NEAREST,
    )

    draw = ImageDraw.Draw(crop)

    image_time = float(Path(image_name).stem)

    # 粗投影空心框
    projection = project_gt(seq, image_time)
    b = coarse_bbox(projection)

    if b is not None:
        zx1, zy1 = original_to_zoom(
            b[0], b[1], bounds
        )
        zx2, zy2 = original_to_zoom(
            b[2], b[3], bounds
        )

        draw.rectangle(
            (zx1, zy1, zx2, zy2),
            outline="orange",
            width=2,
        )

    # 已保存 bbox
    old = load_annotations(seq).get(image_name)

    if old is not None:
        b = old["bbox_xyxy"]

        zx1, zy1 = original_to_zoom(
            b[0], b[1], bounds
        )
        zx2, zy2 = original_to_zoom(
            b[2], b[3], bounds
        )

        draw.rectangle(
            (zx1, zy1, zx2, zy2),
            outline="lime",
            width=2,
        )

    # 当前人工标注
    clicks = clicks or []

    # 第一次点击：显示黄色点
    if len(clicks) >= 1:
        zx, zy = original_to_zoom(
            clicks[0][0],
            clicks[0][1],
            bounds,
        )
        r = 5
        draw.ellipse(
            (zx-r, zy-r, zx+r, zy+r),
            outline="yellow",
            width=3,
        )

    # 第二次点击：形成红色 bbox
    if len(clicks) == 2:
        b = get_bbox(clicks)

        zx1, zy1 = original_to_zoom(
            b[0], b[1], bounds
        )
        zx2, zy2 = original_to_zoom(
            b[2], b[3], bounds
        )

        draw.rectangle(
            (zx1, zy1, zx2, zy2),
            outline="red",
            width=3,
        )

    return crop


# ============================================================
# Status
# ============================================================

def frame_status(seq, image_name, extra=""):
    if not image_name:
        return "No image"

    data = load_sequence(seq)
    names = [p.name for p in data["images"]]

    idx = names.index(image_name)

    image_time = float(Path(image_name).stem)

    projection = project_gt(seq, image_time)
    bounds = zoom_bounds(seq, image_name)

    lines = [
        f"Sequence: {seq}",
        f"Frame: {idx + 1}/{len(names)}",
        f"image_time: {image_time:.6f}",
        f"initial Δt: {TIME_OFFSET_S:+.6f} s",
        f"zoom source area: x={bounds[0]}:{bounds[2]}, y={bounds[1]}:{bounds[3]}",
        f"zoom: {ZOOM_CROP_SIZE}x{ZOOM_CROP_SIZE} → "
        f"{ZOOM_CROP_SIZE * ZOOM_SCALE}x{ZOOM_CROP_SIZE * ZOOM_SCALE}",
    ]

    if projection is None:
        lines.append("coarse projection: INVALID")
    else:
        lines.append(
            f"coarse projection: "
            f"({projection['u']:.1f}, {projection['v']:.1f})"
        )

    annotations = load_annotations(seq)

    lines.append(
        f"saved: {'YES' if image_name in annotations else 'NO'} | "
        f"sequence annotations: {len(annotations)}"
    )

    if extra:
        lines.append(extra)

    return "\n".join(lines)


# ============================================================
# Callbacks
# ============================================================

def on_sequence_change(seq):
    names = [
        p.name
        for p in load_sequence(seq)["images"]
    ]

    if not names:
        return (
            gr.Dropdown(choices=[], value=None),
            None,
            None,
            [],
            f"{seq}: no images",
        )

    image_name = names[0]

    return (
        gr.Dropdown(
            choices=names,
            value=image_name,
        ),
        render_full(seq, image_name, []),
        render_zoom(seq, image_name, []),
        [],
        frame_status(seq, image_name),
    )


def on_image_change(seq, image_name):
    if not image_name:
        return None, None, [], "No image"

    return (
        render_full(seq, image_name, []),
        render_zoom(seq, image_name, []),
        [],
        frame_status(seq, image_name),
    )


def on_zoom_select(
    seq,
    image_name,
    clicks,
    evt: gr.SelectData,
):
    if not image_name or evt.index is None:
        return (
            render_full(seq, image_name, clicks),
            render_zoom(seq, image_name, clicks),
            clicks,
            frame_status(seq, image_name),
        )

    zx, zy = evt.index

    bounds = zoom_bounds(
        seq,
        image_name,
    )

    ox, oy = zoom_to_original(
        zx,
        zy,
        bounds,
    )

    clicks = list(clicks or [])

    if len(clicks) >= 2:
        clicks = []

    clicks.append([ox, oy])

    if len(clicks) == 1:
        msg = (
            f"第1点原图坐标=({ox:.1f},{oy:.1f})；"
            "再点 bbox 对角点"
        )
    else:
        x1, y1, x2, y2 = get_bbox(clicks)

        msg = (
            f"bbox 原图坐标="
            f"({x1:.1f},{y1:.1f})→({x2:.1f},{y2:.1f}), "
            f"size={x2-x1:.1f}×{y2-y1:.1f}px"
        )

    return (
        render_full(seq, image_name, clicks),
        render_zoom(seq, image_name, clicks),
        clicks,
        frame_status(seq, image_name, msg),
    )


def move_image(seq, image_name, offset):
    names = [
        p.name
        for p in load_sequence(seq)["images"]
    ]

    if not names:
        return None

    if image_name not in names:
        return names[0]

    idx = names.index(image_name)

    idx = max(
        0,
        min(len(names) - 1, idx + offset),
    )

    return names[idx]


def previous_image(seq, image_name):
    target = move_image(seq, image_name, -1)

    return (
        target,
        render_full(seq, target, []),
        render_zoom(seq, target, []),
        [],
        frame_status(seq, target),
    )


def next_image(seq, image_name):
    target = move_image(seq, image_name, +1)

    return (
        target,
        render_full(seq, target, []),
        render_zoom(seq, target, []),
        [],
        frame_status(seq, target),
    )


def save_and_next(seq, image_name, clicks):
    clicks = list(clicks or [])

    if len(clicks) != 2:
        return (
            image_name,
            render_full(seq, image_name, clicks),
            render_zoom(seq, image_name, clicks),
            clicks,
            frame_status(
                seq,
                image_name,
                "需要先在右侧放大图中点击 bbox 两个对角点",
            ),
        )

    x1, y1, x2, y2 = get_bbox(clicks)

    if x2 - x1 < 1 or y2 - y1 < 1:
        return (
            image_name,
            render_full(seq, image_name, clicks),
            render_zoom(seq, image_name, clicks),
            clicks,
            frame_status(seq, image_name, "bbox 太小"),
        )

    p = load_sequence(seq)["image_map"][image_name]
    image_time = float(p.stem)

    record = {
        "sequence": seq,
        "image_name": image_name,
        "image_path": str(p),
        "image_time": image_time,

        "bbox_xyxy": [
            x1, y1, x2, y2
        ],

        "center_u": (x1 + x2) / 2.0,
        "center_v": (y1 + y2) / 2.0,

        "width_px": x2 - x1,
        "height_px": y2 - y1,

        "image_width": LEFT_W,
        "image_height": LEFT_H,

        "initial_projection": project_gt(
            seq,
            image_time,
        ),

        "annotation_source":
            "manual_bbox_zoom_v4",
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
        render_full(seq, target, []),
        render_zoom(seq, target, []),
        [],
        frame_status(
            seq,
            target,
            f"已保存；{seq} 累计 {len(annotations)} 个框",
        ),
    )


# ============================================================
# Initial
# ============================================================

INITIAL_SEQ = SEQUENCES[0]

INITIAL_IMAGES = [
    p.name
    for p in load_sequence(INITIAL_SEQ)["images"]
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
    title="MMAUD Interactive Calibration V4",
    css="""
    #zoom-calibration-image img {
        -webkit-user-drag: none !important;
        user-select: none !important;
    }
    """,
) as demo:

    gr.Markdown(
        """
### MMAUD 小目标交互标定 V4

**左侧：** 1280×960 全图位置参考  
**右侧：** 粗投影附近 160×160 原始像素区域，4× 放大显示  

请只在 **右侧放大图** 中点击无人机 bbox 的两个对角点。

- 橙框：粗投影区域
- 红框：当前人工框
- 绿框：已保存人工框
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

    with gr.Row():

        full_image = gr.Image(
            value=render_full(
                INITIAL_SEQ,
                INITIAL_IMAGE,
                [],
            ),
            type="pil",
            interactive=False,
            label="全图参考",
            height=600,
        )

        zoom_image = gr.Image(
            value=render_zoom(
                INITIAL_SEQ,
                INITIAL_IMAGE,
                [],
            ),
            type="pil",
            interactive=True,
            label="4× 局部放大标定窗口（单击两个对角点，不要拖动）",
            height=640,
            elem_id="zoom-calibration-image",
        )

    status = gr.Textbox(
        value=frame_status(
            INITIAL_SEQ,
            INITIAL_IMAGE,
        ),
        label="状态",
        interactive=False,
        lines=9,
    )

    with gr.Row():

        prev_btn = gr.Button("上一张")

        next_btn = gr.Button("下一张")

        skip_btn = gr.Button("跳过")

        save_btn = gr.Button(
            "保存人工框并下一张",
            variant="primary",
        )

    sequence_dropdown.change(
        on_sequence_change,
        inputs=[sequence_dropdown],
        outputs=[
            image_dropdown,
            full_image,
            zoom_image,
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
            full_image,
            zoom_image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    zoom_image.select(
        on_zoom_select,
        inputs=[
            sequence_dropdown,
            image_dropdown,
            click_state,
        ],
        outputs=[
            full_image,
            zoom_image,
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
            full_image,
            zoom_image,
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
            full_image,
            zoom_image,
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
            full_image,
            zoom_image,
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
            full_image,
            zoom_image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )


if __name__ == "__main__":

    print("Sequences:", len(SEQUENCES))
    print("Initial sequence:", INITIAL_SEQ)
    print("Initial images:", len(INITIAL_IMAGES))
    print("Calibration:", CAMERA_CALIB)
    print("Rotation:")
    print(ROTATION)
    print("Zoom crop:", ZOOM_CROP_SIZE)
    print("Zoom scale:", ZOOM_SCALE)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
