#!/usr/bin/env python3

import json
from pathlib import Path

import gradio as gr
import numpy as np
import yaml
from PIL import Image, ImageDraw

from rdq_uav.calibration.omni import OmniRadtanCamera, transform_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQ = "seq0001"
SEQ_DIR = Path("/home/jasoncui/datasets/MMAUD/official/train") / SEQ
IMAGE_DIR = SEQ_DIR / "Image"
GT_DIR = SEQ_DIR / "ground_truth"

CAMERA_CONFIG = PROJECT_ROOT / "configs/calibration/mmaud_v1_omni.yaml"
CAMERA_CALIB = PROJECT_ROOT / "calibration/official_left_fitted_calibration.json"

OUT_DIR = PROJECT_ROOT / "outputs/multimodal_v1/manual_calibration" / SEQ
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "bbox_annotations.json"

LEFT_W = 1280
LEFT_H = 960

# 仅用于显示的建议框大小，不参与最终标定
SUGGEST_HALF_W = 20
SUGGEST_HALF_H = 20


# ----------------------------------------------------------------------
# Load image / GT
# ----------------------------------------------------------------------

IMAGES = sorted(IMAGE_DIR.glob("*.png"), key=lambda p: float(p.stem))
GT_PATHS = sorted(GT_DIR.glob("*.npy"), key=lambda p: float(p.stem))

if not IMAGES:
    raise RuntimeError(f"No images: {IMAGE_DIR}")

if not GT_PATHS:
    raise RuntimeError(f"No GT: {GT_DIR}")

GT_TIMES = np.asarray([float(p.stem) for p in GT_PATHS], dtype=np.float64)
GT_XYZ = np.stack(
    [
        np.asarray(np.load(p, allow_pickle=False), dtype=np.float64).reshape(3)
        for p in GT_PATHS
    ],
    axis=0,
)


# ----------------------------------------------------------------------
# Load current coarse calibration
# ----------------------------------------------------------------------

camera_cfg = yaml.safe_load(CAMERA_CONFIG.read_text())
calib = json.loads(CAMERA_CALIB.read_text())

CAMERA = OmniRadtanCamera.from_config(camera_cfg["cameras"]["left"])

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


# ----------------------------------------------------------------------
# Annotation storage
# ----------------------------------------------------------------------

def load_annotations():
    if not OUT_FILE.exists():
        return {}

    records = json.loads(OUT_FILE.read_text())

    return {
        r["image_name"]: r
        for r in records
    }


ANNOTATIONS = load_annotations()


def save_annotations():
    records = sorted(
        ANNOTATIONS.values(),
        key=lambda x: x["image_time"],
    )

    tmp = OUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(OUT_FILE)


# ----------------------------------------------------------------------
# GT interpolation
# ----------------------------------------------------------------------

def interpolate_gt(image_time):
    """
    Current coarse convention:

        gt_query_time = image_time + TIME_OFFSET_S

    TIME_OFFSET_S is only an INITIAL value for visualization.
    It will later be optimized.
    """

    query_time = float(image_time) + TIME_OFFSET_S

    i = int(np.searchsorted(GT_TIMES, query_time))

    if i <= 0:
        idx = 0
        return GT_XYZ[idx], GT_TIMES[idx], query_time, "nearest"

    if i >= len(GT_TIMES):
        idx = len(GT_TIMES) - 1
        return GT_XYZ[idx], GT_TIMES[idx], query_time, "nearest"

    t0 = GT_TIMES[i - 1]
    t1 = GT_TIMES[i]

    p0 = GT_XYZ[i - 1]
    p1 = GT_XYZ[i]

    if t1 <= t0:
        return p0, t0, query_time, "nearest"

    alpha = (query_time - t0) / (t1 - t0)

    xyz = (1.0 - alpha) * p0 + alpha * p1

    return xyz, query_time, query_time, "interpolated"


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------

def project_gt(image_time):
    xyz, gt_time, query_time, mode = interpolate_gt(image_time)

    point = xyz.reshape(1, 3)

    point_camera = transform_points(
        point,
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
        "gt_time": float(gt_time),
        "query_time": float(query_time),
        "mode": mode,
    }


def suggestion_bbox(projection):
    if projection is None:
        return None

    u = projection["u"]
    v = projection["v"]

    x1 = max(0.0, u - SUGGEST_HALF_W)
    y1 = max(0.0, v - SUGGEST_HALF_H)
    x2 = min(LEFT_W - 1.0, u + SUGGEST_HALF_W)
    y2 = min(LEFT_H - 1.0, v + SUGGEST_HALF_H)

    return [x1, y1, x2, y2]


# ----------------------------------------------------------------------
# BBox helpers
# ----------------------------------------------------------------------

def get_bbox(clicks):
    (xa, ya), (xb, yb) = clicks

    x1, x2 = sorted([float(xa), float(xb)])
    y1, y2 = sorted([float(ya), float(yb)])

    x1 = max(0.0, min(LEFT_W - 1.0, x1))
    x2 = max(0.0, min(LEFT_W - 1.0, x2))
    y1 = max(0.0, min(LEFT_H - 1.0, y1))
    y2 = max(0.0, min(LEFT_H - 1.0, y2))

    return x1, y1, x2, y2


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def render(idx, clicks=None):
    idx = int(max(0, min(len(IMAGES) - 1, idx)))

    p = IMAGES[idx]
    t_img = float(p.stem)

    with Image.open(p) as raw:
        im = raw.convert("RGB").crop((0, 0, LEFT_W, LEFT_H))

    draw = ImageDraw.Draw(im)

    # --------------------------------------------------------------
    # Current coarse GT projection
    # cyan cross + orange suggestion bbox
    # --------------------------------------------------------------

    projection = project_gt(t_img)

    if projection is not None:
        u = projection["u"]
        v = projection["v"]

        r = 10

        draw.line(
            (u - r, v, u + r, v),
            fill="cyan",
            width=3,
        )

        draw.line(
            (u, v - r, u, v + r),
            fill="cyan",
            width=3,
        )

        sb = suggestion_bbox(projection)

        draw.rectangle(
            tuple(sb),
            outline="orange",
            width=3,
        )

        draw.text(
            (sb[0], max(0, sb[1] - 18)),
            "COARSE GT",
            fill="orange",
        )

    # --------------------------------------------------------------
    # Saved manual bbox
    # --------------------------------------------------------------

    old = ANNOTATIONS.get(p.name)

    if old is not None:
        b = old["bbox_xyxy"]

        draw.rectangle(
            tuple(b),
            outline="lime",
            width=3,
        )

        draw.text(
            (b[0], max(0, b[1] - 18)),
            "MANUAL SAVED",
            fill="lime",
        )

    # --------------------------------------------------------------
    # Current manual clicks
    # --------------------------------------------------------------

    clicks = clicks or []

    for x, y in clicks:
        r = 6

        draw.line(
            (x - r, y, x + r, y),
            fill="yellow",
            width=2,
        )

        draw.line(
            (x, y - r, x, y + r),
            fill="yellow",
            width=2,
        )

    if len(clicks) == 2:
        x1, y1, x2, y2 = get_bbox(clicks)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=3,
        )

        draw.text(
            (x1, max(0, y1 - 18)),
            "MANUAL NEW",
            fill="red",
        )

    return im


def frame_status(idx, extra=""):
    idx = int(idx)

    p = IMAGES[idx]
    image_time = float(p.stem)

    projection = project_gt(image_time)

    lines = [
        f"Frame {idx + 1}/{len(IMAGES)}",
        f"image_time = {image_time:.6f}",
        f"initial time_offset = {TIME_OFFSET_S:+.6f} s",
    ]

    if projection is None:
        lines.append("coarse GT projection = INVALID")
    else:
        lines.append(
            "coarse GT projection = "
            f"({projection['u']:.1f}, {projection['v']:.1f})"
        )

        lines.append(
            "GT XYZ = "
            f"({projection['xyz'][0]:.3f}, "
            f"{projection['xyz'][1]:.3f}, "
            f"{projection['xyz'][2]:.3f})"
        )

    lines.append(
        f"manual saved = {'YES' if p.name in ANNOTATIONS else 'NO'}"
    )

    if extra:
        lines.append(extra)

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Interaction
# ----------------------------------------------------------------------

def on_select(idx, clicks, evt: gr.SelectData):
    idx = int(idx)

    if evt.index is None:
        return (
            render(idx, clicks),
            clicks,
            frame_status(idx),
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
            f"第1点 = ({x:.1f}, {y:.1f})；"
            "点击无人机 bbox 对角的第2点"
        )
    else:
        x1, y1, x2, y2 = get_bbox(clicks)

        msg = (
            f"人工 bbox = "
            f"({x1:.1f}, {y1:.1f}) → "
            f"({x2:.1f}, {y2:.1f})"
        )

    return (
        render(idx, clicks),
        clicks,
        frame_status(idx, msg),
    )


def save_and_next(idx, clicks):
    idx = int(idx)
    clicks = list(clicks or [])

    if len(clicks) != 2:
        return (
            render(idx, clicks),
            idx,
            clicks,
            frame_status(
                idx,
                "必须人工点击两个 bbox 对角点后才能保存",
            ),
        )

    p = IMAGES[idx]

    x1, y1, x2, y2 = get_bbox(clicks)

    if x2 - x1 < 1 or y2 - y1 < 1:
        return (
            render(idx, clicks),
            idx,
            clicks,
            frame_status(idx, "bbox 太小"),
        )

    image_t = float(p.stem)
    projection = project_gt(image_t)

    record = {
        "sequence": SEQ,
        "image_name": p.name,
        "image_path": str(p),
        "image_time": image_t,

        "bbox_xyxy": [
            x1,
            y1,
            x2,
            y2,
        ],

        "center_u": (x1 + x2) / 2.0,
        "center_v": (y1 + y2) / 2.0,

        "width_px": x2 - x1,
        "height_px": y2 - y1,

        "image_width": LEFT_W,
        "image_height": LEFT_H,

        # 保存粗投影仅用于之后诊断
        "initial_projection": projection,

        # 明确：人工 bbox 才是 calibration anchor
        "annotation_source": "manual_bbox",
    }

    ANNOTATIONS[p.name] = record
    save_annotations()

    next_idx = min(
        idx + 1,
        len(IMAGES) - 1,
    )

    return (
        render(next_idx, []),
        next_idx,
        [],
        frame_status(
            next_idx,
            f"已保存人工框；累计 {len(ANNOTATIONS)} 帧",
        ),
    )


def skip(idx):
    idx = int(idx)

    next_idx = min(
        idx + 1,
        len(IMAGES) - 1,
    )

    return (
        render(next_idx, []),
        next_idx,
        [],
        frame_status(next_idx, "上一帧已跳过"),
    )


def prev(idx):
    idx = int(idx)

    prev_idx = max(
        idx - 1,
        0,
    )

    return (
        render(prev_idx, []),
        prev_idx,
        [],
        frame_status(prev_idx),
    )


START_IDX = 0


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

with gr.Blocks(title="MMAUD Interactive Calibration V2") as demo:

    gr.Markdown(
        """
### MMAUD 交互式相机标定 V2

**青色十字**：当前粗标定得到的 GT 投影中心  
**橙色框**：以粗投影为中心的建议搜索框  
**红色框**：你当前人工框出的无人机  
**绿色框**：已经保存的人工 bbox

操作只需要：

1. 看粗投影附近；
2. 点击真实无人机 bbox 两个对角点；
3. 保存并下一帧。
"""
    )

    idx_state = gr.State(START_IDX)
    click_state = gr.State([])

    image = gr.Image(
        value=render(START_IDX, []),
        type="pil",
        interactive=True,
        show_label=False,
        height=720,
    )

    status = gr.Textbox(
        value=frame_status(START_IDX),
        label="状态",
        interactive=False,
        lines=6,
    )

    with gr.Row():
        prev_btn = gr.Button("上一帧")
        skip_btn = gr.Button("跳过")
        save_btn = gr.Button(
            "保存人工框并下一帧",
            variant="primary",
        )

    image.select(
        on_select,
        inputs=[
            idx_state,
            click_state,
        ],
        outputs=[
            image,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    save_btn.click(
        save_and_next,
        inputs=[
            idx_state,
            click_state,
        ],
        outputs=[
            image,
            idx_state,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    skip_btn.click(
        skip,
        inputs=[idx_state],
        outputs=[
            image,
            idx_state,
            click_state,
            status,
        ],
        show_progress="hidden",
    )

    prev_btn.click(
        prev,
        inputs=[idx_state],
        outputs=[
            image,
            idx_state,
            click_state,
            status,
        ],
        show_progress="hidden",
    )


if __name__ == "__main__":

    print("Sequence:", SEQ)
    print("Images:", len(IMAGES))
    print("GT records:", len(GT_PATHS))

    print(
        "Initial time offset:",
        TIME_OFFSET_S,
    )

    print(
        "Annotations:",
        OUT_FILE.resolve(),
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
