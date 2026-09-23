#!/usr/bin/env python3

import json
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw


SEQ = "seq0001"
SEQ_DIR = Path("/home/jasoncui/datasets/MMAUD/official/train") / SEQ
IMAGE_DIR = SEQ_DIR / "Image"

OUT_DIR = Path("outputs/multimodal_v1/manual_calibration") / SEQ
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "bbox_annotations.json"

LEFT_W = 1280
LEFT_H = 960

IMAGES = sorted(IMAGE_DIR.glob("*.png"))
if not IMAGES:
    raise RuntimeError(f"No images found: {IMAGE_DIR}")


def load_annotations():
    if not OUT_FILE.exists():
        return {}
    records = json.loads(OUT_FILE.read_text())
    return {r["image_name"]: r for r in records}


ANNOTATIONS = load_annotations()


def save_annotations():
    records = sorted(
        ANNOTATIONS.values(),
        key=lambda x: x["image_time"],
    )
    tmp = OUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(OUT_FILE)


def image_time(path: Path):
    return float(path.stem)


def get_bbox(clicks):
    (xa, ya), (xb, yb) = clicks
    x1, x2 = sorted([float(xa), float(xb)])
    y1, y2 = sorted([float(ya), float(yb)])

    x1 = max(0.0, min(LEFT_W - 1.0, x1))
    x2 = max(0.0, min(LEFT_W - 1.0, x2))
    y1 = max(0.0, min(LEFT_H - 1.0, y1))
    y2 = max(0.0, min(LEFT_H - 1.0, y2))

    return x1, y1, x2, y2


def render(idx, clicks=None):
    idx = int(max(0, min(len(IMAGES) - 1, idx)))
    p = IMAGES[idx]

    with Image.open(p) as raw:
        im = raw.convert("RGB").crop((0, 0, LEFT_W, LEFT_H))

    draw = ImageDraw.Draw(im)

    # 已保存 bbox：绿色
    old = ANNOTATIONS.get(p.name)
    if old is not None:
        b = old["bbox_xyxy"]
        draw.rectangle(tuple(b), outline="lime", width=3)
        draw.text(
            (b[0], max(0, b[1] - 18)),
            "SAVED",
            fill="lime",
        )

    clicks = clicks or []

    # 当前点击：黄色
    for x, y in clicks:
        r = 5
        draw.line((x-r, y, x+r, y), fill="yellow", width=2)
        draw.line((x, y-r, x, y+r), fill="yellow", width=2)

    # 两点形成待保存 bbox：红色
    if len(clicks) == 2:
        x1, y1, x2, y2 = get_bbox(clicks)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

    return im


def frame_status(idx, extra=""):
    p = IMAGES[idx]
    saved = p.name in ANNOTATIONS

    text = (
        f"Frame {idx + 1}/{len(IMAGES)} | "
        f"time={p.stem} | "
        f"saved={'YES' if saved else 'NO'}"
    )

    if extra:
        text += f"\n{extra}"

    return text


def on_select(idx, clicks, evt: gr.SelectData):
    idx = int(idx)

    if evt.index is None:
        return render(idx, clicks), clicks, frame_status(idx)

    x, y = evt.index
    x = float(x)
    y = float(y)

    clicks = list(clicks or [])

    # 已经有两个点时，再点击就重新开始一个框
    if len(clicks) >= 2:
        clicks = []

    clicks.append([x, y])

    if len(clicks) == 1:
        msg = (
            f"第1点 = ({x:.1f}, {y:.1f})；"
            f"现在点击 bbox 对角的第2点"
        )
    else:
        x1, y1, x2, y2 = get_bbox(clicks)
        msg = (
            f"bbox = ({x1:.1f}, {y1:.1f}) "
            f"→ ({x2:.1f}, {y2:.1f})；"
            f"确认后点『保存并下一帧』"
        )

    return render(idx, clicks), clicks, frame_status(idx, msg)


def save_and_next(idx, clicks):
    idx = int(idx)
    clicks = list(clicks or [])

    if len(clicks) != 2:
        return (
            render(idx, clicks),
            idx,
            clicks,
            frame_status(idx, "需要先点击 bbox 的两个对角点"),
        )

    p = IMAGES[idx]
    x1, y1, x2, y2 = get_bbox(clicks)

    if x2 - x1 < 1 or y2 - y1 < 1:
        return (
            render(idx, clicks),
            idx,
            clicks,
            frame_status(idx, "bbox 太小，请重新标"),
        )

    record = {
        "sequence": SEQ,
        "image_name": p.name,
        "image_path": str(p),
        "image_time": image_time(p),

        "bbox_xyxy": [x1, y1, x2, y2],

        "center_u": (x1 + x2) / 2.0,
        "center_v": (y1 + y2) / 2.0,
        "width_px": x2 - x1,
        "height_px": y2 - y1,

        "image_width": LEFT_W,
        "image_height": LEFT_H,
    }

    ANNOTATIONS[p.name] = record
    save_annotations()

    next_idx = min(idx + 1, len(IMAGES) - 1)

    return (
        render(next_idx, []),
        next_idx,
        [],
        frame_status(
            next_idx,
            f"已保存，共 {len(ANNOTATIONS)} 个标定框",
        ),
    )


def skip(idx):
    idx = int(idx)
    next_idx = min(idx + 1, len(IMAGES) - 1)

    return (
        render(next_idx, []),
        next_idx,
        [],
        frame_status(next_idx, "已跳过上一帧"),
    )


def prev(idx):
    idx = int(idx)
    prev_idx = max(idx - 1, 0)

    return (
        render(prev_idx, []),
        prev_idx,
        [],
        frame_status(prev_idx),
    )


# 如果之前标过，启动时从第一个未标注帧继续
START_IDX = 0
for i, p in enumerate(IMAGES):
    if p.name not in ANNOTATIONS:
        START_IDX = i
        break


with gr.Blocks(title="MMAUD Manual Calibration") as demo:
    gr.Markdown(
        """
### MMAUD 左相机交互标定

操作：**点击无人机 bbox 的两个对角点**。

- 黄色十字：点击位置
- 红框：当前待保存 bbox
- 绿框：已经保存的 bbox
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
        lines=2,
    )

    with gr.Row():
        prev_btn = gr.Button("上一帧")
        skip_btn = gr.Button("跳过")
        save_btn = gr.Button("保存并下一帧", variant="primary")

    image.select(
        on_select,
        inputs=[idx_state, click_state],
        outputs=[image, click_state, status],
        show_progress="hidden",
    )

    save_btn.click(
        save_and_next,
        inputs=[idx_state, click_state],
        outputs=[image, idx_state, click_state, status],
        show_progress="hidden",
    )

    skip_btn.click(
        skip,
        inputs=[idx_state],
        outputs=[image, idx_state, click_state, status],
        show_progress="hidden",
    )

    prev_btn.click(
        prev,
        inputs=[idx_state],
        outputs=[image, idx_state, click_state, status],
        show_progress="hidden",
    )


if __name__ == "__main__":
    print(f"Sequence: {SEQ}")
    print(f"Frames: {len(IMAGES)}")
    print(f"Annotations: {OUT_FILE.resolve()}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
