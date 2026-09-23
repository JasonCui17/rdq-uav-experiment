#!/usr/bin/env python
"""Batch 2D drone detection on MMAUD official train/val left fisheye images.

For each seq under official/{train,val}, runs the trained YOLO model on the
left half (1280x960) of each 2560x960 Image/*.png and writes YOLO-format
txt (class cx cy w h, normalized, save_conf) to seq/2d_detect/<name>.txt.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

DATASET_ROOT = Path("/home/jasoncui/datasets/MMAUD/official")
WEIGHTS = Path("/home/jasoncui/projects/rdq-uav-experiment/mmaud_drone_4090/weights/best.pt")

LEFT_W, LEFT_H = 1280, 960
BATCH = 8
LOAD_WORKERS = 8


def load_left(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.width != 2 * LEFT_W:
            raise ValueError(f"unexpected width {im.width} for {path}")
        return np.asarray(im.crop((0, 0, LEFT_W, LEFT_H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--seqs", nargs="*", default=None, help="optional seq names, applies to every split")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save-conf", action="store_true", help="append confidence as 6th column")
    args = ap.parse_args()

    model = YOLO(str(WEIGHTS))
    total_img = 0
    total_det = 0
    t0 = time.time()

    for split in args.splits:
        split_dir = DATASET_ROOT / split
        seq_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if args.seqs:
            seq_dirs = [p for p in seq_dirs if p.name in set(args.seqs)]
        for seq_dir in seq_dirs:
            img_dir = seq_dir / "Image"
            out_dir = seq_dir / "2d_detect"
            out_dir.mkdir(exist_ok=True)
            images = sorted(img_dir.glob("*.png"))
            if not images:
                print(f"[skip] {seq_dir}: no images", flush=True)
                continue

            n_img = n_det = 0
            with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as pool:
                for i in range(0, len(images), args.batch):
                    chunk = images[i : i + args.batch]
                    batch_np = list(pool.map(load_left, chunk))
                    results = model.predict(
                        batch_np,
                        imgsz=1280,
                        conf=args.conf,
                        half=True,
                        verbose=False,
                        device=0,
                    )
                    for path, r in zip(chunk, results):
                        lines = []
                        if r.boxes is not None:
                            cls = r.boxes.cls.cpu().numpy()
                            xywhn = r.boxes.xywhn.cpu().numpy()
                            conf = r.boxes.conf.cpu().numpy()
                            for c, b, cf in zip(cls, xywhn, conf):
                                row = f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                                if args.save_conf:
                                    row += f" {cf:.6f}"
                                lines.append(row)
                                n_det += 1
                        (out_dir / (path.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
                        n_img += 1
            total_img += n_img
            total_det += n_det
            rate = n_img / (time.time() - t0)
            print(f"[done] {seq_dir}: {n_img} imgs, {n_det} dets | global {total_img} imgs, {rate:.1f} img/s", flush=True)

    dt = time.time() - t0
    print(f"ALL DONE: {total_img} imgs, {total_det} dets, {dt:.0f}s ({total_img/dt:.1f} img/s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
