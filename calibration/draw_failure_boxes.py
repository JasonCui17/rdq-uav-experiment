#!/usr/bin/env python3
"""根据每个子文件夹的 official_2d_label.txt (YOLO 格式) 在 official_2d_image.png 上画框，
结果保存为同子文件夹下的 official_2d_drawn.png。
如果 case.json 里有投影点 (proj_nearest_u/v, proj_interp_u/v)，一并画出来方便对比。
"""

import json
from pathlib import Path

import cv2

ROOT = Path("/home/jasoncui/datasets/MMAUD/rdq_uav_experiment/calibration/mapping_failure_cases")


def draw_case(folder: Path) -> bool:
    label_file = folder / "official_2d_label.txt"
    image_file = folder / "official_2d_image.png"
    if not label_file.exists() or not image_file.exists():
        print(f"[skip] {folder.name}: 缺少 label 或 image")
        return False

    img = cv2.imread(str(image_file))
    if img is None:
        print(f"[skip] {folder.name}: 无法读取图像")
        return False
    h, w = img.shape[:2]

    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts[0], *map(float, parts[1:])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"cls{cls}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # 投影点对比（存在才画）
    case_file = folder / "case.json"
    if case_file.exists():
        case = json.loads(case_file.read_text())
        for key, color in [("proj_nearest", (0, 255, 0)), ("proj_interp", (255, 0, 0))]:
            u, v = case.get(f"{key}_u"), case.get(f"{key}_v")
            if u is None or v is None:
                continue
            center = (int(round(u)), int(round(v)))
            cv2.circle(img, center, 5, color, -1)
            cv2.putText(img, key.replace("proj_", ""), (center[0] + 8, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    out = folder / "official_2d_drawn.png"
    cv2.imwrite(str(out), img)
    return True


def main():
    folders = sorted(p for p in ROOT.iterdir() if p.is_dir())
    ok = 0
    for folder in folders:
        if draw_case(folder):
            ok += 1
    print(f"完成: {ok}/{len(folders)}")


if __name__ == "__main__":
    main()
