#!/usr/bin/env python3
"""Read-only browser viewer for one actual Multimodal V2 Dataset Sample."""

from __future__ import annotations

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
from PIL import Image
import yaml

from rdq_uav.multimodal_v2.data import V2_SPLIT_COUNTS, build_split_dataset
from rdq_uav.runtime_paths import apply_runtime_path_overrides, resolve_project_path

PAGE = Path(__file__).with_name("sample_viewer.html")
SENSORS = ((0, "Avia"), (1, "Mid360"))


class Viewer:
    def __init__(self, config: dict, max_points: int):
        self.max_points = max_points
        self.sequences = {}
        for split in V2_SPLIT_COUNTS:
            dataset = build_split_dataset(config, ROOT, split, filter_empty=False)
            for index, record in enumerate(dataset.query_records):
                self.sequences.setdefault(record.sequence_id, {"split": split, "dataset": dataset,
                                                               "indices": []})["indices"].append(index)

    def listing(self):
        return [{"sequence": name, "split": entry["split"], "count": len(entry["indices"])}
                for name, entry in sorted(self.sequences.items())]

    def selected(self, sequence: str, index: int):
        entry = self.sequences.get(sequence)
        if entry is None:
            raise ValueError("unknown sequence")
        if index < 0 or index >= len(entry["indices"]):
            raise ValueError("sample index out of range")
        return entry, entry["indices"][index]

    @lru_cache(maxsize=3)
    def sample(self, sequence: str, index: int):
        entry, absolute_index = self.selected(sequence, index)
        return entry["dataset"][absolute_index]

    def image(self, sequence: str, index: int) -> bytes:
        sample = self.sample(sequence, index)
        if not sample["m_V"]:
            raise ValueError("sample has no selected image")
        array = sample["image_uint8"].permute(1, 2, 0).numpy()
        output = io.BytesIO()
        Image.fromarray(array).save(output, format="JPEG", quality=88)
        return output.getvalue()

    def description(self, sequence: str, index: int):
        entry, _ = self.selected(sequence, index)
        dataset = entry["dataset"]
        sample = self.sample(sequence, index)
        query_time = float(sample["query_time"])
        points = sample["points"].numpy() if sample["m_R"] else np.empty((0, 3), np.float32)
        sensors = sample["sensor_id"].numpy() if sample["m_R"] else np.empty(0, np.int64)
        age = sample["delta_t"].numpy() if sample["m_R"] else np.empty(0, np.float32)
        stride = max(1, int(np.ceil(len(points) / self.max_points)))
        selected = np.arange(0, len(points), stride)
        clouds = {}
        for sensor_id, name in SENSORS:
            ids = selected[sensors[selected] == sensor_id]
            # Display-only decimation. The Sample, counts and supervision retain all input points.
            clouds[name] = {
                "total_points": int(np.count_nonzero(sensors == sensor_id)),
                "display_points": len(ids),
                "points": np.round(np.column_stack((points[ids], age[ids])).astype(np.float64), 4).tolist(),
            }
        events = dataset.select_radar_events(sequence, query_time)
        label_path = (dataset.root / sequence / dataset.label_directory /
                      f"{Path(sample['left_image_path']).stem}.txt") if sample["left_image_path"] else None
        label_text = label_path.read_text(encoding="utf-8-sig").strip() if label_path and label_path.exists() else None
        image = {"valid": bool(sample["m_V"]), "time": sample["image_time"],
                 "delta_t": sample["vision_delta_t"],
                 "file": Path(sample["left_image_path"]).name if sample["left_image_path"] else None,
                 "url": f"/api/image?sequence={sequence}&index={index}" if sample["m_V"] else None,
                 "view_wh": sample["image_view_wh"].tolist() if sample["m_V"] else None,
                 "source_wh": sample["image_source_wh"].tolist() if sample["m_V"] else None,
                 "bbox_source_xyxy": sample["gt_box_xyxy_px"].tolist() if sample["gt_2d_valid"] else None,
                 "bbox_view_xyxy": (
                     (sample["gt_box_xyxy_px"] * sample["image_scale_xy"].repeat(2)).tolist()
                     if sample["gt_2d_valid"] else None),
                 "label_file": (f"{Path(sample['left_image_path']).stem}.txt"
                                if sample["left_image_path"] else None),
                 "label_text": label_text}
        return {
            "sequence": sequence, "split": entry["split"], "index": index,
            "total_samples": len(entry["indices"]), "sample_id": sample["sample_id"],
            "query_time": query_time,
            "gt": {"valid": bool(sample["target_valid"]),
                   "xyz_m": sample["target_xyz"].tolist() if sample["target_valid"] else None,
                   "file": Path(sample["target_source_path"]).name if sample["target_source_path"] else None},
            "radar": {"valid": bool(sample["m_R"]), "history_s": sample["radar_history_s"],
                      "total_points": len(points), "display_points": len(selected),
                      "clouds": clouds,
                      "events": [{"sensor": event.sensor_name, "file": event.file_path.name,
                                  "time": event.timestamp, "delta_t": event.timestamp-query_time}
                                 for event in events]},
            "image": image,
        }


def handler_type(viewer: Viewer):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, content: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self.respond(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                    return
                if url.path == "/api/sequences":
                    payload = {"sequences": viewer.listing(), "max_points": viewer.max_points}
                elif url.path in ("/api/sample", "/api/image"):
                    query = parse_qs(url.query)
                    sequence = query.get("sequence", [""])[0]
                    index = int(query.get("index", ["0"])[0])
                    if url.path == "/api/image":
                        self.respond(200, viewer.image(sequence, index), "image/jpeg")
                        return
                    payload = viewer.description(sequence, index)
                else:
                    self.respond(404, b"Not found", "text/plain; charset=utf-8")
                    return
                self.respond(200, json.dumps(payload, allow_nan=False).encode(), "application/json; charset=utf-8")
            except (ValueError, OSError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/multimodal_v2/configs/b0_standalone.yaml"))
    parser.add_argument("--data-root", type=Path, help="override config data.root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-points", type=int, default=80000,
                        help="maximum rendered points per sample; Sample statistics use all points")
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    config = apply_runtime_path_overrides(yaml.safe_load(resolve_project_path(args.config, ROOT).read_text()))
    if args.data_root is not None:
        config["data"]["root"] = str(resolve_project_path(args.data_root, ROOT))
    viewer = Viewer(config, args.max_points)
    with ThreadingHTTPServer((args.host, args.port), handler_type(viewer)) as server:
        print(f"V2 sample viewer: http://{args.host}:{args.port} ({len(viewer.sequences)} sequences)", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
