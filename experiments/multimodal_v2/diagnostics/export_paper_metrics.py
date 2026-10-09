#!/usr/bin/env python3
"""Recompute paper tables from enriched per_query.jsonl without model/GPU access."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from rdq_uav.multimodal_v2.paper_metrics import paper_report, write_paper_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--mode', choices=['B0', 'B1', 'B2', 'B3'], required=True)
    p.add_argument('--split', choices=['validation_sub', 'heldout_test_sub'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--conditions', type=Path)
    p.add_argument('--skip-coco', action='store_true')
    p.add_argument('--method')
    p.add_argument('--bandwidth', default='')
    args = p.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    required = {'sample_id', 'sequence_id', 'has_gt3d', 'gt_xyz', 'pred_xyz', 'has_gt2d'}
    if any(required-set(row) for row in rows):
        raise ValueError('Old per_query.jsonl lacks XYZ/2D records. Re-run evaluate.py with the upgraded code.')
    conditions = json.loads(args.conditions.read_text()) if args.conditions else None
    report = paper_report(rows, args.mode, conditions, args.output/'coco', not args.skip_coco)
    report.update(split=args.split, source_per_query=str(args.input.resolve()),
                  conditions_file=str(args.conditions.resolve()) if args.conditions else None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'paper_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    write_paper_csv(report, args.output/'paper_metrics.csv', method=args.method or args.mode,
                    modality={'B0':'LiDAR','B1':'RGB','B2':'LiDAR+RGB','B3':'LiDAR+RGB'}[args.mode],
                    bandwidth=args.bandwidth)
    print(args.output/'paper_metrics.csv')


if __name__ == '__main__': main()
