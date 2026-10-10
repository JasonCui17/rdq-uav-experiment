# V2 offline audit
Run from repository root. Requires existing paper_metrics.py, numpy, Pillow, pycocotools.
No model, loss or selector changes. Do not retrain.

Visual commands:
```bash
for kind in best_validation last_validation best_heldout; do
  PYTHONPATH=src python modality_audit_20261010/audit_modalities.py --mode B1 \
    --input "outputs/b1_coco_cdn_bs4_seed42/eval_${kind}_paper/per_query.jsonl" \
    --output "outputs/vision_audit_20261010/${kind}" \
    --data-root /root/autodl-tmp/MMAUD/official/train --visualize-per-category 4
done
```
Per-sequence AP, deduplicated top1/oracle IoU50, box size, top1 confidence, oracle confidence,
center error in source pixels, width/height predicted-to-GT ratios. Categories: success;
ranking_failure (oracle IoU>=0.5, top1<0.5); candidate_miss (all final boxes IoU<0.5);
no_output. Oracle is only a diagnostic, not a deployable prediction or metric used for selection.
Confidence is not assumed calibrated. Per-image CSV permits success/failure subgroup analysis.
Visualization: green GT, red top1, cyan final-candidate oracle; full source left view plus enlarged
GT/top1 region. Fixed deterministic worst-IoU examples per category/sequence, not random sampling.
Visualization read failures are listed, not silently suppressed.

Radar commands:
```bash
bash modality_audit_20261010/run_radar.sh
```
Check experiment-to-config mapping in run_radar.sh before use. Missing checkpoints are explicitly
skipped; no checkpoint is copied, deleted or retrained. No global RDQ_LIDAR_CONFIG override.
RMSE is conditional on finite top1 outputs, with coverage and missing count. Success counts no-output
as failures. Includes global and per-sequence XYZ/3D RMSE, axis bias/MAE and error quantiles.
Global RMSE is calculated by pooling queries, never averaging sequence RMSE.
Heldout is for reporting; don't select models or tune against heldout.
No day/night labels assumed. Physical XYZ axis directions remain unverified.
Metrics are final pipeline COCO AP, <=10 boxes/image, annotated images only, not raw DINO AP.

Upload result directories without checkpoints for review. Synthetic checks do not replace real-data audit.
