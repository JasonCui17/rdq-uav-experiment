#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RDQ_DATA_ROOT="${RDQ_DATA_ROOT:-/root/autodl-tmp/MMAUD/official/train}"
# Preserve each experiment's embedding/backbone config, no cross-experiment override.
unset RDQ_LIDAR_CONFIG
found=0
while read -r name config experiment; do
  checkpoint="outputs/$experiment/checkpoints/best.ckpt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "SKIP missing checkpoint: $checkpoint"
    continue
  fi
  found=$((found+1))
  for split in validation_sub heldout_test_sub; do
    out="outputs/paper_radar_20261010/$name/$split"
    python experiments/multimodal_v2/evaluate.py \
      --config "experiments/multimodal_v2/configs/$config" --mode B0 \
      --checkpoint "$checkpoint" --split "$split" --output "$out" \
      --device cuda:0 --num-workers 2 --method "$name" --skip-coco
    python modality_audit_20261010/audit_modalities.py \
      --mode B0 --input "$out/per_query.jsonl" --output "$out/audit"
  done
done <<'MAP'
A0_SBE b0_a0_sbe.yaml radar_embedding_ab_seed42/a0_sbe
A1_Legacy b0_a1_legacy.yaml radar_embedding_ab_seed42/a1_legacy
A2_LearnedSBE32 b0_a2_learned_sbe32.yaml radar_embedding_ab_seed42/a2_learned_sbe32
A1_Deep b0_a1_deep.yaml a1_deep_seed42
A2_Deep b0_a2_deep.yaml a2_deep_seed42
B0_Original b0_standalone.yaml b0_formal_seed42
MAP
if [[ "$found" == 0 ]]; then
  echo 'No checkpoints found. Locate old output directories before evaluating.' >&2
  exit 1
fi
python - <<'PY'
import csv,json
from pathlib import Path
rows=[]
for p in sorted(Path('outputs/paper_radar_20261010').glob('*/*/audit/audit_summary.json')):
    report=json.loads(p.read_text())
    for group, metrics in report['groups'].items():
        rows.append({'experiment':p.parents[2].name,'split':p.parents[1].name,'group':group,
                     **{k:v for k,v in metrics.items() if not isinstance(v,(dict,list))}})
keys=list(dict.fromkeys(k for r in rows for k in r))
out=Path('outputs/paper_radar_20261010/radar_comparison.csv')
with out.open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
print(out)
PY
