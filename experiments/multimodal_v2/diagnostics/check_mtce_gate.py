"""Check A2/MTCE zero-init equality and export real per-bin statistics."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
import yaml

from rdq_uav.runtime_paths import apply_runtime_path_overrides
from rdq_uav.multimodal_v2.data import build_split_dataset, collate_multimodal_v2, radar_model_batch
from rdq_uav.multimodal_v2.radar_model import LiDARUAVDetector
from rdq_uav.multimodal_v2.radar_selector import CandidateSelector
from rdq_uav.multimodal_v2.radar_temporal_centroid import load_a2_warmstart

ROOT = Path(__file__).resolve().parents[3]


def resolve(value):
    path=Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/multimodal_v2/configs/b0_a2_mtce3.yaml")
    parser.add_argument("--split", choices=("train_sub","validation_sub"), default="validation_sub")
    parser.add_argument("--indices", nargs="+", type=int, default=[1,2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/mtce_equivalence_gate")
    args=parser.parse_args()
    cfg=apply_runtime_path_overrides(yaml.safe_load(resolve(args.config).read_text()))
    radar_cfg=yaml.safe_load(resolve(cfg["initialization"]["lidar_config"]).read_text())
    old_cfg=copy.deepcopy(radar_cfg)
    old_cfg["model"]["voxel"]["motion_centroid"]["enabled"]=False
    old=LiDARUAVDetector(old_cfg).eval().to(args.device)
    new=LiDARUAVDetector(radar_cfg).eval().to(args.device)
    checkpoint=resolve(cfg["initialization"]["b0_warmstart_checkpoint"])
    load_a2_warmstart(new,checkpoint)
    prefix="network.lidar.detector."
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)["state_dict"]
    old.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
    selector_cfg=yaml.safe_load(resolve(cfg["initialization"]["candidate_config"]).read_text())
    selector=CandidateSelector(selector_cfg["candidate"]["radar"])
    dataset=build_split_dataset(cfg,ROOT,args.split)
    output=resolve(args.output);output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for index in args.indices:
        sample=dataset[index]
        batch=radar_model_batch(collate_multimodal_v2([sample]))
        if not len(batch["points"]):
            raise ValueError(f"Index {index} has no radar points; choose another query")
        batch={k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        before,after=old(batch),new(batch)
        errors={}
        for key in ("logits","residual_xyz","pred_xyz","fine_features"):
            torch.testing.assert_close(before[key],after[key],rtol=1e-6,atol=1e-6)
            errors[key]=float((before[key]-after[key]).abs().max())
        for a,b in zip(selector(before),selector(after)):
            for kind in ("raw","nms"):
                assert torch.equal(a[kind]["source_token_id"],b[kind]["source_token_id"])
                torch.testing.assert_close(a[kind]["xyz"],b[kind]["xyz"],rtol=1e-6,atol=1e-6)
        _,debug=new.temporal_centroid(batch,after["layouts"],return_debug=True)
        torch.save([{k:v.cpu() for k,v in level.items()} for level in debug],output/f"query_{index}_temporal.pt")
        reports.append({"index":index,"sample_id":sample["sample_id"],"max_abs_difference":errors,
                        "levels":[{k:int(level[k]) for k in ("supported_pair_count","rejected_small_gap_count","valid_pair_count","voxels_with_motion_support")} for level in debug]})
    report={"checkpoint":str(checkpoint),"split":args.split,"passed":True,"queries":reports}
    (output/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
