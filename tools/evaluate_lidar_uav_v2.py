#!/usr/bin/env python3
"""Full CSV validation with rolling causal query history; endpoints counted once."""
import argparse,sys,json
from pathlib import Path
import torch,yaml
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (LiDARUAVDetector,LiDARUAVValidationDataset,TemporalQueryClipDataset,
    collate_temporal_queries,QueryCausalLoss,CandidateSelector)
from rdq_uav.lidar_v2.training import validate,write_csv
from rdq_uav.lidar_v2.contracts import effective_config,validate_frozen_v2_config

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/lidar_uav_v2.yaml')
    p.add_argument('--val-root',type=Path,default=Path('/home/jasoncui/datasets/MMAUD/official/val'))
    p.add_argument('--val-reference',type=Path,default=Path('/home/jasoncui/datasets/MMAUD/official/validation_ref_new (for your ref).csv'))
    p.add_argument('--device',default='cpu');p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--export-candidates',action='store_true');a=p.parse_args()
    protected=(ROOT/'outputs/own_multimodal_research/lidar_uav_v1').resolve()
    if a.output.resolve()==protected or protected in a.output.resolve().parents:raise ValueError('V1 output is frozen')
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(a.config.read_text());validate_frozen_v2_config(cfg)
    cfg['data'].update(val_root=str(a.val_root),val_reference=str(a.val_reference))
    device=torch.device('cpu' if a.device=='cpu' else f'cuda:{a.device}')
    effective_cfg,training_precision,evaluation_precision=effective_config(cfg,device)
    (a.output/'effective_config.yaml').write_text(yaml.safe_dump(effective_cfg,sort_keys=False))
    (a.output/'effective_config.json').write_text(json.dumps(effective_cfg,indent=2))
    print(f'Training precision: {training_precision.effective}\nEvaluation precision: {evaluation_precision.effective}')
    state=torch.load(a.checkpoint,map_location=device);model=LiDARUAVDetector(cfg).to(device);model.load_state_dict(state['model_state'],strict=True)
    ds=TemporalQueryClipDataset(LiDARUAVValidationDataset(a.val_root,a.val_reference),cfg['temporal']['clip_length'],validation=True)
    loader=DataLoader(ds,batch_size=cfg['evaluation']['batch_size'],shuffle=False,num_workers=a.num_workers,collate_fn=collate_temporal_queries)
    metrics,rows,health=validate(model,loader,QueryCausalLoss(cfg),CandidateSelector(cfg),device,evaluation_precision,
        export_dir=a.output/'candidate_exports' if a.export_candidates else None,checkpoint_path=a.checkpoint)
    (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2));(a.output/'loss.json').write_text(json.dumps(health,indent=2))
    write_csv(a.output/'query_predictions.csv',rows)
if __name__=='__main__':main()
