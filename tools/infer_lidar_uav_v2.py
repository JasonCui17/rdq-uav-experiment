#!/usr/bin/env python3
"""GT-free arbitrary query inference returning raw and NMS candidate sets."""
import argparse,sys,json
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import LiDARQueryBuilder,collate_lidar_samples,LiDARUAVDetector,CandidateSelector

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sequence-id',required=True);p.add_argument('--query-times',nargs='+',type=float,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/lidar_uav_v2.yaml')
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text())
    query_time=args.query_times[-1]
    batch=collate_lidar_samples([LiDARQueryBuilder(args.root,cfg['data']['num_merged_frames']).build_inference_query(args.sequence_id,query_time)])
    model=LiDARUAVDetector(cfg);model.load_state_dict(torch.load(args.checkpoint,map_location='cpu')['model_state'],strict=True);model.eval()
    with torch.no_grad():out=model(batch);c=CandidateSelector(cfg)(out)[-1]
    print(json.dumps(dict(query_time=query_time,raw_xyz=c['raw']['xyz'].tolist(),
        raw_scores=c['raw']['score'].tolist(),nms_xyz=c['nms']['xyz'].tolist(),nms_scores=c['nms']['score'].tolist())))
if __name__=='__main__':main()
