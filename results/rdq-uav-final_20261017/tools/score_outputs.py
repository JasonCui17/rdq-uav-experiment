"""Post-inference scoring only; no annotations enter sensor inference."""
import json
from pathlib import Path
import numpy as np
from rdq_uav.multimodal_v3.modeling.data import load_yolo_box
from rdq_uav.multimodal_v3.detection_metrics import image_detection_metrics


def read(path):return [json.loads(x) for x in Path(path).open() if x.strip()]


def labeled_predictions(data_root, sequence, image_dir, method):
    labeled=[];missing=0
    for row in read(image_dir/(method+'.jsonl')):
        label=data_root/sequence/'2d_detect'/Path(row['image_path']).with_suffix('.txt').name
        tensor=load_yolo_box(label,(1280,960))
        if tensor is None:missing+=1;continue
        labeled.append({**row,'has_gt2d':True,'gt_box_xyxy_px':tensor.tolist()})
    return labeled,missing


def score(data_root,sequence,tracks,image_dir,schedule):
    methods={}
    for mode,name in [('realtime','track_crop_causal'),('lag05','track_crop_lag05')]:
        labeled,missing=labeled_predictions(data_root,sequence,image_dir,name)
        methods[mode]=image_detection_metrics(labeled)
        methods[mode]['unlabeled_images_not_scored']=missing
    rows=read(tracks/'per_query.jsonl');times=np.array([r['query_time'] for r in rows]);radar={}
    for mode,key in [('realtime','xyz'),('lag05','lagged_xyz')]:
        distances=[];n=0
        for path in sorted((data_root/sequence/'ground_truth').glob('*.npy')):
            gt=np.load(path,allow_pickle=False).reshape(3)
            if not np.isfinite(gt).all():continue
            n+=1;t=float(path.stem);right=int(np.searchsorted(times,t))
            if right<len(times) and times[right]==t:xyz=rows[right].get(key)
            elif mode=='realtime':
                left=right-1
                if left<0 or t-times[left]>.25 or rows[left].get('xyz') is None or rows[left].get('velocity_mps') is None:xyz=None
                else:xyz=np.array(rows[left]['xyz'])+(t-times[left])*np.array(rows[left]['velocity_mps'])
            elif right==0 or right==len(times) or times[right]-times[right-1]>.25:xyz=None
            else:
                a,b=rows[right-1],rows[right]
                if a.get(key) is None or b.get(key) is None or b['event']=='birth':xyz=None
                else:
                    alpha=(t-times[right-1])/(times[right]-times[right-1]);xyz=(1-alpha)*np.array(a[key])+alpha*np.array(b[key])
            if xyz is not None:distances.append(float(np.linalg.norm(np.array(xyz)-gt)))
        d=np.array(distances)
        radar[mode]={'GT_timestamps':n,'scored_outputs':len(d),'missing_at_GT':n-len(d),'coverage_at_GT':len(d)/n if n else None,'rmse_3d_m':float(np.sqrt(np.mean(d*d))) if len(d) else None,'success_1m_all_GT':int(sum(d<=1))/n if n else None,'severe_gt10m':int(sum(d>10)),'output_timeline_coverage':sum(r.get(key) is not None for r in rows)/len(rows) if rows else None,'metric_alignment':'exact GT timestamps' if schedule=='benchmark' else ('causal past-state velocity extrapolation to GT timestamp; max age 0.25s' if mode=='realtime' else 'post-inference linear interpolation at GT timestamps; max bracket 0.25s; no gap/birth bridging')}
    return {'sequence_id':sequence,'schedule':schedule,'radar':radar,'image':methods,'annotation_protocol':'MMAUD 2d_detect labels only; missing or empty labels excluded, not negative','inference_used_GT':schedule=='benchmark','benchmark_GT_usage':'filenames define schedule; values only used for scoring' if schedule=='benchmark' else 'no GT inputs during inference'}
