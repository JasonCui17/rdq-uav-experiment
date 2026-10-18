#!/usr/bin/env python3
"""Self-contained RDQ-UAV train, inference, and post-inference evaluation."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'third_party/detrex'),str(ROOT/'third_party/detrex/detectron2'),str(ROOT/'third_party/stonesoup')]


def environment(data_root):
    env={k:v for k,v in os.environ.items() if not k.startswith('RDQ_')}
    env.update(PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src',ROOT/'tools',ROOT/'third_party/detrex',ROOT/'third_party/detrex/detectron2',ROOT/'third_party/stonesoup'])),RDQ_DATA_ROOT=str(data_root),RDQ_SPLIT_FILE=str(ROOT/'splits/mmaud_v2_annotated20_seed42.json'),RDQ_LIDAR_CONFIG=str(ROOT/'configs/final/radar_architecture.yaml'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env


def execute(script,args,env):
    command=[sys.executable,'-u','-m',script,*map(str,args)]
    print('RUN:', ' '.join(command),flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)


def doctor():
    import torch,importlib
    from rdq_uav.multimodal_v3.runtime_paths import ensure_detrex_config_link
    ensure_detrex_config_link(ROOT/'third_party/detrex')
    results={'root':str(ROOT),'python':sys.version,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'modules':{},'weights':{}}
    for name in ['rdq_uav','detrex','detectron2','stonesoup','pycocotools']:
        module=importlib.import_module(name);path=Path(module.__file__).resolve();results['modules'][name]=str(path)
        if name in ['rdq_uav','detrex','detectron2','stonesoup'] and not path.is_relative_to(ROOT):raise RuntimeError('External project import: '+str(path))
    coco=ROOT/'checkpoints/dino_swin_t/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth'
    if not coco.is_file():raise FileNotFoundError(coco)
    import detrex._C
    if not Path(detrex._C.__file__).resolve().is_relative_to(ROOT):raise RuntimeError('External detrex extension')
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists():
        data=json.loads(manifest.read_text())
        for key,digest in data['files_sha256'].items():
            if hashlib.sha256((ROOT/key).read_bytes()).hexdigest()!=digest:raise RuntimeError('Release file hash mismatch: '+key)
    for name in ['radar.ckpt','radar_warmstart.ckpt','vision_full.ckpt','vision_crop.ckpt']:
        p=ROOT/'weights'/name
        if not p.is_file():raise FileNotFoundError(p)
        payload=torch.load(p,map_location='cpu',weights_only=False)
        results['weights'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'state_dict_keys':len(payload.get('state_dict',payload))}
    print(json.dumps(results,ensure_ascii=False,indent=2));return results


def infer(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU inference required; CUDA unavailable')
    data=args.data_root.resolve()
    if not data.is_dir():raise FileNotFoundError(data)
    output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    weights=[(ROOT/p).resolve() for p in [args.radar_checkpoint,args.full_checkpoint,args.crop_checkpoint]]
    if any(not p.is_relative_to(ROOT) for p in weights):raise ValueError('All model checkpoints must be inside rdq-uav-final')
    if not all(p.is_file() for p in weights):raise FileNotFoundError('Incomplete final weights')
    if args.sequences:sequences=args.sequences
    elif args.split=='all':sequences=sorted(p.name for p in data.glob('seq*') if p.is_dir())
    else:sequences=json.loads((ROOT/'splits/mmaud_v2_annotated20_seed42.json').read_text())[args.split]
    if len(set(sequences))!=len(sequences) or not sequences:raise ValueError('Empty/duplicate sequences')
    if any(Path(s).name!=s or not (data/s).is_dir() for s in sequences):raise ValueError('Unknown sequence')
    output.mkdir(parents=True);env=environment(data);all_metrics={}
    status={'complete':False,'sequences':sequences,'schedule':args.schedule,'GT_used_for_sensor_inference':False,'all_left_images_required':args.schedule=='sensor' and args.limit_queries is None,'smoke':args.limit_queries is not None,'root':str(ROOT),'data_root':str(data),'GPU':torch.cuda.get_device_name(0)}
    (output/'run_status.json').write_text(json.dumps(status,indent=2))
    from score_outputs import score,read,labeled_predictions
    for seq in sequences:
        directory=output/seq;directory.mkdir()
        extra=[] if args.limit_queries is None else ['--limit',args.limit_queries]
        execute('rdq_uav.multimodal_v3.pipeline.extract_observations',['--repo',ROOT,'--config','configs/final/radar.yaml','--checkpoint',weights[0],'--vision-checkpoint',weights[1],'--data-root',data,'--split','heldout_test_sub' if args.split=='heldout_test_sub' else 'validation_sub','--sequence',seq,'--schedule',args.schedule,'--cache',directory/'cache','--device','cuda:0','--num-workers',args.num_workers,*extra],env)
        execute('rdq_uav.multimodal_v3.pipeline.evaluate_visual_birth',['--cache',directory/'cache','--samples',directory/'cache/image_samples.jsonl','--visual-cache',directory/'cache/raw_view_cache.jsonl','--config','configs/final/tracker.json','--output',directory/'tracks'],env)
        execute('rdq_uav.multimodal_v3.pipeline.prepare_image_guidance',['--samples',directory/'cache/image_samples.jsonl','--tracks',directory/'tracks/per_query.jsonl','--config','configs/final/image_guidance.json','--all-available-images','--output',directory/'priors'],env)
        execute('rdq_uav.multimodal_v3.pipeline.evaluate_image_guidance',['--checkpoint',weights[1],'--crop-checkpoint',weights[2],'--samples',directory/'cache/image_samples.jsonl','--priors',directory/'priors/image_priors.jsonl','--data-root',data,'--config','configs/final/image_guidance.json','--all-available-images','--output',directory/'image_eval'],env)
        tracks=read(directory/'tracks/per_query.jsonl');public=[]
        for r in tracks:public.append({'sequence_id':seq,'timestamp':r['query_time'],'xyz_m':r['xyz'],'lag05_xyz_m':r.get('lagged_xyz'),'velocity_mps':r.get('velocity_mps'),'event':r['event'],'output_kind':r.get('output_kind'),'lookahead_s':r.get('lookahead_s')})
        (directory/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in public))
        with (directory/'trajectory.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['timestamp','x','y','z','lag05_x','lag05_y','lag05_z','event'])
            for r in public:writer.writerow([r['timestamp'],*(r['xyz_m'] or [None]*3),*(r['lag05_xyz_m'] or [None]*3),r['event']])
        causal=read(directory/'image_eval/track_crop_causal.jsonl');lagged={r['image_path']:r for r in read(directory/'image_eval/track_crop_lag05.jsonl')};bbox=[]
        for r in causal:
            b=lagged[r['image_path']];bbox.append({'sequence_id':seq,'image_path':r['image_path'],'image_time':r['image_time'],'source_wh':[1280,960],'bbox_format':'xyxy_source_left_pixels','realtime':{'boxes':r['pred_boxes_xyxy_px'],'scores':r['pred_box_scores']},'lag05':{'boxes':b['pred_boxes_xyxy_px'],'scores':b['pred_box_scores']}})
        (directory/'bboxes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bbox))
        expected={f'{seq}/Image/{p.name}' for p in (data/seq/'Image').glob('*.png')};actual={r['image_path'] for r in bbox}
        if status['all_left_images_required'] and actual!=expected:raise RuntimeError(f'Image coverage mismatch: {len(actual)}/{len(expected)}')
        metrics=score(data,seq,directory/'tracks',directory/'image_eval',args.schedule)
        metrics['coverage']={'all_left_images':len(expected),'predicted_images':len(actual),'complete_image_coverage':actual==expected,'sensor_queries':len(public)}
        (directory/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2));all_metrics[seq]=metrics
        print('SEQUENCE COMPLETE:',seq,metrics['coverage'],flush=True)
    # Image AP is pooled over images; it is not the average of sequence AP values.
    from rdq_uav.multimodal_v3.detection_metrics import image_detection_metrics
    pooled={}
    for mode,name in [('realtime','track_crop_causal'),('lag05','track_crop_lag05')]:
        rows=[];missing=0
        for seq in sequences:
            labeled,count=labeled_predictions(data,seq,output/seq/'image_eval',name)
            rows.extend(labeled);missing+=count
        pooled[mode]=image_detection_metrics(rows)
        pooled[mode]['unlabeled_images_not_scored']=missing
    radar={}
    for mode in ['realtime','lag05']:
        parts=[m['radar'][mode] for m in all_metrics.values()];n=sum(p['scored_outputs'] for p in parts);g=sum(p['GT_timestamps'] for p in parts)
        radar[mode]={'rmse_3d_m':(sum(p['rmse_3d_m']**2*p['scored_outputs'] for p in parts if p['rmse_3d_m'] is not None)/n)**.5 if n else None,'scored_outputs':n,'GT_timestamps':g,'coverage_at_GT':n/g if g else None}
    status['complete']=True
    (output/'metrics.json').write_text(json.dumps({'protocol':status,'radar':radar,'image':pooled,'sequences':all_metrics},ensure_ascii=False,indent=2))
    status['complete']=True;(output/'run_status.json').write_text(json.dumps(status,indent=2));print('RESULTS:',output/'metrics.json',flush=True)


def train(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU training required')
    env=environment(args.data_root.resolve());common=['--output',args.output.resolve(),'--accelerator','gpu','--devices',1,'--precision','32-true' if args.stage=='radar' else '16-mixed','--batch-size',args.batch_size,'--accumulate',args.accumulate,'--num-workers',args.num_workers,'--epochs',args.epochs]
    if args.fast_dev_run:common+=['--fast-dev-run']
    if args.stage=='radar':execute('rdq_uav.multimodal_v3.training.train',['--config','configs/final/radar.yaml',*common],env)
    elif args.stage=='image-coco':execute('rdq_uav.multimodal_v3.training.train',['--config','configs/final/vision_coco.yaml',*common],env)
    else:execute('rdq_uav.multimodal_v3.training.train_image',['--warmstart','weights/vision_full.ckpt',*(['--crop-training'] if args.stage=='image-crop' else []),*common],env)


def main():
    os.chdir(ROOT);p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('doctor')
    f=sub.add_parser('infer');f.add_argument('--data-root',type=Path,required=True);f.add_argument('--output',type=Path,required=True);f.add_argument('--split',choices=['validation_sub','heldout_test_sub','train_sub','all'],default='validation_sub');f.add_argument('--sequences',nargs='+');f.add_argument('--schedule',choices=['sensor','benchmark'],default='sensor');f.add_argument('--limit-queries',type=int);f.add_argument('--num-workers',type=int,default=2);f.add_argument('--radar-checkpoint',type=Path,default=Path('weights/radar.ckpt'));f.add_argument('--full-checkpoint',type=Path,default=Path('weights/vision_full.ckpt'));f.add_argument('--crop-checkpoint',type=Path,default=Path('weights/vision_crop.ckpt'))
    t=sub.add_parser('train');t.add_argument('--data-root',type=Path,required=True);t.add_argument('--output',type=Path,required=True);t.add_argument('--stage',choices=['radar','image-coco','image-full','image-crop'],required=True);t.add_argument('--epochs',type=int,default=12);t.add_argument('--batch-size',type=int,default=4);t.add_argument('--accumulate',type=int,default=1);t.add_argument('--num-workers',type=int,default=2);t.add_argument('--fast-dev-run',action='store_true')
    args=p.parse_args()
    if args.command=='doctor':doctor()
    elif args.command=='infer':infer(args)
    else:train(args)


if __name__=='__main__':main()
