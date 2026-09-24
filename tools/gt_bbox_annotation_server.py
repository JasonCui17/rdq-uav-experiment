#!/usr/bin/env python3
"""WSL-friendly MMAUD left-camera GT-guided YOLO annotation web server.

Run from repository root:
 python tools/gt_bbox_annotation_server.py --host 127.0.0.1 --port 8765
Uses only numpy, PyYAML, Pillow and the Python standard library.
Never writes unconfirmed predictions to YOLO labels.
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
import math
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / 'annotation_web' / 'index.html'
SEQ_PATTERN = re.compile(r'^seq[0-9]{4,}$')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}


class UserError(ValueError):
    pass


def atomic_write(path: Path, contents: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + f'.tmp.{os.getpid()}.{threading.get_ident()}')
    try:
        temporary.write_text(contents, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def project_omni(xyz, cal):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return None
    x, y, z = cal['R'] @ xyz + cal['t']
    xi, fu, fv, pu, pv = cal['intr']
    k1, k2, p1, p2 = cal['dist']
    norm = math.sqrt(x*x+y*y+z*z)
    denom = z + xi*norm
    if not math.isfinite(denom) or denom <= 1e-12:
        return None
    xn, yn = x/denom, y/denom
    r2 = xn*xn+yn*yn
    radial = 1+k1*r2+k2*r2*r2
    xd = xn*radial + 2*p1*xn*yn+p2*(r2+2*xn*xn)
    yd = yn*radial + p1*(r2+2*yn*yn)+2*p2*xn*yn
    u, v = fu*xd+pu, fv*yd+pv
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    if not (0 <= u < cal['wh'][0] and 0 <= v < cal['wh'][1]):
        return None
    return [float(u), float(v)]


def load_calibration(camera: Path, geometry: Path):
    cam = yaml.safe_load(camera.read_text(encoding='utf-8'))['cameras']['left']
    geo = json.loads(geometry.read_text(encoding='utf-8'))
    if cam.get('model') != 'omni' or cam.get('distortion_model') != 'radtan':
        raise UserError('Requires left omni/radtan camera')
    if geo.get('time_convention') != 'gt_query_time = image_time + time_offset_s':
        raise UserError('Unrecognized GT/image time convention')
    ext = geo['cameras']['left']
    cal = {'R': np.asarray(ext['rotation_camera_from_gt'], dtype=np.float64),
           't': np.asarray(ext['translation_camera_from_gt_m'], dtype=np.float64),
           'intr': np.asarray(cam['intrinsics'], dtype=np.float64),
           'dist': np.asarray(cam['distortion_coeffs'], dtype=np.float64),
           'wh': tuple(int(x) for x in cam['resolution']),
           'dt': float(geo['time_offset_s'])}
    if cal['R'].shape != (3, 3) or cal['t'].shape != (3,) or cal['intr'].shape != (5,) or cal['dist'].shape != (4,):
        raise UserError('Malformed calibration array')
    if not np.isfinite(cal['R']).all() or not np.isfinite(cal['t']).all() or not np.isfinite(cal['intr']).all() or not np.isfinite(cal['dist']).all():
        raise UserError('Non-finite calibration')
    return cal


def yolo_to_xyxy(row, wh):
    fields = row.strip().split()
    if len(fields) != 5:
        raise UserError(f'YOLO row must have 5 columns: {row!r}')
    category = int(fields[0]); cx, cy, w, h = (float(v) for v in fields[1:])
    if category < 0 or not all(math.isfinite(t) for t in (cx, cy, w, h)) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise UserError(f'Invalid YOLO row: {row!r}')
    W, H = wh
    box = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
    if box[0] < -1e-3 or box[1] < -1e-3 or box[2] > W+1e-3 or box[3] > H+1e-3:
        raise UserError(f'YOLO box outside left camera frame: {row!r}')
    return {'class_id': category, 'box': [max(0,box[0]),max(0,box[1]),min(W,box[2]),min(H,box[3])]}


def xyxy_to_yolo(ann, wh):
    W, H = wh
    if type(ann.get('class_id')) is not int or ann['class_id'] < 0:
        raise UserError('class_id must be a nonnegative integer')
    box = ann.get('box')
    if not isinstance(box, list) or len(box) != 4:
        raise UserError('box must be [x1,y1,x2,y2]')
    x1,y1,x2,y2 = [float(v) for v in box]
    if not all(math.isfinite(x) for x in [x1,y1,x2,y2]) or not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
        raise UserError(f'Invalid bbox {box} for left image size {wh}')
    return f"{ann['class_id']} {((x1+x2)/2/W):.9f} {((y1+y2)/2/H):.9f} {((x2-x1)/W):.9f} {((y2-y1)/H):.9f}"


def point_box_distance(point, box):
    u,v = point; x1,y1,x2,y2 = box
    return math.hypot(max(x1-u,0,u-x2),max(y1-v,0,v-y2))


@dataclass
class Frame:
    name: str
    time: float
    image_path: Path
    projection: list | None
    gt_gap_s: float | None
    gt_path: str | None
    boxes: list[dict]
    status: str  # confirmed, confirmed_negative, uncertain, draft, invalid, unlabeled
    source: str
    force_anchor: bool = False
    candidate_anchor: str | None = None
    warning: str | None = None


class SequenceStore:
    def __init__(self, data_root: Path, seq: str, cal, *, max_gt_gap_s=.08, discrepancy_px=32., backup=True):
        if not SEQ_PATTERN.fullmatch(seq):
            raise UserError('Invalid sequence ID')
        self.data_root=data_root
        self.seq=seq
        self.root=data_root/seq
        if not self.root.is_dir() or not (self.root/'Image').is_dir():
            raise UserError(f'Missing sequence Image/: {seq}')
        self.cal=cal
        self.max_gt_gap_s=float(max_gt_gap_s)
        self.discrepancy_px=float(discrepancy_px)
        self.backup=backup
        self.labels_dir=self.root/'2d_detect'
        self.state_path=self.labels_dir/'.gt_bbox_review_state.json'
        self.frames=[]
        self.by_name={}
        self.lock=threading.RLock()
        self._load()

    def _load(self):
        try:
            old=json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {}
        except (ValueError,OSError) as ex:
            raise UserError(f'Invalid review state {self.state_path}: {ex}')
        image_paths=[]
        for p in (self.root/'Image').iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                try: t=float(p.stem)
                except ValueError: continue
                if math.isfinite(t): image_paths.append((t,p))
        image_paths.sort(key=lambda pair:(pair[0],pair[1].name))
        if not image_paths: raise UserError(f'No timestamp-named images under {self.root/"Image"}')
        gt_files=[]
        gt_root=self.root/'ground_truth'
        if gt_root.is_dir():
            for p in gt_root.glob('*.npy'):
                try: t=float(p.stem)
                except ValueError: continue
                if math.isfinite(t): gt_files.append((t,p))
        gt_files.sort(key=lambda pair:pair[0]); gt_times=[t for t,_ in gt_files]
        reviews=old.get('frames',{})
        for t,path in image_paths:
            gt_target=t+self.cal['dt']
            position=bisect.bisect_left(gt_times,gt_target)
            candidates=[k for k in (position-1,position) if 0<=k<len(gt_times)]
            projection=None; gap=None; gt_path=None; warning=None
            if candidates:
                k=min(candidates,key=lambda i:(abs(gt_times[i]-gt_target),i))
                gap=abs(gt_times[k]-gt_target)
                gt_path=gt_files[k][1].name
                if gap<=self.max_gt_gap_s:
                    try: xyz=np.load(gt_files[k][1],allow_pickle=False).reshape(-1)
                    except (ValueError,OSError) as exc: warning=f'GT unreadable: {exc}'
                    else: projection=project_omni(xyz,self.cal)
                    if projection is None and warning is None: warning='GT projection invalid/outside left image'
                else: warning=f'Nearest GT gap {gap:.4f}s > {self.max_gt_gap_s:.4f}s'
            else: warning='No GT files'
            label=self.labels_dir/(path.stem+'.txt')
            boxes=[]; status='unlabeled'; source='none'; force_anchor=False
            if label.exists():
                try:
                    boxes=[yolo_to_xyxy(line,self.cal['wh']) for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
                except (ValueError,UserError) as exc:
                    warning=f'Invalid existing YOLO label: {exc}'
                    status='invalid';source='existing_invalid'
                else:
                    status='confirmed' if boxes else 'confirmed_negative'
                    source='existing_yolo'
            meta=reviews.get(path.name,{})
            if status=='unlabeled' and meta.get('status')=='uncertain':
                status='uncertain';source='manual_uncertain'
            if status=='confirmed' and meta.get('force_anchor'):
                force_anchor=True
            frame=Frame(path.name,t,path,projection,gap,gt_path,boxes,status,source,force_anchor,warning=warning)
            self.frames.append(frame);self.by_name[frame.name]=frame
        self._regenerate(initial=True)

    def _save_review(self):
        frames={f.name:{'status':f.status, 'source':f.source, 'force_anchor':f.force_anchor,
                        'confirmed_at':time.time() if f.status in {'confirmed','confirmed_negative'} else None}
                for f in self.frames if f.status in {'confirmed','confirmed_negative','uncertain'}}
        atomic_write(self.state_path,json.dumps({'version':1,'sequence':self.seq,'frames':frames},ensure_ascii=False,indent=2)+'\n')

    def _backup_label(self,label: Path):
        if not self.backup or not label.exists():return
        dest=self.labels_dir/'_annotation_backup'/label.name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(label,dest)

    def _save_label(self,frame:Frame):
        label=self.labels_dir/(Path(frame.name).stem+'.txt')
        self._backup_label(label)
        contents='\n'.join(xyxy_to_yolo(a,self.cal['wh']) for a in frame.boxes)
        atomic_write(label,contents+('\n' if contents else ''))
        self._save_review()

    def _usable_anchor(self,f:Frame):
        if f.status!='confirmed' or len(f.boxes)!=1 or f.projection is None:
            return False
        if f.force_anchor:return True
        return point_box_distance(f.projection,f.boxes[0]['box'])<=self.discrepancy_px

    def _clear_drafts(self,*,after_index=None):
        for i,f in enumerate(self.frames):
            if after_index is not None and i<=after_index:continue
            if f.status in {'draft','invalid','unlabeled'}:
                f.boxes=[];f.status='unlabeled';f.source='none';f.candidate_anchor=None
                if f.projection is None:f.status='invalid'

    def _regenerate(self,*,initial=False,after_index=None):
        # Initial fill permits backward propagation from first eligible anchor, except across negative barriers.
        # Subsequent updates affect only unconfirmed frames AFTER the edited frame.
        self._clear_drafts(after_index=None if initial else after_index)
        latest=None
        future_anchors=[i for i,f in enumerate(self.frames) if self._usable_anchor(f)]
        for i,f in enumerate(self.frames):
            if f.status=='confirmed_negative':latest=None;continue
            if self._usable_anchor(f): latest=f;continue
            if f.status in {'confirmed','uncertain'}:continue
            if not initial and after_index is not None and i<=after_index:continue
            if f.projection is None:
                f.status='invalid';continue
            anchor=latest
            if anchor is None and initial:
                # Allow backward fill ONLY before the first valid anchor in the current segment.
                barrier=next((j for j in range(i,len(self.frames)) if self.frames[j].status=='confirmed_negative'),len(self.frames))
                j=next((j for j in future_anchors if i<j<barrier),None)
                if j is not None:anchor=self.frames[j]
            if anchor is None: f.status='unlabeled';continue
            du=f.projection[0]-anchor.projection[0];dv=f.projection[1]-anchor.projection[1]
            W,H=self.cal['wh']; x1,y1,x2,y2=anchor.boxes[0]['box']
            box=[max(0.,min(float(W),x1+du)),max(0.,min(float(H),y1+dv)),
                 max(0.,min(float(W),x2+du)),max(0.,min(float(H),y2+dv))]
            if box[2]-box[0] <= 0.001 or box[3]-box[1] <= 0.001:
                f.status='invalid';f.warning='Propagated bbox is fully outside image';continue
            f.boxes=[{'class_id':anchor.boxes[0]['class_id'],'box':box}]
            f.status='draft';f.source='gt_shift';f.candidate_anchor=anchor.name

    def _serial(self,f,i):
        discrepancy=None
        if f.projection is not None and f.boxes:
            discrepancy=round(point_box_distance(f.projection,f.boxes[0]['box']),2)
        warning=f.warning
        if f.status=='confirmed' and discrepancy is not None and discrepancy>self.discrepancy_px and not f.force_anchor:
            warning=f'GT point {discrepancy:.1f}px outside bbox: anchor paused; manually verify or force-enable'
        return {'index':i,'name':f.name,'time':f.time,'boxes':f.boxes,'status':f.status,'source':f.source,
                'projection':f.projection,'gt_gap_s':f.gt_gap_s,'gt_name':f.gt_path,
                'candidate_anchor':f.candidate_anchor,'anchor_usable':self._usable_anchor(f),
                'force_anchor':f.force_anchor,'discrepancy_px':discrepancy,'warning':warning}

    def snapshot(self):
        with self.lock:
            rows=[self._serial(f,i) for i,f in enumerate(self.frames)]
            counts={k:sum(f.status==k for f in self.frames) for k in ('confirmed','confirmed_negative','draft','invalid','unlabeled','uncertain')}
            return {'sequence':self.seq,'resolution':list(self.cal['wh']),'counts':counts,'frames':rows,
                    'settings':{'max_gt_gap_s':self.max_gt_gap_s,'discrepancy_px':self.discrepancy_px}}

    def _get(self,name):
        f=self.by_name.get(name)
        if f is None:raise UserError('Image not in selected sequence')
        return f

    def save(self,name,boxes,*,uncertain=False,force_anchor=False):
        with self.lock:
            f=self._get(name); i=self.frames.index(f)
            if uncertain:
                # Uncertain frames are excluded from training; do not create empty YOLO negatives.
                label=self.labels_dir/(Path(f.name).stem+'.txt')
                if label.exists() and f.status in {'confirmed','confirmed_negative'}:
                    raise UserError('Cannot mark a confirmed YOLO label uncertain; delete/replace the label explicitly first')
                f.boxes=[];f.status='uncertain';f.source='manual_uncertain';f.force_anchor=False
                self._save_review()
            else:
                if not isinstance(boxes,list):raise UserError('boxes must be a list')
                # Round trip through YOLO validator, canonicalize to strictly valid boxes.
                canonical=[yolo_to_xyxy(xyxy_to_yolo(a,self.cal['wh']),self.cal['wh']) for a in boxes]
                f.boxes=canonical;f.status='confirmed' if canonical else 'confirmed_negative'
                f.source='manual_review';f.force_anchor=bool(force_anchor) if canonical else False
                f.candidate_anchor=None
                self._save_label(f)
            self._regenerate(initial=False,after_index=i)
            return self.snapshot()


    def bulk(self, names, action):
        if action not in {"delete", "confirm"}:
            raise UserError("Unsupported batch action")
        if not isinstance(names, list) or not names:
            raise UserError("Select at least one image")
        if len(set(names)) != len(names):
            raise UserError("Duplicate image selection")

        with self.lock:
            frames = [self._get(name) for name in names]

            confirmed = []
            skipped = []

            for f in frames:
                if action == "confirm":
                    # 已有人工标注、无框或异常帧均跳过，不中断整个批次。
                    if f.status in {"confirmed", "confirmed_negative"}:
                        skipped.append({"name": f.name, "reason": "already_confirmed"})
                        continue

                    if f.status != "draft":
                        skipped.append({"name": f.name, "reason": f.status})
                        continue

                    if not f.boxes:
                        skipped.append({"name": f.name, "reason": "no_bbox"})
                        continue

                    if f.warning:
                        skipped.append({"name": f.name, "reason": f.warning})
                        continue

                    # 先检查框是否合法，再修改状态。
                    try:
                        canonical = [
                            yolo_to_xyxy(
                                xyxy_to_yolo(box, self.cal["wh"]),
                                self.cal["wh"],
                            )
                            for box in f.boxes
                        ]
                    except (ValueError, UserError) as exc:
                        skipped.append({"name": f.name, "reason": str(exc)})
                        continue

                    f.boxes = canonical
                    f.status = "confirmed"
                    f.source = "manual_batch_confirm"
                    f.force_anchor = False

                else:
                    # 批量删除：保留原有行为，所选帧全部保存为空标签。
                    f.boxes = []
                    f.status = "confirmed_negative"
                    f.source = "manual_batch_delete"
                    f.force_anchor = False

                f.candidate_anchor = None
                self._save_label(f)
                confirmed.append(f.name)

            # 重新生成尚未确认的候选框；不改变已经确认的标签。
            if confirmed:
                self._regenerate(initial=True)

            result = self.snapshot()
            result["bulk_result"] = {
                "action": action,
                "selected": len(frames),
                "saved": len(confirmed),
                "skipped": skipped,
            }
            return result

    def recompute(self,after=None):
        with self.lock:
            if after is None:self._regenerate(initial=True)
            else:
                f=self._get(after);self._regenerate(initial=False,after_index=self.frames.index(f))
            return self.snapshot()


class App:
    def __init__(self,data_root,camera,geometry,max_gap,discrepancy,backup=True):
        self.data_root=Path(data_root).expanduser().resolve()
        self.cal=load_calibration(Path(camera),Path(geometry))
        self.max_gap=max_gap;self.discrepancy=discrepancy;self.backup=backup
        self.cache={};self.lock=threading.Lock()

    def sequences(self):
        if not self.data_root.is_dir():raise UserError(f'Dataset root missing: {self.data_root}')
        return sorted(p.name for p in self.data_root.iterdir() if p.is_dir() and SEQ_PATTERN.fullmatch(p.name) and (p/'Image').is_dir())

    def store(self,seq):
        if not SEQ_PATTERN.fullmatch(seq):raise UserError('Invalid sequence ID')
        with self.lock:
            if seq not in self.cache:
                self.cache[seq]=SequenceStore(self.data_root,seq,self.cal,max_gt_gap_s=self.max_gap,discrepancy_px=self.discrepancy,backup=self.backup)
            return self.cache[seq]


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def respond(self,code,payload):
            b=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)

        def do_GET(self):
            try:
                url=urlparse(self.path);p=parse_qs(url.query)
                first=lambda k:p.get(k,[''])[0]
                if url.path=='/':
                    html=WEB.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(html)));self.end_headers();self.wfile.write(html);return
                if url.path=='/api/sequences':self.respond(200,{'sequences':app.sequences()});return
                if url.path=='/api/frames':self.respond(200,app.store(first('seq')).snapshot());return
                if url.path=='/api/image':
                    store=app.store(first('seq'));f=store._get(first('name'))
                    with Image.open(f.image_path) as original:
                        image=ImageOps.exif_transpose(original)
                        W,H=app.cal['wh']
                        if image.height!=H or image.width not in (W,2*W):
                            raise UserError(f'Unexpected image dimensions {image.size}: {f.name}; expected {(W,H)} or {(2*W,H)}')
                        image=image.crop((0,0,W,H)).convert('RGB')
                        buf=io.BytesIO();image.save(buf,format='JPEG',quality=92)
                    content=buf.getvalue();self.send_response(200);self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',str(len(content)));self.send_header('Cache-Control','private, max-age=3600');self.end_headers();self.wfile.write(content);return
                self.respond(404,{'error':'Not found'})
            except (UserError,KeyError,ValueError) as exc:self.respond(400,{'error':str(exc)})
            except Exception as exc:self.respond(500,{'error':f'{type(exc).__name__}: {exc}'})

        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length>1_000_000:raise UserError('Request too large')
                payload=json.loads(self.rfile.read(length))
                url=urlparse(self.path)
                store=app.store(payload['seq'])
                if url.path=='/api/save':out=store.save(payload['name'],payload.get('boxes',[]),uncertain=bool(payload.get('uncertain',False)),force_anchor=bool(payload.get('force_anchor',False)))
                elif url.path=='/api/bulk':out=store.bulk(payload.get('names'),payload.get('action'))
                elif url.path=='/api/recompute':out=store.recompute(payload.get('after'))
                else:self.respond(404,{'error':'Not found'});return
                self.respond(200,out)
            except (UserError,KeyError,ValueError,TypeError) as exc:self.respond(400,{'error':str(exc)})
            except Exception as exc:self.respond(500,{'error':f'{type(exc).__name__}: {exc}'})
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/mmaud_official_train')
    parser.add_argument('--camera-config',type=Path,default=ROOT/'configs/calibration/mmaud_v1_omni.yaml')
    parser.add_argument('--geometry',type=Path,default=ROOT/'calibration/official_left_p4_current_geometry.json')
    parser.add_argument('--max-gt-gap-s',type=float,default=.08)
    parser.add_argument('--discrepancy-px',type=float,default=32.)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-backup',action='store_true')
    args=parser.parse_args()
    if args.max_gt_gap_s<0 or args.discrepancy_px<0:parser.error('Thresholds must be nonnegative')
    app=App(args.data_root,args.camera_config,args.geometry,args.max_gt_gap_s,args.discrepancy_px,backup=not args.no_backup)
    print(f'MMAUD GT bbox annotation: http://{args.host}:{args.port}/',flush=True)
    print(f'Data root: {app.data_root}; camera: left {app.cal["wh"]}; GT offset: {app.cal["dt"]}s',flush=True)
    with ThreadingHTTPServer((args.host,args.port),make_handler(app)) as server:
        server.serve_forever()


if __name__=='__main__':main()
