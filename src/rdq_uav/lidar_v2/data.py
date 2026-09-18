"""MMAUD dual-LiDAR dataset; variable points remain packed and unpadded."""
from __future__ import annotations
import ast,csv,json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from rdq_uav.multimodal.merged_lidar import LidarFrameEvent, load_released_xyz, merge_frame_streams, select_last_history

SENSORS = ((0, "Avia", "livox_avia"), (1, "Mid360", "lidar_360"))

def _paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.npy"), key=lambda p: (float(p.stem), str(p)))

class LiDARUAVDataset(Dataset[dict[str, Any]]):
    """Returns one GT-time sample with all valid points from the last 20 merged events."""
    def __init__(self, root: str|Path, split_file: str|Path, split: str, max_events: int=20,
                 sequence_limit: int|None=None, deterministic_indices: list[int]|None=None):
        self.root=Path(root); self.split_file=Path(split_file); self.max_events=max_events
        payload=json.loads(self.split_file.read_text()); sequences=list(payload[split])
        if sequence_limit is not None: sequences=sequences[:sequence_limit]
        self.streams={}; self.samples=[]
        for seq in sequences:
            sd=self.root/seq; streams=[]
            for sid,name,directory in SENSORS:
                streams.append([LidarFrameEvent(seq,float(p.stem),sid,name,p) for p in _paths(sd/directory)])
            self.streams[seq]=merge_frame_streams(streams)
            gt=_paths(sd/"ground_truth")
            indices=range(len(gt)) if deterministic_indices is None else [i for i in deterministic_indices if i<len(gt)]
            self.samples.extend((seq,i,p) for i,p in ((i,gt[i]) for i in indices))
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        seq,gt_index,gt_path=self.samples[index]; t0=float(gt_path.stem)
        selected=select_last_history(self.streams[seq],t0,self.max_events); recent_start=max(0,len(selected)-4)
        xyz=[]; sensor=[]; dt=[]; recent=[]
        for fi,event in enumerate(selected):
            points,_,_=load_released_xyz(event.file_path)
            if len(points):
                xyz.append(points.astype(np.float32)); sensor.append(np.full(len(points),event.sensor_id,np.int64))
                dt.append(np.full(len(points),event.timestamp-t0,np.float32)); recent.append(np.full(len(points),fi>=recent_start,bool))
        cat=lambda xs,shape,dtype: np.concatenate(xs) if xs else np.empty(shape,dtype)
        points=cat(xyz,(0,3),np.float32)
        return {"points":torch.from_numpy(points),"sensor_id":torch.from_numpy(cat(sensor,(0,),np.int64)),
                "delta_t":torch.from_numpy(cat(dt,(0,),np.float32)),"recent_mask":torch.from_numpy(cat(recent,(0,),bool)),
                "gt_xyz":torch.from_numpy(np.asarray(np.load(gt_path),np.float32).reshape(3)),
                "sequence_id":seq,"sample_id":f"{seq}_g{gt_index:06d}","t0":t0,"gt_timestamp":t0,
                "event_count":len(selected),"event_timestamps":[e.timestamp for e in selected],
                "gt_source":"sequence_ground_truth"}

class ValidationReferenceAdapter:
    """Strict adapter for validation_ref_new CSV; never reads sequence ground_truth."""
    EXPECTED_FIELDS=("Sequence","Timestamp","Position","Classification")
    def __init__(self,path:str|Path):
        self.path=Path(path); self.invalid_rows=[]; self.duplicate_keys=[]
        with self.path.open(newline="",encoding="utf-8-sig") as handle:
            reader=csv.DictReader(handle)
            if tuple(reader.fieldnames or ())!=self.EXPECTED_FIELDS: raise ValueError(f"Ambiguous validation schema: {reader.fieldnames}; expected {self.EXPECTED_FIELDS}")
            raw=list(reader)
        self.total_rows=len(raw); self.records=[]; seen=set()
        for index,row in enumerate(raw):
            try:
                sequence=row["Sequence"].strip(); timestamp=float(row["Timestamp"]); position=np.asarray(ast.literal_eval(row["Position"]),dtype=np.float64)
                if not sequence or position.shape!=(3,) or not np.isfinite(position).all() or not np.isfinite(timestamp): raise ValueError("invalid sequence/timestamp/position")
                key=(sequence,timestamp)
                if key in seen:self.duplicate_keys.append(key)
                seen.add(key);self.records.append({"sequence_id":sequence,"sample_id":f"{sequence}_valref_{index:06d}","t0":timestamp,"gt_timestamp":timestamp,"gt_xyz":position.astype(np.float32),"classification":row["Classification"],"csv_row":index+2})
            except Exception as exc:self.invalid_rows.append({"row":index+2,"error":repr(exc),"data":row})
        if self.duplicate_keys: raise ValueError(f"Duplicate validation (sequence,timestamp) keys: {self.duplicate_keys[:10]}")

class LiDARUAVValidationDataset(Dataset[dict[str,Any]]):
    """Validation samples whose t0/XYZ come exclusively from ValidationReferenceAdapter."""
    def __init__(self,root:str|Path,reference:str|Path,max_events:int=20):
        self.root=Path(root);self.adapter=ValidationReferenceAdapter(reference);self.max_events=max_events;self.streams={};self.missing_sequences=[]
        for sequence in sorted({r["sequence_id"] for r in self.adapter.records}):
            sd=self.root/sequence
            if not sd.is_dir():self.missing_sequences.append(sequence);self.streams[sequence]=[];continue
            streams=[]
            for sid,name,directory in SENSORS:streams.append([LidarFrameEvent(sequence,float(p.stem),sid,name,p) for p in _paths(sd/directory)])
            self.streams[sequence]=merge_frame_streams(streams)
    def __len__(self):return len(self.adapter.records)
    def __getitem__(self,index):
        record=self.adapter.records[index];selected=select_last_history(self.streams.get(record["sequence_id"],[]),record["t0"],self.max_events);recent_start=max(0,len(selected)-4);xyz=[];sensor=[];dt=[];recent=[]
        for fi,event in enumerate(selected):
            points,_,_=load_released_xyz(event.file_path)
            if len(points):xyz.append(points.astype(np.float32));sensor.append(np.full(len(points),event.sensor_id,np.int64));dt.append(np.full(len(points),event.timestamp-record["t0"],np.float32));recent.append(np.full(len(points),fi>=recent_start,bool))
        cat=lambda xs,shape,dtype:np.concatenate(xs) if xs else np.empty(shape,dtype)
        return {"points":torch.from_numpy(cat(xyz,(0,3),np.float32)),"sensor_id":torch.from_numpy(cat(sensor,(0,),np.int64)),"delta_t":torch.from_numpy(cat(dt,(0,),np.float32)),"recent_mask":torch.from_numpy(cat(recent,(0,),bool)),"gt_xyz":torch.from_numpy(record["gt_xyz"]),"sequence_id":record["sequence_id"],"sample_id":record["sample_id"],"t0":record["t0"],"gt_timestamp":record["gt_timestamp"],"event_count":len(selected),"event_timestamps":[e.timestamp for e in selected],"gt_source":"validation_ref_csv"}

def collate_lidar_samples(samples: list[dict[str,Any]]) -> dict[str,Any]:
    """Pack points from B samples; batch_index prevents all cross-sample geometry/attention."""
    counts=[len(x["points"]) for x in samples]
    def cat(key,shape,dtype): return torch.cat([x[key] for x in samples]) if sum(counts) else torch.empty(shape,dtype=dtype)
    return {"points":cat("points",(0,3),torch.float32),"sensor_id":cat("sensor_id",(0,),torch.long),
            "delta_t":cat("delta_t",(0,),torch.float32),"recent_mask":cat("recent_mask",(0,),torch.bool),
            "point_batch_index":torch.repeat_interleave(torch.arange(len(samples)),torch.tensor(counts)),
            "gt_xyz":torch.stack([x["gt_xyz"] for x in samples]),"sample_id":[x["sample_id"] for x in samples],
            "sequence_id":[x["sequence_id"] for x in samples],"t0":torch.tensor([x["t0"] for x in samples],dtype=torch.float64),
            "gt_timestamp":torch.tensor([x["gt_timestamp"] for x in samples],dtype=torch.float64),
            "gt_source":[x["gt_source"] for x in samples],
            "event_count":torch.tensor([x["event_count"] for x in samples]),"event_timestamps":[x["event_timestamps"] for x in samples]}
