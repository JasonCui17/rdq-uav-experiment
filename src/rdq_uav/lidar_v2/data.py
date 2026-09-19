"""Causal query construction; targets are optional supervision metadata only."""
from __future__ import annotations
import ast, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from rdq_uav.multimodal.merged_lidar import LidarFrameEvent, load_released_xyz, merge_frame_streams, select_last_history

SENSORS = ((0, "Avia", "livox_avia"), (1, "Mid360", "lidar_360"))
def _paths(directory):
    return sorted(Path(directory).glob("*.npy"), key=lambda p: (float(p.stem), str(p)))

@dataclass
class QueryRequest:
    sequence_id: str
    query_time: float
    target_timestamp: float | None = None
    target_xyz: Any = None
    target_valid: bool = False
    sample_id: str = ""

class LiDARQueryBuilder:
    """query seconds -> packed causal XYZ meters; never reads GT directories."""
    def __init__(self, root, max_events=20):
        self.root=Path(root); self.max_events=max_events; self.streams={}
    def stream(self, sequence_id):
        if sequence_id not in self.streams:
            self.streams[sequence_id]=merge_frame_streams([
                [LidarFrameEvent(sequence_id,float(p.stem),sid,name,p)
                 for p in _paths(self.root/sequence_id/directory)]
                for sid,name,directory in SENSORS])
        return self.streams[sequence_id]
    def build(self, sequence_id, query_time, target_xyz=None, target_timestamp=None,
              target_valid=False, sample_id=None, **metadata):
        query_time=float(query_time)
        if not np.isfinite(query_time): raise ValueError("Non-finite query_time")
        events=select_last_history(self.stream(sequence_id),query_time,self.max_events)
        parts=[]; sensors=[]; times=[]; recent=[]
        for i,event in enumerate(events):
            assert event.timestamp<=query_time, "Future event violation"
            points=load_released_xyz(event.file_path)[0].astype(np.float32)
            parts.append(points);sensors.append(np.full(len(points),event.sensor_id,np.int64))
            times.append(np.full(len(points),event.timestamp-query_time,np.float32))
            recent.append(np.full(len(points),i>=max(0,len(events)-4),bool))
        cat=lambda a,shape,dtype: torch.from_numpy(np.concatenate(a) if a else np.empty(shape,dtype))
        result={"points":cat(parts,(0,3),np.float32),"sensor_id":cat(sensors,(0,),np.int64),
                "delta_t":cat(times,(0,),np.float32),"recent_mask":cat(recent,(0,),bool),
                "sequence_id":sequence_id,"query_time":query_time,"num_samples":1,
                "sample_id":sample_id or f"{sequence_id}_query_{query_time:.9f}",
                "event_count":len(events),"event_timestamps":[e.timestamp for e in events],
                "has_observation":sum(len(p) for p in parts)>0,"target_valid":bool(target_valid),
                "query_valid":True,"metadata":metadata}
        if target_valid:
            if target_xyz is None or target_timestamp is None: raise ValueError("Valid target requires XYZ and timestamp")
            xyz=torch.as_tensor(target_xyz,dtype=torch.float32)
            if xyz.shape!=(3,) or not torch.isfinite(xyz).all() or not np.isfinite(target_timestamp):
                raise ValueError("Invalid target")
            result.update(target_xyz=xyz,target_timestamp=float(target_timestamp))
        return result
    def build_inference_query(self,sequence_id,query_time):
        return self.build(sequence_id,query_time)

class LiDARUAVDataset(Dataset):
    """GT records create supervised requests; builder remains GT-independent."""
    def __init__(self,root,split_file,split,max_events=20,sequence_limit=None,deterministic_indices=None):
        self.root=Path(root);self.max_events=max_events;self.builder=LiDARQueryBuilder(root,max_events)
        sequences=json.loads(Path(split_file).read_text())[split]
        if sequence_limit is not None:sequences=sequences[:sequence_limit]
        self.records=[]
        for seq in sequences:
            paths=_paths(self.root/seq/"ground_truth")
            indices=range(len(paths)) if deterministic_indices is None else [i for i in deterministic_indices if i<len(paths)]
            for i in indices:
                p=paths[i];self.records.append(dict(sequence_id=seq,query_time=float(p.stem),target_timestamp=float(p.stem),
                    target_path=p,target_valid=True,sample_id=f"{seq}_g{i:06d}"))
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=dict(self.records[index]);r['target_xyz']=np.load(r.pop('target_path'),allow_pickle=False).reshape(3)
        return self.builder.build(**r,gt_source='sequence_ground_truth')

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
                seen.add(key);self.records.append({"sequence_id":sequence,"sample_id":f"{sequence}_valref_{index:06d}","query_time":timestamp,"target_timestamp":timestamp,"target_xyz":position.astype(np.float32),"target_valid":True,"classification":row["Classification"],"csv_row":index+2})
            except Exception as exc:self.invalid_rows.append({"row":index+2,"error":repr(exc),"data":row})
        if self.duplicate_keys: raise ValueError(f"Duplicate validation (sequence,timestamp) keys: {self.duplicate_keys[:10]}")

class LiDARUAVValidationDataset(Dataset):
    """Strict CSV GT source; preserves all valid reference rows including empty input."""
    def __init__(self,root,reference,max_events=20):
        self.root=Path(root);self.max_events=max_events;self.builder=LiDARQueryBuilder(root,max_events)
        self.adapter=ValidationReferenceAdapter(reference)
        if self.adapter.invalid_rows: raise ValueError(f"Invalid validation references: {self.adapter.invalid_rows[:5]}")
        self.records=sorted(self.adapter.records,key=lambda r:(r['sequence_id'],r['query_time']))
    def __len__(self):return len(self.records)
    def __getitem__(self,index):return self.builder.build(**self.records[index],gt_source='validation_ref_csv')

class TemporalQueryClipDataset(Dataset):
    """Right-padded causal rolling prefixes; validation scores only the endpoint once.

    Each item ends at an actual query, includes up to clip_length prior queries,
    never crosses a sequence. Training supervises all valid slots.
    """
    def __init__(self,queries,clip_length=8,stride=1,validation=False):
        if clip_length<1 or stride<1:raise ValueError('Positive clip length and stride required')
        self.queries=queries;self.clip_length=clip_length;self.validation=validation;self.windows=[]
        groups={}
        for i,r in enumerate(queries.records):groups.setdefault(r['sequence_id'],[]).append(i)
        for seq,ids in sorted(groups.items()):
            ids.sort(key=lambda i:queries.records[i]['query_time'])
            times=[queries.records[i]['query_time'] for i in ids]
            if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Queries must increase strictly within sequence')
            for end in range(0,len(ids),1 if validation else stride):
                self.windows.append(ids[max(0,end-clip_length+1):end+1])
    def __len__(self):return len(self.windows)
    def __getitem__(self,index):
        return {'queries':[self.queries[i] for i in self.windows[index]],'clip_length':self.clip_length,
                'score_last_only':self.validation}

def collate_temporal_queries(items):
    """[B,T] queries -> packed points and explicit [B,T] causal metadata.

    Right padding has no points and no target. Missing real observations are
    valid queries and MUST NOT become padding.
    """
    clips=[i['queries'] for i in items];B=len(clips)
    T=max(i.get('clip_length',len(i['queries'])) for i in items)
    samples=[];valid=torch.zeros((B,T),dtype=torch.bool);score=valid.clone()
    for b,clip in enumerate(clips):
        if not clip:raise ValueError('Empty query history')
        if len({q['sequence_id'] for q in clip})!=1:raise ValueError('Cross-sequence clip')
        if any(y['query_time']<=x['query_time'] for x,y in zip(clip,clip[1:])):raise ValueError('Non-increasing query history')
        for t in range(T):
            if t<len(clip):
                q=clip[t];valid[b,t]=True;score[b,t]=not items[b].get('score_last_only',False) or t==len(clip)-1
            else:
                q={'points':torch.empty((0,3)),'sensor_id':torch.empty(0,dtype=torch.long),
                   'delta_t':torch.empty(0),'recent_mask':torch.empty(0,dtype=torch.bool),
                   'query_time':clip[-1]['query_time'],'sequence_id':clip[0]['sequence_id'],
                   'sample_id':'PADDING','event_count':0,'event_timestamps':[],
                   'has_observation':False,'target_valid':False}
            if any(ts>q['query_time'] for ts in q['event_timestamps']) or bool((q['delta_t']>0).any()):
                raise ValueError('Future event violation')
            samples.append(q)
    counts=torch.tensor([len(q['points']) for q in samples],dtype=torch.long)
    batch={k:torch.cat([q[k] for q in samples]) for k in ('points','sensor_id','delta_t','recent_mask')}
    batch.update(num_samples=B*T,point_batch_index=torch.repeat_interleave(torch.arange(B*T),counts),
                 clip_batch_index=torch.arange(B).repeat_interleave(T),clip_position=torch.arange(T).repeat(B),
                 query_valid_mask=valid,score_mask=score,point_counts=counts,
                 query_time=torch.tensor([q['query_time'] for q in samples],dtype=torch.float64),
                 has_observation=torch.tensor([len(q['points'])>0 for q in samples]).reshape(B,T),
                 target_valid=torch.tensor([q.get('target_valid',False) for q in samples],dtype=torch.bool),
                 sample_id=[q['sample_id'] for q in samples],sequence_id=[q['sequence_id'] for q in samples],
                 event_count=torch.tensor([q['event_count'] for q in samples]),
                 event_timestamps=[q['event_timestamps'] for q in samples])
    batch['query_time_clip']=batch['query_time'].reshape(B,T)
    batch['target_valid_clip']=batch['target_valid'].reshape(B,T)&valid
    if bool(batch['target_valid'].any()):
        xyz=torch.stack([torch.as_tensor(q['target_xyz']).float() if q.get('target_valid',False) else torch.zeros(3) for q in samples])
        batch.update(target_xyz=xyz,target_xyz_clip=xyz.reshape(B,T,3),
                     target_timestamp=torch.tensor([q.get('target_timestamp',float('nan')) for q in samples],dtype=torch.float64))
    return batch

def collate_lidar_samples(samples):
    return collate_temporal_queries([{'queries':[q]} for q in samples])

def build_query_history(builder,sequence_id,query_times,clip_length=8):
    return collate_temporal_queries([{'queries':[builder.build_inference_query(sequence_id,t) for t in query_times[-clip_length:]]}])
