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

def assert_query_integrity(query, *, require_events=False):
    """Validate sequence identity and causal event selection for one query."""
    sequence=query.get("sequence_id")
    if not isinstance(sequence,str) or not sequence.strip():raise AssertionError("empty_sequence_id")
    query_time=float(query["query_time"])
    if not np.isfinite(query_time):raise AssertionError("invalid_query_timestamp")
    event_times=query.get("event_timestamps",[]);event_sequences=query.get("event_sequence_ids",[])
    if len(event_times)!=len(event_sequences):raise AssertionError("event metadata length mismatch")
    if any(item!=sequence for item in event_sequences):raise AssertionError("event_sequence_mismatch")
    if any(float(item)>query_time for item in event_times):raise AssertionError("future_event")
    if bool((query["delta_t"]>0).any()):raise AssertionError("positive delta_t")
    if require_events and int(query.get("event_count",len(event_times)))!=len(event_times):
        raise AssertionError("event_count mismatch")

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
    def select_events(self,sequence_id,query_time):
        """Return causal metadata, explicitly verifying each event's sequence."""
        if not isinstance(sequence_id,str) or not sequence_id.strip():raise AssertionError('empty_sequence_id')
        if not np.isfinite(query_time):raise AssertionError('invalid_query_timestamp')
        events=select_last_history(self.stream(sequence_id),query_time,self.max_events)
        for event in events:
            if event.sequence_id!=sequence_id:
                raise AssertionError(f'event_sequence_mismatch: query={sequence_id}@{query_time}, event={event}')
            if event.timestamp>query_time:raise AssertionError(f'future_event: query={sequence_id}@{query_time}, event={event}')
        return events
    def build(self, sequence_id, query_time, target_xyz=None, target_timestamp=None,
              target_valid=False, sample_id=None, query_uid=None, **metadata):
        query_time=float(query_time)
        if not np.isfinite(query_time): raise ValueError("Non-finite query_time")
        events=self.select_events(sequence_id,query_time)
        parts=[]; sensors=[]; times=[]; recent=[]
        for i,event in enumerate(events):
            assert event.timestamp<=query_time, "Future event violation"
            points=load_released_xyz(event.file_path)[0].astype(np.float32)
            parts.append(points);sensors.append(np.full(len(points),event.sensor_id,np.int64))
            times.append(np.full(len(points),event.timestamp-query_time,np.float32))
            recent.append(np.full(len(points),i>=max(0,len(events)-4),bool))
        cat=lambda a,shape,dtype: torch.from_numpy(np.concatenate(a) if a else np.empty(shape,dtype))
        result={"points":cat(parts,(0,3),np.float32),"sensor_id":cat(sensors,(0,),np.int64),
                "delta_t":cat(times,(0,),np.float32),"supervision_recent_mask":cat(recent,(0,),bool),
                "sequence_id":sequence_id,"query_time":query_time,"num_samples":1,
                "sample_id":sample_id or f"{sequence_id}_query_{query_time:.9f}",
                "query_uid":query_uid if query_uid is not None else (sample_id or f"query_{query_time:.9f}"),
                "event_count":len(events),"event_timestamps":[e.timestamp for e in events],
                "event_sequence_ids":[e.sequence_id for e in events],
                "target_valid":bool(target_valid),"metadata":metadata}
        if target_valid:
            if target_xyz is None or target_timestamp is None: raise ValueError("Valid target requires XYZ and timestamp")
            xyz=torch.as_tensor(target_xyz,dtype=torch.float32)
            if xyz.shape!=(3,) or not torch.isfinite(xyz).all() or not np.isfinite(target_timestamp):
                raise ValueError("Invalid target")
            result.update(target_xyz=xyz,target_timestamp=float(target_timestamp))
        assert_query_integrity(result,require_events=True)
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
                p=paths[i];self.records.append(dict(sequence_id=seq,query_uid=i,query_time=float(p.stem),target_timestamp=float(p.stem),
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
        ordinal={}
        for record in self.records:
            sequence=record['sequence_id'];record['query_uid']=ordinal.get(sequence,0);ordinal[sequence]=record['query_uid']+1
    def __len__(self):return len(self.records)
    def __getitem__(self,index):return self.builder.build(**self.records[index],gt_source='validation_ref_csv')

def collate_lidar_samples(samples):
    """Pack independent spatial queries into exactly one model sample each."""
    if not samples:raise ValueError('Cannot collate an empty spatial query batch')
    for query in samples:assert_query_integrity(query,require_events=True)
    counts=torch.tensor([len(q['points']) for q in samples],dtype=torch.long);n=len(samples)
    batch={k:torch.cat([q[k] for q in samples]) for k in ('points','sensor_id','delta_t','supervision_recent_mask')}
    batch.update(num_samples=n,
        point_batch_index=torch.repeat_interleave(torch.arange(n),counts),point_counts=counts,
        query_time=torch.tensor([q['query_time'] for q in samples],dtype=torch.float64),
        target_valid=torch.tensor([q.get('target_valid',False) for q in samples],dtype=torch.bool),
        sample_id=[q['sample_id'] for q in samples],sequence_id=[q['sequence_id'] for q in samples],
        query_uid=[q.get('query_uid',q['sample_id']) for q in samples],
        event_count=torch.tensor([q['event_count'] for q in samples]),
        event_timestamps=[q['event_timestamps'] for q in samples],
        event_sequence_ids=[q['event_sequence_ids'] for q in samples])
    xyz=torch.stack([torch.as_tensor(q['target_xyz']).float() if q.get('target_valid',False) else torch.zeros(3) for q in samples])
    timestamps=torch.tensor([q.get('target_timestamp',float('nan')) for q in samples],dtype=torch.float64)
    batch.update(target_xyz=xyz,target_timestamp=timestamps)
    return batch
