"""EQS-v1 metadata-only sampling, coverage, resume, and loader regressions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2 import (EpochCyclicQuerySampler,TemporalQueryClipDataset,
    collate_temporal_queries,planned_epoch_stats)


def query(sequence,index):
    time=float(index)+10.
    return dict(sequence_id=sequence,sample_id=f'{sequence}_{index}',query_time=time,
        points=torch.tensor([[index+.1,0.,0.]]),sensor_id=torch.zeros(1,dtype=torch.long),
        delta_t=torch.tensor([-.01]),recent_mask=torch.ones(1,dtype=torch.bool),
        event_count=1,event_timestamps=[time-.01],event_sequence_ids=[sequence],
        has_observation=True,target_valid=True,target_timestamp=time,target_xyz=torch.zeros(3))


class Queries:
    def __init__(self,lengths=(11,7)):
        self.samples=[query(sequence,i) for sequence,n in zip(('A','B'),lengths) for i in range(n)]
        self.records=[{k:q[k] for k in ('sequence_id','sample_id','query_time')} for q in self.samples]
    def __getitem__(self,index):return self.samples[index]


class EQSTests(unittest.TestCase):
    def setUp(self):self.dataset=TemporalQueryClipDataset(Queries(),clip_length=8,stride=1)

    def test_full_dataset_and_sequence_local_ordinals(self):
        self.assertEqual(len(self.dataset),18)
        by_sequence={}
        for record in self.dataset.clip_metadata:by_sequence.setdefault(record['sequence_id'],[]).append(record['anchor_query_ordinal'])
        self.assertEqual(by_sequence,{'A':list(range(11)),'B':list(range(7))})

    def test_stride_one_exact_equivalence(self):
        sampler=EpochCyclicQuerySampler(self.dataset,1,42,True)
        self.assertEqual(set(sampler),set(range(len(self.dataset))))
        self.assertEqual(len(sampler),len(self.dataset))

    def test_stride_1248_selection_rule(self):
        for stride in (1,2,4,8):
            sampler=EpochCyclicQuerySampler(self.dataset,stride,42,False)
            for epoch in range(1,stride+1):
                sampler.set_epoch(epoch);offset=epoch-1
                expected={r['dataset_index'] for r in self.dataset.clip_metadata if r['anchor_query_ordinal']%stride==offset}
                self.assertEqual(set(sampler),expected)

    def test_epoch_cycle_and_resume_order(self):
        sampler=EpochCyclicQuerySampler(self.dataset,4,42,True)
        sequences=[]
        for epoch in range(1,18):sampler.set_epoch(epoch);sequences.append(list(sampler))
        self.assertNotEqual(set(sequences[0]),set(sequences[1]));self.assertEqual(set(sequences[0]),set(sequences[4]))
        resumed=EpochCyclicQuerySampler(self.dataset,4,42,True);resumed.set_epoch(17)
        self.assertEqual(sequences[16],list(resumed))

    def test_four_epoch_anchor_and_query_coverage(self):
        sampler=EpochCyclicQuerySampler(self.dataset,4,42,False);chosen=[];queries=set()
        for epoch in range(1,5):
            selected=sampler.selected_indices(epoch,False);chosen.extend(selected)
            for index in selected:queries.update(self.dataset.clip_metadata[index]['query_indices'])
        self.assertEqual(len(chosen),len(set(chosen)));self.assertEqual(set(chosen),set(range(len(self.dataset))))
        self.assertEqual(queries,set(range(len(self.dataset.queries.records))))

    def test_dense_clip_context_is_unchanged(self):
        sampler=EpochCyclicQuerySampler(self.dataset,4,42,False);sampler.set_epoch(1)
        for index in sampler:
            window=self.dataset.windows[index]
            self.assertEqual(window,list(range(window[0],window[-1]+1)))
            self.assertLessEqual(len(window),8)

    def test_loader_batch_counts_clips(self):
        sampler=EpochCyclicQuerySampler(self.dataset,4,42,False);sampler.set_epoch(1)
        loader=DataLoader(self.dataset,batch_size=2,sampler=sampler,shuffle=False,collate_fn=collate_temporal_queries)
        batch=next(iter(loader));self.assertEqual(batch['query_valid_mask'].shape[0],2)
        self.assertEqual(batch['query_valid_mask'].shape[1],8)

    def test_scheduler_plan_uses_variable_epoch_lengths(self):
        sampler=EpochCyclicQuerySampler(self.dataset,4,42,False)
        plan=planned_epoch_stats(sampler,5,batch_size=2,accumulate=2)
        self.assertEqual([r['offset'] for r in plan],[0,1,2,3,0])
        for row in plan:
            clips=len(sampler.selected_indices(row['epoch'],False))
            self.assertEqual(row['selected_clips'],clips)
            self.assertEqual(row['optimizer_updates'],(row['batches']+1)//2)

    def test_short_sequence_may_have_zero_for_offset(self):
        dataset=TemporalQueryClipDataset(Queries((2,1)),8,1)
        sampler=EpochCyclicQuerySampler(dataset,4,42,False);sampler.set_epoch(4)
        self.assertEqual(len(sampler),0)

    def test_invalid_stride_and_epoch(self):
        for stride in (0,-1):
            with self.assertRaises(ValueError):EpochCyclicQuerySampler(self.dataset,stride)
        sampler=EpochCyclicQuerySampler(self.dataset,4)
        with self.assertRaises(ValueError):sampler.set_epoch(0)


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
