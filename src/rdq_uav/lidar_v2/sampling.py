"""Epoch-cyclic clip sampling and update-plan math; never loads point clouds."""
from __future__ import annotations

import math
import torch
from torch.utils.data import Sampler


class EpochCyclicQuerySampler(Sampler[int]):
    """Select complete dense clips by sequence-local anchor ordinal.

    Epoch numbering is one-based: offset=(epoch-1)%stride. Selection never
    changes a clip's internal query indices or any query's LiDAR history.
    """
    def __init__(self,dataset,stride=4,seed=42,shuffle_selected=True):
        if not isinstance(stride,int) or stride<1:raise ValueError('stride must be a positive integer')
        if not hasattr(dataset,'clip_metadata'):raise TypeError('dataset must expose clip_metadata')
        self.dataset=dataset;self.stride=stride;self.seed=int(seed);self.shuffle_selected=bool(shuffle_selected);self.epoch=1
        if len(dataset.clip_metadata)!=len(dataset):raise AssertionError('clip_metadata length mismatch')
        for i,record in enumerate(dataset.clip_metadata):
            if record['dataset_index']!=i:raise AssertionError('clip metadata index mismatch')

    def set_epoch(self,epoch):
        if not isinstance(epoch,int) or epoch<1:raise ValueError('epoch must use one-based positive numbering')
        self.epoch=epoch

    def offset_for_epoch(self,epoch=None):
        epoch=self.epoch if epoch is None else epoch
        if epoch<1:raise ValueError('epoch must be positive')
        return (epoch-1)%self.stride

    def selected_indices(self,epoch=None,shuffle=None):
        epoch=self.epoch if epoch is None else int(epoch);offset=self.offset_for_epoch(epoch)
        indices=[r['dataset_index'] for r in self.dataset.clip_metadata if r['anchor_query_ordinal']%self.stride==offset]
        do_shuffle=self.shuffle_selected if shuffle is None else bool(shuffle)
        if do_shuffle and len(indices)>1:
            generator=torch.Generator().manual_seed(self.seed+epoch)
            order=torch.randperm(len(indices),generator=generator).tolist()
            indices=[indices[i] for i in order]
        return indices

    def __iter__(self):return iter(self.selected_indices())
    def __len__(self):return len(self.selected_indices(shuffle=False))


class OverlapAwareBatchSampler(Sampler[list[int]]):
    """Group nearby selected anchors within each sequence, then shuffle groups.

    The wrapped EQS selected index set is preserved exactly. Only incomplete
    same-sequence groups enter a cross-sequence leftover pool; those clips remain
    independent samples and can never deduplicate because query keys include the
    sequence identity.
    """
    def __init__(self,sampler,batch_size,seed=42,shuffle_batches=True,drop_last=False):
        if batch_size<1:raise ValueError('batch_size must be positive')
        if drop_last:raise ValueError('UQP-v1 does not drop selected clips')
        self.sampler=sampler;self.dataset=sampler.dataset;self.batch_size=int(batch_size)
        self.seed=int(seed);self.shuffle_batches=bool(shuffle_batches);self.epoch=1

    def set_epoch(self,epoch):self.epoch=int(epoch);self.sampler.set_epoch(self.epoch)

    def batches_for_epoch(self,epoch=None,shuffle=None):
        epoch=self.epoch if epoch is None else int(epoch)
        selected=self.sampler.selected_indices(epoch,shuffle=False);groups={}
        for index in selected:
            record=self.dataset.clip_metadata[index]
            groups.setdefault(record['sequence_id'],[]).append(index)
        batches=[];leftovers=[]
        for sequence in sorted(groups):
            indices=sorted(groups[sequence],key=lambda i:self.dataset.clip_metadata[i]['anchor_query_ordinal'])
            complete=(len(indices)//self.batch_size)*self.batch_size
            batches.extend(indices[i:i+self.batch_size] for i in range(0,complete,self.batch_size))
            leftovers.extend(indices[complete:])
        batches.extend(leftovers[i:i+self.batch_size] for i in range(0,len(leftovers),self.batch_size))
        flat=[i for batch in batches for i in batch]
        if len(flat)!=len(selected) or set(flat)!=set(selected) or len(flat)!=len(set(flat)):
            raise AssertionError('Overlap-aware grouping changed the EQS selected set')
        do_shuffle=self.shuffle_batches if shuffle is None else bool(shuffle)
        if do_shuffle and len(batches)>1:
            generator=torch.Generator().manual_seed(self.seed+epoch)
            order=torch.randperm(len(batches),generator=generator).tolist()
            batches=[batches[i] for i in order]
        return batches

    def __iter__(self):return iter(self.batches_for_epoch())
    def __len__(self):return len(self.batches_for_epoch(shuffle=False))


def planned_epoch_stats(sampler,epochs,batch_size,accumulate):
    """Dry-run clip/batch/update counts for absolute epochs 1..epochs."""
    if epochs<1 or batch_size<1 or accumulate<1:raise ValueError('positive plan arguments required')
    result=[]
    for epoch in range(1,epochs+1):
        clips=len(sampler.selected_indices(epoch,shuffle=False))
        batches=math.ceil(clips/batch_size)
        result.append(dict(epoch=epoch,offset=sampler.offset_for_epoch(epoch),selected_clips=clips,
                           batches=batches,optimizer_updates=math.ceil(batches/accumulate)))
    return result
