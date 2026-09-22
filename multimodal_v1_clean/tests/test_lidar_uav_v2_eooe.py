"""EOOE slot topology, density, invariance, and integration regressions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rdq_uav.lidar_v2.geometry import SparseLevel
from rdq_uav.lidar_v2.model import SparseMerge
from rdq_uav.lidar_v2.sbe import subvoxel_coordinates


def level(coords,point_count=None,batch=None,scale=1.):
    if isinstance(coords,list) and coords and torch.is_tensor(coords[0]):coords=torch.stack(coords)
    coords=torch.as_tensor(coords,dtype=torch.long).reshape(-1,3);n=len(coords)
    if point_count is None:point_count=torch.ones(n,dtype=torch.long)
    else:point_count=torch.as_tensor(point_count,dtype=torch.long)
    if batch is None:batch=torch.zeros(n,dtype=torch.long)
    else:batch=torch.as_tensor(batch,dtype=torch.long)
    return SparseLevel(coords,batch,(coords.float()+.5)*scale,point_count)


def hierarchy(patterns,raw_counts=None,dim=8):
    """Synthetic parents with ordered child patterns and deterministic features."""
    parents=[];children=[];maps=[];child_points=[]
    for parent_id,slots in enumerate(patterns):
        parent=torch.tensor([parent_id*4,0,0]);parents.append(parent)
        total=raw_counts[parent_id] if raw_counts else len(slots)
        counts=[1]*len(slots);counts[-1]+=total-len(slots)
        for slot,count in zip(slots,counts):
            local=torch.tensor([(slot>>2)&1,(slot>>1)&1,slot&1])
            children.append(2*parent+local);maps.append(parent_id);child_points.append(count)
    child=level(children,child_points);parent=level(parents,raw_counts or [len(x) for x in patterns],scale=2.)
    x=torch.arange(len(children)*dim,dtype=torch.float32).reshape(len(children),dim)/100
    return x,child,parent,torch.tensor(maps,dtype=torch.long)


class EOOETests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);self.merge=SparseMerge(8)

    def test_slot_mapping_matches_sbe(self):
        x,child,parent,mapping=hierarchy([list(range(8))])
        local,slot,occupancy=self.merge.octant_occupancy(child,parent,mapping)
        self.assertEqual(slot.tolist(),list(range(8)))
        self.assertTrue(torch.equal(occupancy,torch.ones(1,8)))
        q=local.float()*.5-.25;sbe_slot,_=subvoxel_coordinates(q)
        self.assertTrue(torch.equal(slot,sbe_slot))

    def test_occupancy_patterns_same_ratio_are_distinct(self):
        x,child,parent,mapping=hierarchy([[0,1,2,3],[0,2,4,6]])
        _,_,occupancy=self.merge.octant_occupancy(child,parent,mapping)
        self.assertEqual(occupancy.tolist(),[[1,1,1,1,0,0,0,0],[1,0,1,0,1,0,1,0]])
        self.assertTrue(torch.equal(occupancy.sum(1),torch.tensor([4.,4.])))

    def test_density_is_independent_from_topology(self):
        x,child,parent,mapping=hierarchy([[0,1],[0,1]],[4,80])
        parent_input,occupancy,_=self.merge.parent_input(x,child,parent,mapping)
        self.assertTrue(torch.equal(occupancy[0],occupancy[1]))
        torch.testing.assert_close(parent_input[:,-9],torch.log1p(torch.tensor([4.,80.])))
        self.assertNotEqual(float(parent_input[0,-9]),float(parent_input[1,-9]))

    def test_occupancy_count_consistency_and_invalid_local(self):
        x,child,parent,mapping=hierarchy([[0,3,4],[1,2,5,7]])
        _,_,occupancy=self.merge.octant_occupancy(child,parent,mapping)
        self.assertEqual(occupancy.sum(1).tolist(),[3.,4.])
        bad=level(child.coords.clone());bad.coords[0]=torch.tensor([-1,0,0])
        with self.assertRaisesRegex(AssertionError,'Invalid child octant'):
            self.merge.octant_occupancy(bad,parent,mapping)

    def test_permutation_invariance(self):
        x,child,parent,mapping=hierarchy([[0,1,3,6],[0,2,5,7]],[20,30])
        base_input,base_occ,_=self.merge.parent_input(x,child,parent,mapping);base=self.merge(x,child,parent,mapping)
        order=torch.tensor([7,1,5,0,6,2,4,3]);permuted=SparseLevel(child.coords[order],child.batch_index[order],child.centers[order],child.point_count[order])
        other_input,other_occ,_=self.merge.parent_input(x[order],permuted,parent,mapping[order]);other=self.merge(x[order],permuted,parent,mapping[order])
        self.assertTrue(torch.equal(base_occ,other_occ))
        self.assertLessEqual(float((base_input-other_input).abs().max()),1e-6)
        self.assertLessEqual(float((base-other).abs().max()),1e-6)

    def test_topology_reaches_parent_projection(self):
        x,child,parent,mapping=hierarchy([[0,1,2,3],[0,2,4,6]],[16,16])
        with torch.no_grad():
            self.merge.child[0].weight.zero_();self.merge.child[0].bias.zero_()
        parent_input,occupancy,_=self.merge.parent_input(torch.ones_like(x),child,parent,mapping)
        torch.testing.assert_close(parent_input[0,:-8],parent_input[1,:-8])
        self.assertFalse(torch.equal(occupancy[0],occupancy[1]))
        self.assertFalse(torch.equal(parent_input[0],parent_input[1]))

    def test_projection_contract(self):
        x,child,parent,mapping=hierarchy([[0,1,2,3]])
        parent_input,occupancy,_=self.merge.parent_input(x,child,parent,mapping)
        self.assertEqual(tuple(parent_input.shape),(1,25)) # 2*D + density + occupancy8
        self.assertEqual(tuple(occupancy.shape),(1,8))
        self.assertEqual(self.merge.parent.in_features,25)
        self.assertTrue(torch.isfinite(self.merge(x,child,parent,mapping)).all())


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
