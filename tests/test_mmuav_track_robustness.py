import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rdq_uav.mmuav.track_robustness import select_track_v2,stitch_tracks,support_statistics
from rdq_uav.mmuav.observation_path import (select_observation_path,extract_observation_nodes,
    deduplicate_observation_nodes,build_observation_dag,PathConfig,build_bounded_output_path)


class TrackRobustnessTest(unittest.TestCase):
    def row(self,t,x,tid,update=True):
        return dict(timestamp=t,track_id=tid,x=x,y=0.,z=0.,vx=1.,vy=0.,vz=0.,measurement_update=update)

    def test_supported_stitching(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]+[self.row(1.,9.,'a',False)]
        rows += [self.row(t,t,'b') for t in [.3,.4]]
        result=select_track_v2(rows)
        self.assertEqual([r['track_id'] for r in result],['a']*3+['b']*2)
        self.assertTrue(all(r['measurement_update'] for r in result))

    def test_gate_and_empty(self):
        self.assertEqual(select_track_v2([]),[])
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]+[self.row(t,100.,'b') for t in [.3,.4]]
        self.assertEqual(len(select_track_v2(rows)),3)

    def test_no_gt_inputs(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]
        self.assertEqual(len(select_track_v2(rows)),3)
        changed=[dict(r,gt_x=999,gt_distance=-999,sequence_id='arbitrary') for r in rows]
        self.assertEqual(select_track_v2(rows),select_track_v2(changed))

    def test_long_tail_does_not_win(self):
        rows=[self.row(t,t,'old') for t in [0.,.1]]+[self.row(20,20,'old',False)]
        rows += [self.row(t,t,'supported') for t in [5.,5.1,5.2]]
        self.assertEqual({r['track_id'] for r in select_track_v2(rows)},{'supported'})

    def test_large_gap_rejected(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]+[self.row(t,t,'b') for t in [2.,2.1]]
        result=stitch_tracks(rows)
        self.assertFalse(any(d['accepted'] for d in result['stitching_decisions']))
        self.assertEqual(len(result['selected']),3)

    def test_large_distance_rejected(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]+[self.row(t,100,'b') for t in [.3,.4]]
        self.assertTrue(any(d['reason']=='DISTANCE_TOO_LARGE' for d in stitch_tracks(rows)['stitching_decisions']))

    def test_no_fragment_reuse_and_nearest_choice(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]
        rows += [self.row(t,t,'b') for t in [.3,.4]]+[self.row(t,t+1,'c') for t in [.3,.4]]
        result=stitch_tracks(rows);links=[d for d in result['stitching_decisions'] if d['accepted']]
        self.assertEqual(len({d['source_fragment'] for d in links}),len(links))
        self.assertEqual(len({d['target_fragment'] for d in links}),len(links))
        self.assertEqual(links[0]['target_track'],'b')
        ids=[i for c in result['chains'] for i in c['fragment_ids']]
        self.assertEqual(len(ids),len(set(ids)))

    def test_deterministic_and_duplicate_policy(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]+[self.row(.1,99,'a',False)]
        rows += [self.row(t,t,'b') for t in [.3,.4]]
        self.assertEqual(stitch_tracks(rows),stitch_tracks(list(reversed(rows))))
        output=select_track_v2(rows)
        self.assertEqual(len(output),len({r['timestamp'] for r in output}))
        self.assertTrue(all(a['timestamp']<b['timestamp'] for a,b in zip(output,output[1:])))

    def test_statistics(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]+[self.row(1.,1.,'a',False)]
        _,stats=support_statistics(rows);s=stats[0]
        self.assertEqual(s['measurement_count'],3)
        self.assertAlmostEqual(s['prediction_only_tail'],.8)
        self.assertAlmostEqual(s['supported_duration'],.2)


class ObservationPathTest(unittest.TestCase):
    row=TrackRobustnessTest.row
    def test_prediction_tail_invariance(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]
        baseline=select_observation_path(rows)
        for end in [1.,5.,100.]:
            result=select_observation_path(rows+[self.row(end,999,'a',False)])
            for key in ['selected','selected_node_ids','transitions','score']:
                self.assertEqual(baseline[key],result[key])

    def test_overlapping_handoff(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2,.3]]+[self.row(t,t,'b') for t in [.2,.3,.4,.5]]
        result=select_observation_path(rows)
        self.assertEqual(len(result['selected']),6)
        self.assertTrue(any(not e['same_track'] for e in result['transitions']))

    def test_global_beats_nearest_greedy(self):
        a=self.row(0,0,'a');a['vx']=0
        b=self.row(.9,0,'b');b['vx']=-100
        c=self.row(.9,1,'c');c['vx']=10
        d=self.row(1.8,10,'d');d['vx']=10
        e=self.row(2.7,19,'e');e['vx']=10
        result=select_observation_path([a,b,c,d,e])
        self.assertEqual([r['track_id'] for r in result['selected']],['a','c','d','e'])

    def test_observation_gates(self):
        nodes=extract_observation_nodes([self.row(0,0,'a'),self.row(.5,99,'b'),self.row(2,2,'c')])
        graph=build_observation_dag(nodes)
        self.assertEqual(graph['edges'],[])
        self.assertGreater(graph['rejection_counts']['DT_GATE'],0)

    def test_singleton_and_same_timestamp(self):
        rows=[self.row(0,0,'a'),self.row(0,1,'b'),self.row(.1,.1,'c')]
        result=select_observation_path(rows)
        self.assertEqual(len(result['graph']['nodes']),3)
        self.assertEqual(len(result['selected']),2)
        self.assertEqual(len({r['timestamp'] for r in result['selected']}),2)

    def test_identity_freedom_and_determinism(self):
        rows=[self.row(t,t,k) for t,k in [(0,'a'),(.1,'b'),(.2,'c')]]
        a=select_observation_path(rows);b=select_observation_path(rows)
        self.assertEqual([r['track_id'] for r in a['selected']],['a','b','c'])
        for key in ['selected_node_ids','transitions','score']:self.assertEqual(a[key],b[key])

    def test_observation_no_gt(self):
        rows=[self.row(t,t,'a') for t in [0.,.1,.2]]
        poisoned=[dict(r,gt_x=999,gt_y=-999,oracle=True,candidate_gt_error=-999) for r in rows]
        for key in ['selected','transitions','score']:
            self.assertEqual(select_observation_path(rows)[key],select_observation_path(poisoned)[key])

    def test_measurement_dedup(self):
        a=self.row(0,0,'a');b=self.row(.1,.1,'b')
        a['measurement_id']=b['measurement_id']='same'
        nodes,_=deduplicate_observation_nodes(extract_observation_nodes([a,b]))
        self.assertEqual(len(nodes),1)
        c=self.row(0,1e-10,'c')
        nodes,_=deduplicate_observation_nodes(extract_observation_nodes([self.row(0,0,'a'),c]))
        self.assertEqual(len(nodes),1)

    def test_bounded_output(self):
        selected=select_observation_path([self.row(0,0,'a'),self.row(.2,.2,'b')])['selected']
        out=build_bounded_output_path(selected,[0,.1,.2,.25,5.5])
        self.assertEqual([r['state_type'] for r in out],['observed','predicted','observed','predicted','missing'])
