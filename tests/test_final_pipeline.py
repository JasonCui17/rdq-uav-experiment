import json
from pathlib import Path
import sys
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'tools')]
from score_outputs import score,labeled_predictions


def test_realtime_scoring_never_interpolates_from_future(tmp_path):
    data=tmp_path/'data';gt=data/'seq0001/ground_truth';gt.mkdir(parents=True);np.save(gt/'0.1.npy',np.array([.1,0,0]))
    tracks=tmp_path/'tracks';tracks.mkdir();images=tmp_path/'images';images.mkdir()
    rows=[{'query_time':0.,'xyz':[0,0,0],'velocity_mps':[1,0,0],'lagged_xyz':[0,0,0],'event':'high_update'}, {'query_time':.2,'xyz':[20,0,0],'velocity_mps':[1,0,0],'lagged_xyz':[20,0,0],'event':'high_update'}]
    (tracks/'per_query.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for name in ['track_crop_causal','track_crop_lag05']:(images/(name+'.jsonl')).write_text('')
    result=score(data,'seq0001',tracks,images,'sensor')
    assert result['radar']['realtime']['rmse_3d_m']==0
    assert result['radar']['lag05']['rmse_3d_m']>9
    assert result['radar']['realtime']['GT_timestamps']==1


def test_scoring_reads_only_dataset_labels_and_missing_is_not_negative(tmp_path):
    data=tmp_path/'data';(data/'seq0001/ground_truth').mkdir(parents=True)
    tracks=tmp_path/'tracks';tracks.mkdir();(tracks/'per_query.jsonl').write_text('')
    images=tmp_path/'images';images.mkdir();key='seq0001/Image/1.png'
    row={'image_path':key,'sequence_id':'seq0001','has_gt2d':True,'pred_boxes_xyxy_px':[[1,1,10,10]],'pred_box_scores':[.99]}
    for name in ['track_crop_causal','track_crop_lag05']:(images/(name+'.jsonl')).write_text(json.dumps(row)+'\n')
    result=score(data,'seq0001',tracks,images,'sensor')
    assert result['image']['realtime']['unlabeled_images_not_scored']==1
    assert result['image']['realtime']['unique_labeled_images']==0
    label=data/'seq0001/2d_detect/1.txt';label.parent.mkdir();label.write_text('0 .5 .5 .25 .25\n')
    labeled,missing=labeled_predictions(data,'seq0001',images,'track_crop_causal')
    assert missing==0 and labeled[0]['gt_box_xyxy_px']==[480,360,800,600]
    label.write_text('0 .25 .5 .25 .25\n')
    assert labeled_predictions(data,'seq0001',images,'track_crop_causal')[0][0]['gt_box_xyxy_px']==[160,360,480,600]


def test_training_reads_current_dataset_labels_and_keeps_unlabeled_image(tmp_path):
    from tests.test_v2_data import release,dataset
    release(tmp_path,'seq0001',images=(10.,));ds=dataset(tmp_path)
    row=ds[0]
    assert row['m_V'] and not row['gt_2d_valid'] and row['target_valid']
    label=tmp_path/'seq0001/2d_detect/10.0.txt';label.write_text('0 .5 .5 .25 .25\n')
    assert ds[0]['gt_2d_valid'] and ds[0]['gt_box_xyxy_px'].tolist()==[6,3,10,5]
    label.write_text('0 .25 .5 .25 .25\n')
    assert ds[0]['gt_box_xyxy_px'].tolist()==[2,3,6,5]
