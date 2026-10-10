import importlib.util
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location('audit',str(Path(__file__).with_name('audit_modalities.py')))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
base=dict(sequence_id='seq0001',sample_id='1',query_time=10.,image_time=9.9,
          image_path='missing.png',image_source_wh=[100,100],has_gt2d=True,
          gt_box_xyxy_px=[10.,10.,20.,20.],has_gt3d=True,gt_xyz=[0.,0.,0.],pred_xyz=[3.,4.,0.])
rows=[]
for i,(boxes,category) in enumerate([
    ([[10,10,20,20]],'success'),
    ([[40,40,50,50],[10,10,20,20]],'ranking_failure'),
    ([[40,40,50,50]],'candidate_miss'),
    ([], 'no_output')]):
    row=dict(base,sample_id=str(i),image_path=f'{i}.png',
             pred_boxes_xyxy_px=boxes,pred_box_scores=[.9,.5][:len(boxes)])
    result=a.image_record(row)
    assert result['category']==category
    rows.append(row)
result=a.radar_summary([base,dict(base,pred_xyz=None)])
assert result['rmse_x_m']==3 and result['rmse_y_m']==4 and result['rmse_3d_m']==5
assert result['coverage']==.5 and result['success_1m']==0
assert a.radar_summary([dict(base,pred_xyz=None)])['rmse_3d_m'] is None
same=dict(rows[0],sample_id='duplicate',query_time=10.2)
assert len(a.deduplicate_images([rows[0],same]))==1
coco=a.coco_bbox_metrics(rows)
assert coco['unique_labeled_images']==4 and 0<=coco['AP']<=1
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp)
    for row in rows:
        image=folder/'seq0001'/'Image'/Path(row['image_path']).name
        image.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(200,100)).save(image)
    assert not a.draw_cases(rows,[a.image_record(r) for r in rows],folder/'out',1,folder)
    assert len(list((folder/'out').rglob('*.png')))==8
print('PASS: categories, RMSE, missing outputs, deduplication, COCO and visualizations')
