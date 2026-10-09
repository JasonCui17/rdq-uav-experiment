import copy
import math

import pytest

from rdq_uav.multimodal_v2.paper_metrics import (
    assign_conditions, coco_bbox_metrics, deduplicate_images, localization_metrics,
    paper_report, write_paper_csv,
)


def row(sample='a', pred=None, time=10.):
    return dict(sample_id=sample, sequence_id='seq0001', query_time=time,
                has_gt3d=True, gt_xyz=[0., 0., 0.], pred_xyz=pred,
                has_gt2d=False, image_path=None, image_time=None)


def image_row(sample='a', time=10., predictions=True):
    r = row(sample, [0., 0., 0.], time)
    r.update(has_gt2d=True, image_path='seq0001/Image/10.png', image_time=10.,
             image_source_wh=[100, 100], gt_box_xyxy_px=[10., 10., 20., 20.],
             pred_boxes_xyxy_px=[[10., 10., 20., 20.]] if predictions else [],
             pred_box_scores=[.9] if predictions else [])
    return r


def test_rmse_axis_identity_and_missing_denominators():
    m = localization_metrics([row('a', [0., 0., 0.]), row('b', [2., 3., 6.]), row('c')])
    assert m['coverage'] == 2/3 and m['missing_queries'] == 1
    assert m['rmse_denominator'] == 2 and m['success_1m'] == 1/3
    assert m['rmse_x_m'] == pytest.approx(math.sqrt(2))
    assert m['rmse_y_m'] == pytest.approx(math.sqrt(4.5))
    assert m['rmse_z_m'] == pytest.approx(math.sqrt(18))
    assert m['rmse_3d_m'] == pytest.approx(math.sqrt(24.5))
    assert m['rmse_3d_m']**2 == pytest.approx(sum(m[f'rmse_{a}_m']**2 for a in 'xyz'))
    assert m['mean_error_m'] == 3.5  # Euclidean mean is not RMSE.


def test_no_outputs_and_nonfinite_predictions():
    assert localization_metrics([row()])['rmse_3d_m'] is None
    assert localization_metrics([row()])['success_1m'] == 0
    assert localization_metrics([])['coverage'] is None
    with pytest.raises(ValueError):
        localization_metrics([row(pred=[float('nan'), 0, 0])])


def test_pooled_rmse_is_not_macro_day_night_mean():
    rows = [row('day', [1., 0., 0.])] + [row(str(i), [3., 0., 0.]) for i in range(3)]
    mapping = {'samples': {'day': 'day', **{str(i): 'night' for i in range(3)}}}
    r = paper_report(rows, 'B0', mapping, compute_coco=False)
    assert r['localization']['rmse_3d_m'] == pytest.approx(math.sqrt(7))
    assert r['day_night']['macro_mean_day_night_rmse_m'] == 2
    assert paper_report(rows, 'B0', compute_coco=False)['day_night']['status'] == 'disabled_no_labels'
    with pytest.raises(ValueError, match='missing'):
        assign_conditions(rows, {'samples': {'day': 'day'}})
    with pytest.raises(ValueError):
        assign_conditions(rows, {'sequences': {'seq0001': 'unknown'}})


def test_dedup_is_order_independent_and_not_oracle_selection():
    good = image_row('good', 10.2)
    bad = image_row('bad', 10.1, False)
    assert deduplicate_images([good, bad]) == deduplicate_images([bad, good]) == [bad]
    other = copy.deepcopy(good); other['sequence_id'] = 'seq0002'
    assert len(deduplicate_images([good, other])) == 2
    conflict = copy.deepcopy(good); conflict['gt_box_xyxy_px'][0] = 9.
    with pytest.raises(ValueError, match='inconsistent'):
        deduplicate_images([good, conflict])


def test_official_coco_perfect_empty_and_duplicate_images(tmp_path):
    pytest.importorskip('pycocotools')
    r = coco_bbox_metrics([image_row('a'), image_row('b', 10.1)], tmp_path)
    assert r['labeled_queries'] == 2 and r['unique_labeled_images'] == 1
    assert r['AP'] == pytest.approx(1.) and r['AP50'] == pytest.approx(1.)
    assert (tmp_path/'coco_selected_queries.json').exists()
    missed = coco_bbox_metrics([image_row(predictions=False)])
    assert missed['AP'] == 0. and missed['AR100'] == 0.
    # A high-score false box before the true box lowers AP; Top1 IoU alone cannot compute AP.
    bad = image_row(); bad['pred_boxes_xyxy_px'].insert(0, [50., 50., 60., 60.])
    bad['pred_box_scores'].insert(0, .99)
    assert 0 < coco_bbox_metrics([bad])['AP50'] < 1.


def test_b1_no_meter_rmse_and_strict_json_csv(tmp_path):
    import json
    report = paper_report([image_row()], 'B1')
    report['split'] = 'heldout_test_sub'
    assert report['localization'] is None
    json.dumps(report, allow_nan=False)
    write_paper_csv(report, tmp_path/'table.csv', method='DINO', modality='RGB')
    text = (tmp_path/'table.csv').read_text()
    assert 'AP50' in text and 'Bandwidth' in text
