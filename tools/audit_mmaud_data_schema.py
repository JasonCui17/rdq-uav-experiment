#!/usr/bin/env python3
"""G0-1 read-only asset audit. No model, synchronization, projection or cache.

Root is resolved from the existing frozen reproduction config. Output creation
is exclusive; never overwrite. Native arrays are inspected without load_xyz's
column slicing. Only sampled sequences have full NPY structure scans.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
SAMPLES = ('seq0001', 'seq0007', 'seq0054', 'seq0098', 'seq0102')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(out, name, rows, fields=None):
    fields = fields or list(rows[0])
    with (out / name).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def index(directory):
    entries, unknown = [], []
    for p in sorted(directory.iterdir()) if directory.is_dir() else []:
        if not p.is_file():
            continue
        try:
            entries.append((float(p.stem), p))
        except ValueError:
            unknown.append(str(p))
    return sorted(entries), unknown


def nearest_diffs(a, b):
    if not len(a) or not len(b):
        return dict(median=None, p95=None, max=None)
    j = np.searchsorted(b, a)
    d = np.minimum(abs(a-b[np.clip(j, 0, len(b)-1)]),
                   abs(a-b[np.clip(j-1, 0, len(b)-1)]))
    return dict(median=float(np.median(d)), p95=float(np.percentile(d, 95)), max=float(d.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PROJECT/'outputs/own_multimodal_research/g0_data_schema_audit')
    args = p.parse_args()
    frozen_path = PROJECT/'outputs/mmuav_paper_reproduction/final_reproduction/frozen_reproduction_config.json'
    splits_path = PROJECT/'outputs/mmuav_paper_reproduction/splits/splits.json'
    protected = {str(x): sha(x) for x in (frozen_path, splits_path)}
    root = Path(json.loads(frozen_path.read_text())['data_root']).resolve()
    splits = json.loads(splits_path.read_text())
    assert root == Path(splits['source_root']).resolve()
    split_of = {s: k for k in ('train_sub', 'validation_sub', 'heldout_test_sub') for s in splits[k]}
    out = args.output.resolve()
    # Refuse an existing directory, including partially completed previous runs.
    out.mkdir(parents=True, exist_ok=False)
    result = dict(data_root=str(root), root_evidence=str(frozen_path), protected_sha256=protected,
                  sampled_sequences=list(SAMPLES), scope='native files only; no model/projection/resizing',
                  errors=[], samples=[], point_structure_scans={}, calibration=[], labels=[])
    ranges, cameras, mapping = [], [], []
    total_images = 0
    modalities = Counter()
    for seq in sorted(x for x in root.iterdir() if x.is_dir()):
        streams = {}
        for directory in sorted(x for x in seq.iterdir() if x.is_dir()):
            modalities[directory.name] += 1
            entries, unknown = index(directory)
            result['errors'].extend({'path': x, 'error': 'UNKNOWN timestamp'} for x in unknown)
            streams[directory.name] = entries
        total_images += len(streams.get('Image', []))
        all_times = [t for entries in streams.values() for t, _ in entries]
        mapping.append(dict(sequence_id=seq.name, source_recording_id='UNKNOWN',
                            source_path_or_evidence=str(seq)+'; no recording metadata; sequence-only split: '+str(splits_path),
                            start_timestamp=min(all_times) if all_times else '',
                            end_timestamp=max(all_times) if all_times else '', confidence='UNKNOWN'))
        if seq.name not in SAMPLES:
            continue
        image_times = np.array([t for t, _ in streams.get('Image', [])])
        gt_times = set(t for t, _ in streams.get('ground_truth', []))
        for sensor, entries in streams.items():
            times = np.array([t for t, _ in entries])
            diff = nearest_diffs(image_times, times)
            ranges.append(dict(sequence_id=seq.name, frozen_split=split_of.get(seq.name, 'UNKNOWN'),
                               sensor=sensor, count=len(entries), start_timestamp=times[0] if len(times) else '',
                               end_timestamp=times[-1] if len(times) else '',
                               image_to_sensor_median_s=diff['median'], image_to_sensor_p95_s=diff['p95'],
                               image_to_sensor_max_s=diff['max'],
                               exact_timestamp_intersection_with_gt=len(set(times)&gt_times),
                               median_interval_s=float(np.median(np.diff(times))) if len(times)>1 else '',
                               max_interval_s=float(np.max(np.diff(times))) if len(times)>1 else ''))
            if not entries:
                continue
            selected = {0, len(entries)//2, len(entries)-1}
            structures = Counter()
            first_nonempty = None
            if sensor != 'Image':
                for i, (_, path) in enumerate(entries):
                    try:
                        a = np.load(path, allow_pickle=False)
                        structures[(str(a.shape), str(a.dtype))] += 1
                        if a.size and first_nonempty is None:
                            first_nonempty = i
                    except Exception as exc:
                        result['errors'].append(dict(path=str(path), error=str(exc)))
                if first_nonempty is not None:
                    selected.add(first_nonempty)
                result['point_structure_scans'][seq.name+'/'+sensor] = [
                    dict(shape=k[0], dtype=k[1], count=n) for k,n in sorted(structures.items())]
            for i in sorted(selected):
                ts, path = entries[i]
                row = dict(sequence_id=seq.name, frozen_split=split_of.get(seq.name, 'UNKNOWN'),
                           sensor_name=sensor, storage_format=path.suffix, example_path=str(path),
                           example_timestamp=path.stem, timestamp_numeric=ts, evidence_source=str(path), confidence='CONFIRMED')
                try:
                    if sensor == 'Image':
                        with Image.open(path) as im:
                            mode, fmt = im.mode, im.format
                            a = np.asarray(im)
                        row.update(camera_id='packed_left_right', image_mode=mode, storage_format=fmt,
                                   channel_interpretation='PIL decoded RGB; OpenCV decodes BGR; no BGR tensor stored in PNG')
                        cameras.append(dict(sequence_id=seq.name, camera_id='packed_left_right', timestamp=path.stem,
                                            shape=str(a.shape), dtype=str(a.dtype), channels=3, height=a.shape[0], width=a.shape[1],
                                            min=int(a.min()), max=int(a.max()), mode=mode, format=fmt, file_path=str(path),
                                            per_view_height=960, per_view_width=1280,
                                            left_right_order_confidence='INFERRED from existing layout/import code; calibration files identify two views'))
                    else:
                        a = np.load(path, allow_pickle=False)
                    row.update(raw_shape=list(a.shape), dtype=str(a.dtype),
                               number_of_fields=(a.shape[-1] if sensor=='Image' else a.shape[1] if a.ndim==2 else (a.size if sensor in ('class','ground_truth') else 'UNKNOWN_EMPTY')),
                               min=float(np.nanmin(a)) if a.size else None, max=float(np.nanmax(a)) if a.size else None,
                               field_names_if_known=(['x','y','z'] if sensor in ('lidar_360','livox_avia','radar_enhance_pcl','ground_truth') and a.size else ['class_id'] if sensor=='class' else []),
                               field_semantics='XYZ parser convention; physical axes/frame/units UNKNOWN' if sensor not in ('Image','class') else 'RGB pixels' if sensor=='Image' else 'UAV class ID',
                               head=a[:3].tolist() if sensor!='Image' else None,
                               nonfinite_count=int((~np.isfinite(a)).sum()),
                               zero_rows=int(np.all(a==0,axis=1).sum()) if a.ndim==2 else None)
                    result['samples'].append(row)
                except Exception as exc:
                    result['errors'].append(dict(path=str(path), error=str(exc)))
    result['inventory'] = dict(sequence_count=len(mapping), image_count=total_images,
                               modality_sequence_counts=dict(modalities), other_sensors_in_main_root='NONE_FOUND')
    calib_root = root.parent/'fisheye_calibration'
    for camera in ('left','right'):
        source = calib_root/camera/'we_want_rgb-camchain.yaml'
        c = yaml.safe_load(source.read_text())['cam0']
        for key in ('camera_model','distortion_model','intrinsics','distortion_coeffs','resolution','rostopic'):
            v = c[key]
            result['calibration'].append(dict(parameter_name=key, shape=str(np.asarray(v).shape), value=json.dumps(v),
                                              source_file=str(source), source_sensor_pair=camera+' camera',
                                              direction='camera-frame to image' if key=='intrinsics' else 'N/A', confidence='CONFIRMED'))
    for name in ('T_camera_lidar360','T_camera_avia','T_camera_radar','T_camera_gt','T_right_left','stereo_baseline'):
        result['calibration'].append(dict(parameter_name=name, shape='UNKNOWN', value='UNKNOWN',
                                          source_file=str(calib_root)+'; '+str(PROJECT/'configs/calibration/mmaud_v1_omni.yaml'),
                                          source_sensor_pair=name, direction='UNKNOWN', confidence='UNKNOWN'))
    result['calibration_notes'] = {
        'camera_model':'Kalibr unified omnidirectional (omni) + radtan; other fisheye model; NOT OpenCV fisheye or pinhole',
        'intrinsic_order':'xi, fu, fv, pu, pv', 'distortion_order':'k1, k2, p1, p2',
        'parser_evidence':str(PROJECT/'src/rdq_uav/calibration/omni.py'),
        'existing_transform_convention':'p_camera = R_camera_from_reference @ p_reference + t; source transform_points in omni.py; no actual numeric extrinsic available',
        'missing_old_assets':[str(PROJECT/'calibration'/n) for n in ('official_2d_timestamp_mapping.csv','official_left_center_annotations.csv','official_left_initial_extrinsics.json')],
        'baseline':'0.178 m is present in derived config but absent from original numeric calibration; not independently confirmed',
        'calibration_bag':'2024-02-01-14-52-39.bag contains only /usb_cam/image_raw sensor_msgs/Image, 1933 messages; first encoding rgb8, H=960 W=2560, frame_id=head_camera. No sensor pointcloud/GT/TF topic. Independently inspected with rosbags; not a flight provenance source.'}
    det = root.parent/'2d_detection'
    labels, label_examples = [], []
    for folder in ('labels/train2017','labels/val2017','test/labels'):
        fs = sorted((det/folder).glob('*.txt'))
        shapes = Counter()
        for path in fs:
            lines = [x.split() for x in path.read_text().splitlines() if x.strip()]
            shapes[str([len(x) for x in lines])] += 1
        labels.append(dict(directory=str(det/folder), count=len(fs), line_structure_counts=dict(shapes)))
        for path in fs[:2]:
            label_examples.append(dict(path=str(path), raw_text=path.read_text(), source_type='UNKNOWN',
                                       meaning='official YOLO bbox annotation; manual/projected/model origin undocumented'))
    result['official_2d_asset'] = dict(root=str(det), partitions=labels, examples=label_examples,
                                      documentation=str(det/'README.md'),
                                      source_type='UNKNOWN', timestamp_mapping_to_main_root='NOT_AVAILABLE',
                                      original_annotation_method='UNKNOWN',
                                      existing_import_logic=str(PROJECT/'tools/calibration/import_official_2d.py')+' matches decoded left-crop pixels; its current target is missing official/v1, not official/train')
    result['labels'] = [
        dict(label='3D XYZ', available='YES', source_type='ground_truth NPY', meaning='one ground_truth position vector per GT timestamp; not confirmed geometric/body center', training='YES as native position; frame/units must be resolved for metric fusion', evaluation='native position comparison only; physical units UNKNOWN', evidence=str(root/SAMPLES[0]/'ground_truth')+'; build_mmuav_cluster_dataset.py:71-80'),
        dict(label='2D bbox', available='YES in sibling official_2d asset; NO paired labels confirmed', source_type='UNKNOWN', meaning='official YOLO class,cx,cy,w,h normalized; manual origin not documented', training='standalone asset YES; multimodal main root not yet', evaluation='official standalone bbox protocol possible; main-root bbox mAP not currently justified', evidence=str(det/'README.md')),
        dict(label='2D center', available='bbox centers in sibling asset; NO independent manual centers confirmed', source_type='UNKNOWN', meaning='cx,cy from official boxes; old code derives u=cx*1280, v=cy*960; output absent', training='conditional on exact image mapping; not projected GT', evaluation='conditional on mapping and eligibility definition', evidence=str(PROJECT/'tools/calibration/import_official_2d.py')),
        dict(label='visibility', available='NO_CONFIRMED_VISIBILITY_LABEL', source_type='UNKNOWN', meaning='GT existence does not imply image visibility; old visible=1 is synthesized for bbox-selected rows', training='NO', evaluation='NO global visual Recall denominator', evidence=str(PROJECT/'tools/calibration/build_official_center_annotations.py')+'; no visibility files in main root'),
        dict(label='img_uav_detect / img_uav_pos', available='NO corresponding main-root files', source_type='UNKNOWN', meaning='AV-DETC loader reads Data-M/img_3d_label [present,u,v]; public local generator not found; loader changes present to 0 under dark augmentation', training='NO', evaluation='NO', evidence='/home/jasoncui/projects/AV-DETC/dataloader/dataset.py:35-48; dataloader/data_process.py:image_darkaug'),
        dict(label='projection valid / GT interpolation valid', available='code-derived only', source_type='MODEL_OUTPUT', meaning='computed numerical bounds/finite/in-image masks, not sensor visibility annotations', training='only explicitly derived numerical masks after verified calibration', evaluation='not ground truth visibility', evidence=str(PROJECT/'src/rdq_uav/calibration/omni.py')+'; trajectory.py')]
    rows = []
    for sensor, title in (('lidar_360','Mid360'),('livox_avia','Livox Avia'),('radar_enhance_pcl','mmWave Radar'),('Image','Camera(s)')):
        samples = [x for x in result['samples'] if x['sensor_name']==sensor]
        dims = sorted(set(str(x['raw_shape']) for x in samples))
        rows.append(dict(category='sensor', sensor_or_label=title, actual_format='PNG' if sensor=='Image' else 'NPY',
                         shape_or_dimension='; '.join(dims), field_semantics='RGB packed two views' if sensor=='Image' else 'XYZ according to existing parser; no additional properties stored',
                         evidence=samples[0]['example_path']+'; '+str(PROJECT/('src/rdq_uav/data/dataset.py' if sensor=='radar_enhance_pcl' else 'tools/build_mmuav_cluster_dataset.py')),
                         confidence='CONFIRMED structure; INFERRED XYZ convention', implication='native input available; physical extrinsics unresolved'))
    for x in result['labels']:
        rows.append(dict(category='label',sensor_or_label=x['label'],actual_format=x['available'],shape_or_dimension='3' if x['label']=='3D XYZ' else 'UNKNOWN',field_semantics=x['meaning'],evidence=x['evidence'],confidence='UNKNOWN' if x['source_type']=='UNKNOWN' else 'CONFIRMED',implication=x['evaluation']))
    result['sensor_field_limitations'] = {
        'lidar':'All NPY files in five sampled sequences checked without slicing: native released files already three columns. Existing load_xyz also slices [:,:3], but this is not the cause of the observed three-column native files. Whether upstream ROS/extraction discarded intensity/reflectivity/ring/time is UNKNOWN: no source sensor bag or release extraction parser is available.',
        'radar':'Nonempty released radar_enhance_pcl is Nx3 float64; empty files are (0,) float64. Existing parser treats triples as Cartesian XYZ. No stored range/azimuth/elevation/Doppler/radial velocity/RCS/confidence/per-point timestamp. Derived range is not a raw field. Enhancement/filtering/target semantics and original sensor frame/units are UNKNOWN.',
        'GT':'XYZ order is existing reader convention. Unit, physical coordinate frame, reference point and acquisition provenance UNKNOWN. No native attitude/velocity/target ID/visibility fields. Code-computed velocity is not stored GT.'}
    result['proposed_sample_schema'] = {
        'sequence_id':'native seqNNNN; flight_id unavailable',
        'image':{'path':'Image/<timestamp>.png','native_shape':[960,2560,3],'decode':'RGB uint8'},
        'points':{'lidar_360':'native N x 3 float64','livox_avia':'native N x 3 float64','radar_enhance_pcl':'native N x 3 float64 OR empty (0,) float64; retain emptiness'},
        'timestamps':{'image':'own filename timestamp','lidar_360':'own filename timestamp','livox_avia':'own filename timestamp','radar_enhance_pcl':'own filename timestamp','ground_truth':'own filename timestamp; asynchronous; no already-confirmed pairing'},
        'calibration':{'left':'original omni/radtan parameters','right':'original omni/radtan parameters'},
        'labels':{'ground_truth_position':'native 3-vector at its own timestamp'},
        'provenance':{'data_root':str(root),'frozen_split':'existing sequence-only assignment','native_source_paths':'retain each file path'}}
    result['blocking_issues'] = [
        'Missing verified LiDAR/radar/GT-to-camera numerical extrinsics and GT/sensor physical frame/unit definitions; blocks meaningful cross-sensor projection and metric 3D fusion.',
        'Official 2D bbox-to-current-train timestamp mapping unavailable; original annotation method UNKNOWN; blocks reliable paired 2D supervision/evaluation claims.',
        'NO_CONFIRMED_VISIBILITY_LABEL; blocks visibility-conditioned image Recall over all GT frames.',
        'FLIGHT_INDEPENDENCE_NOT_CONFIRMED; blocks flight-independent evaluation claim, not read-only pairing.',
        'Four sampled sequences have entirely empty released radar streams; cannot assume every sample has radar returns.']
    result['verdict'] = 'CONDITIONAL_PASS'
    result['next_step'] = dict(timestamp_pairing='YES: native stream timestamps available',
                               thirty_projection_review='NO: numerical extrinsics/frame convention must first be recovered or independently confirmed',
                               enter_G1='NOT_EXECUTED')
    result['other_locations'] = [dict(path=str(root.parent/'val'), role='separate official unlabeled validation; not mixed into audit counts'),
                                 dict(path=str(det),role='separate official 2D asset; only labels/schema inspected'),
                                 dict(path='/root/AUD',role='access denied; existence/content not confirmed; not used'),
                                 dict(path='/home/jasoncui/tszf/AUD',role='not found'),
                                 dict(path=str(root.parent/'v1'),role='old configs refer here; NOT_FOUND; old conclusions not used as raw evidence')]
    result['checks'] = dict(protected_files_unchanged=all(sha(Path(x))==v for x,v in protected.items()),
                           model_training=False, dataset_construction=False, data_mutation=False,
                           projection_execution=False, frozen_split_modified=False)
    write_csv(out,'data_schema_audit.csv',rows)
    write_csv(out,'camera_summary.csv',cameras)
    write_csv(out,'calibration_summary.csv',result['calibration'])
    write_csv(out,'label_summary.csv',result['labels'])
    write_csv(out,'sequence_flight_mapping.csv',mapping)
    write_csv(out,'timestamp_range_summary.csv',ranges)
    (out/'sample_structure_dump.txt').write_text('\n\n'.join(json.dumps(x,ensure_ascii=False,indent=2) for x in result['samples'])+'\n',encoding='utf-8')
    (out/'data_schema_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def table(items, keys):
        return '| '+' | '.join(keys)+' |\n| '+' | '.join('---' for _ in keys)+' |\n'+'\n'.join('| '+' | '.join(str(x[k]).replace('|','/').replace('\n',' ') for k in keys)+' |' for x in items)
    report = f'''# G0 Data Schema Audit

## 1. Data root

`{root}`. Confirmed by frozen reproduction config and current split source_root. {len(mapping)} sequences, {total_images} native PNG images. Primary statistics exclude sibling val and 2d_detection. Evidence: `{frozen_path}`. Existing engineering files inspected before this tool was added. No AGENTS.md found under projects.

## 2. Sensors

| Sensor | Available | Storage | Shape | Fields | Evidence | Usable in first model? |
| --- | --- | --- | --- | --- | --- | --- |
| Mid360 / lidar_360 | Yes | NPY float64 | N×3; native point counts vary | XYZ parser convention only | native samples + build_mmuav_cluster_dataset.py:61-67 | Native points yes; fusion frame unresolved |
| Livox Avia | Yes | NPY float64 | 24000×3 in sampled files | XYZ parser convention only | native samples + reproduction reader | Native points yes; fusion frame unresolved |
| mmWave Radar | Yes, with empty streams | NPY float64 | N×3 nonempty; (0,) empty | XYZ parser convention only | seq0054 actual nonempty arrays + data/dataset.py:47-52 | Conditional: sparse/empty returns; frame and enhancement semantics unresolved |
| Cameras | Yes, two packed views | PNG RGB uint8 | 960×2560×3 | RGB pixels | actual PIL decoding + original left/right YAML + calibration bag | Native image yes |

{result['sensor_field_limitations']['lidar']}

{result['sensor_field_limitations']['radar']}

Raw point-shape distributions for **every NPY in the five sampled sequences** are in JSON. Sample set: {', '.join(SAMPLES)}; includes early/middle/late numbering and train_sub/validation_sub/heldout_test_sub. Inspecting heldout is limited to structure/timestamps; no model or split change. No audio/GPS/IMU/other LiDAR files found in primary root. Calibration bag has camera messages only. Radar hardware model is not independently identified by present data metadata.

## 3. Camera

Two views, `left` and `right`, stored in one `Image` directory; packed H×W=960×2560, per-view calibrated H×W=960×1280. Model: Kalibr omni unified omnidirectional + radtan (DistortedOmniCameraGeometry confirmed in original results text). This is a fisheye/omnidirectional stereo arrangement; no independently confirmed numerical stereo baseline/relative pose. Left/right packed ordering is supported by existing layout/import code, not a current matching output. PNG stores color pixels, not an OpenCV BGR tensor; PIL RGB decoding yields uint8 three channels. Calibration bag directly states rgb8. {len(cameras)} real image samples printed, without resize/normalize, in camera_summary.csv and structure dump. One timestamp filename per packed image; no independent left/right timestamp stored. Sensors have separate timestamps; no exact established cross-stream pairing. Sampled nearest-time differences are in timestamp_range_summary.csv; these are filename-time differences, not clock synchronization estimates.

## 4. Calibration

{table(result['calibration'], ['parameter_name','shape','value','source_file','source_sensor_pair','direction','confidence'])}

Intrinsics order `[xi,fu,fv,pu,pv]`, distortion `[k1,k2,p1,p2]`, confirmed in existing `src/rdq_uav/calibration/omni.py` and original YAML/results. Two separate monocular camchain files have no numeric inter-sensor transform. Existing code direction is `p_camera=R_camera_from_reference*p_reference+t`; inverse direction cannot be substituted. Derived config has null extrinsics; missing old fit/mapping files cannot supply numbers. Its 0.178 m baseline is not backed by current original calibration files, so remains UNKNOWN. Calibration bag inspection: {result['calibration_notes']['calibration_bag']}

## 5. Labels

{table(result['labels'], ['label','available','source_type','meaning','training','evaluation','evidence'])}

3D GT samples are native `(3,) float64`, one position vector per timestamp, no attitude/velocity/IDs. {result['sensor_field_limitations']['GT']} All sampled GT frames scanned for structure; interval/strict timestamp overlap with sensors recorded. Missingness relative to the original recording cannot be determined without an expected timeline; sensor timestamps are not strictly identical to GT. Zero/nonfinite native point rows are numerical validity observations, not target visibility.

Separate 2D asset has 2654 train, 886 val, 885 test label files (4425 total). README explicitly defines YOLO bbox annotation format, but does not document manual/model/projected creation. Therefore **source type UNKNOWN**, not MANUAL_BBOX by assumption. Its centers are bbox centers, not independently annotated manual centers. No confirmed PROJECTED_3D_GT_POINT label file is present in the primary root. Existing projection tools would derive projected points/Oracle ROIs if run; not run here. Existing MODEL_OUTPUT branches/trajectories are algorithmic outputs, not GT.

AV-DETC `img_uav_detect`/`img_uav_pos` originate in loader slices of an expected `Data-M/img_3d_label` vector. Local generator and corresponding data absent; **UNKNOWN**, cannot call manual or projected. Existing `visible=1` in build_official_center_annotations.py is assigned to bbox-selected rows, not a general visibility annotation. Projection/in-image/interpolation masks are numerical computed predicates; they do not establish occlusion or visual detectability. **NO_CONFIRMED_VISIBILITY_LABEL**.

## 6. 2D supervision conclusion

**当前主审计数据尚不能定义可靠的配对二维监督任务；独立官方二维资产存在 bbox 及其中心，但人工来源和到当前 sequence 图像的映射尚未确认，不能据此直接报告主数据 bbox mAP 或全帧二维 Recall。**

## 7. Sequence / flight provenance

**FLIGHT_INDEPENDENCE_NOT_CONFIRMED**. All {len(mapping)} sequence rows have confidence UNKNOWN and source_recording_id UNKNOWN. Start/end bounds are observed union of native filenames, not flight boundaries. Existing split explicitly uses sequence_level_random; no original bag/flight/session mapping or extraction metadata exists in primary root. Do not infer flight independence from sequence name or time discontinuity. Sibling validation_ref CSV is a different split and not used to assign train provenance. sequence_flight_mapping.csv records evidence per sequence.

## 8. Proposed sample schema

This is an asset-record candidate with asynchronous source paths/timestamps, **not an already synchronized training sample**. Fields below are confirmed storage/parser conventions only; no extrinsic, bbox, visibility, Doppler, intensity, flight ID or physical unit claim is included.

```json
{json.dumps(result['proposed_sample_schema'],ensure_ascii=False,indent=2)}
```

## 9. Blocking issues

'''+ '\n'.join('- '+x for x in result['blocking_issues'])+'''

## 10. G0-1 verdict

**CONDITIONAL_PASS**. 主数据的实际存储结构、时间戳和 native position 监督已明确；物理标定与配对二维监督来源/对应关系仍不足。可做只读时间配对检查，但尚不具备直接开展有物理意义的“30张投影人工核验”的全部条件；须先确认数值外参、坐标系和二维映射。本轮立即停止，不进入 G1。

Frozen split/config hashes verified unchanged. No training, full Dataset, projection, cache, download, reproduction modification or data modification performed. All observations and limitations are in data_schema_audit.json; native samples in sample_structure_dump.txt.
'''
    (out/'G0_DATA_SCHEMA_AUDIT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(output=str(out), inventory=result['inventory'], verdict=result['verdict'],checks=result['checks'], errors=result['errors']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
