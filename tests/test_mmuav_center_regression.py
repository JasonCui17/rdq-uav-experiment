import importlib.util
import sys
from pathlib import Path
import numpy as np
import unittest
from contextlib import contextmanager


@contextmanager
def assert_raises(exception):
    with unittest.TestCase().assertRaises(exception):
        yield
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
sys.path.insert(0,str(ROOT / "tools"))
from rdq_uav.mmuav.center_regressor import CenterRegressor,sample_local,regression_metrics,verify_evaluation_ids
from build_mmuav_center_regression_dataset import logits_only,freeze_config,preprocess


def test_tuple_logits():
    logits=torch.tensor([[0.,1.]])
    assert logits_only((logits,torch.ones(1,20))) is logits


def test_full_center_fixed():
    points=np.array([[0.,0.,0.],[4.,0.,0.],[10.,0.,0.]])
    center=points.mean(0);before=center.copy()
    sample_local(points,center,64,42)
    np.testing.assert_array_equal(center,before)


def test_deterministic_validation():
    p=np.arange(90).reshape(30,3)
    np.testing.assert_array_equal(sample_local(p,p.mean(0),64,3),sample_local(p,p.mean(0),64,3))


def test_forward_backward_no_gt():
    model=CenterRegressor()
    local=torch.randn(2,64,3);center=torch.randn(2,3)
    pred=model(local,center)
    assert pred.shape==(2,3)
    pred.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert list(importlib.import_module("inspect").signature(model.forward).parameters)==["local_xyz","geometric_center"]


def test_frozen_train_threshold():
    rows=[{"split":"train_sub","time_gap_ms":3.},{"split":"validation_sub","time_gap_ms":999.}]
    assert freeze_config(rows,2)["timestamp_tolerance_ms"] < 4
    assert freeze_config(rows,2)["association_threshold_m"]==2


def test_cumulative_thresholds():
    distances=np.array([.2,.7,1.5,2.5])
    assert [int((distances<=t).sum()) for t in (.5,1,2,3)]==[1,2,3,4]


def test_metrics_same_samples_and_definitions():
    pred=np.array([[1.,2.,3.]])
    m=regression_metrics(pred,np.zeros((1,3)))
    assert np.isclose(m["MSE_3D"],3*m["MSE_coord"])
    with assert_raises(ValueError):regression_metrics(pred,np.zeros((2,3)))
    with assert_raises(ValueError):regression_metrics(np.full((1,3),np.nan),pred)


def test_missing_input_is_failure(tmp_path):
    with assert_raises(FileNotFoundError):preprocess(tmp_path,tmp_path / "out",None)


def test_valid_empty_is_not_failure(tmp_path):
    import build_mmuav_center_regression_dataset as b
    for sensor in ("lidar_360","livox_avia"):
        (tmp_path/sensor).mkdir()
        for i in range(20):np.save(tmp_path/sensor/f"{i}.npy",np.zeros((2,3)))
    result=preprocess(tmp_path,tmp_path / "out",None)
    assert result["nonempty_output_frames"]==0
    assert len(list((tmp_path / "out/lidar_360_processed").glob("*.npy")))==20


def test_heldout_not_preprocessed():
    source=(ROOT / "tools/build_mmuav_center_regression_dataset.py").read_text()
    assert 'for split in ("train_sub", "validation_sub")' in source
    assert 'for split in ("train_sub", "validation_sub", "heldout_test_sub")' not in source


def test_frozen_sample_ids():
    verify_evaluation_ids(["a","b"],["a","b"])
    with assert_raises(ValueError):verify_evaluation_ids(["a"],["a","b"])
    with assert_raises(ValueError):verify_evaluation_ids(["a","a"],["a","a"])


def test_gt_not_forward_input():
    import inspect
    assert list(inspect.signature(CenterRegressor().forward).parameters) == ["local_xyz","geometric_center"]


def test_synthetic_trainer(tmp_path):
    import json
    from unittest.mock import patch
    from build_mmuav_center_regression_dataset import write_csv
    import train_mmuav_center_regressor as trainer
    torch.set_num_threads(1)
    rng=np.random.default_rng(42)
    points=rng.normal(size=(48,3))
    shard=tmp_path / "candidates.npz"
    np.savez(shard,points=points,offsets=np.arange(0,49,8))
    rows=[]
    for i in range(6):
        center=points[i*8:(i+1)*8].mean(0)
        rows.append({"sample_id":str(i),"split":"train_sub" if i<4 else "validation_sub",
                     "shard_path":str(shard),"shard_index":i,"accepted":True,
                     **{f"geometric_{a}":center[j] for j,a in enumerate("xyz")},
                     **{f"gt_{a}":center[j]+.1 for j,a in enumerate("xyz")}})
    write_csv(tmp_path / "metadata.csv",rows)
    write_csv(tmp_path / "evaluation_sample_ids.csv",[{"sample_id":"4"},{"sample_id":"5"}])
    (tmp_path / "dataset_summary.json").write_text(json.dumps({"evaluation_mode":"synthetic"}))
    with patch.object(sys,"argv",["trainer","--dataset-dir",str(tmp_path),"--output-dir",str(tmp_path / "run"),"--epochs","2"]):
        trainer.main()
    assert (tmp_path / "run/center_regression_comparison.csv").exists()
    assert (tmp_path / "run/best_val_loss.pth").exists()


if __name__ == "__main__":
    import tempfile
    tests = [v for k,v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        if "tmp_path" in test.__code__.co_varnames[:test.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as tmp:
                if test.__code__.co_argcount == 2:
                    test(Path(tmp), None)
                else:
                    test(Path(tmp))
        else:
            test()
    print(f"{len(tests)}/{len(tests)} tests passed")
