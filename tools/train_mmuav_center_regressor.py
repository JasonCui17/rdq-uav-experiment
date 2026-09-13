#!/usr/bin/env python3
"""Manual M2 training. Frozen GT-conditioned validation sample set."""
import argparse
import json
import sys
import subprocess
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from rdq_uav.mmuav.center_regressor import CenterRegressor,sample_local,regression_metrics,verify_evaluation_ids,predict_delta
from evaluate_mmuav_center_regression import accepted_rows,arrays
from build_mmuav_center_regression_dataset import write_csv,MODE


class ClusterDataset(Dataset):
    def __init__(self,rows,n,train,variant="full"):
        self.rows,self.n,self.train=rows,n,train
        self.variant=variant
        self.cache={}
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        center,gt=arrays([r]); center,gt=center[0].astype(np.float32),gt[0].astype(np.float32)
        if self.variant == "center_only":
            # Never open the raw-point shard for this variant.
            return torch.empty(0,3),torch.from_numpy(center),torch.from_numpy(gt)
        path=r["shard_path"]
        if path not in self.cache:
            with np.load(path,allow_pickle=False) as f:
                self.cache[path]=(f["points"],f["offsets"])
        points,offsets=self.cache[path]; j=int(r["shard_index"])
        center,gt=arrays([r]); center,gt=center[0].astype(np.float32),gt[0].astype(np.float32)
        seed=int(np.random.randint(2**31)) if self.train else 42+i
        local=sample_local(points[offsets[j]:offsets[j+1]],center,self.n,seed)
        return torch.from_numpy(local),torch.from_numpy(center),torch.from_numpy(gt)


def update_ablation_comparison(parent, frozen):
    """Only merge completed experiments with exactly the same evaluation IDs."""
    combined=[]
    for name,folder in (("M2_FULL","pointnet_m2"),("M2_POINTS_ONLY","m2_points_only"),
                        ("M2_CENTER_ONLY","m2_center_only")):
        path=parent/folder
        if not (path/'evaluation_summary.json').exists():
            continue
        import csv
        ids=[r['sample_id'] for r in csv.DictReader((path/'evaluation_sample_ids.csv').open())]
        verify_evaluation_ids(ids,frozen)
        summary=json.loads((path/'evaluation_summary.json').read_text())
        if summary['evaluation_mode']!=MODE: raise ValueError('Evaluation modes differ')
        geometric,regressed=summary['comparison']
        if not combined:
            combined.append(dict(Model='Geometric baseline',**geometric,relative_reduction_vs_geometric=0))
        else:
            if not np.isclose(combined[0]['MSE_coord'],geometric['MSE_coord'],rtol=1e-10):
                raise ValueError('Geometric baselines differ')
        combined.append(dict(Model=name,**regressed,
                             relative_reduction_vs_geometric=1-regressed['MSE_coord']/geometric['MSE_coord']))
    if combined:
        write_csv(parent/'m2_5_comparison.csv',combined)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("full","points_only","center_only"),default="full")
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--patience",type=int,default=15)
    p.add_argument("--num-points",type=int,default=64)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--learning-rate",type=float,default=.001)
    args=p.parse_args()
    if min(args.epochs,args.patience,args.num_points,args.batch_size)<=0 or args.learning_rate<=0:
        p.error("Training dimensions/limits and learning rate must be positive")
    out=args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError("Refusing to overwrite experiment")
    out.mkdir(parents=True,exist_ok=True)
    np.random.seed(42);torch.manual_seed(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} batch_size={args.batch_size} lr={args.learning_rate}",flush=True)
    rows=accepted_rows(args.dataset_dir)
    train=[r for r in rows if r["split"]=="train_sub"]
    val=[r for r in rows if r["split"]=="validation_sub"]
    if not train or not val: raise ValueError("Accepted train and validation samples required")
    import csv
    with (args.dataset_dir / "evaluation_sample_ids.csv").open() as f:
        frozen=[r["sample_id"] for r in csv.DictReader(f)]
    verify_evaluation_ids([r["sample_id"] for r in val],frozen)
    loaders=[DataLoader(ClusterDataset(sub,args.num_points,training,args.variant),batch_size=args.batch_size,
                        shuffle=training) for sub,training in ((train,True),(val,False))]
    model=CenterRegressor(args.variant).to(device);opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    history=[];best=float("inf");counter=0
    config={**vars(args),"seed":42,"evaluation_mode":MODE,
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "dataset_summary_sha256":hashlib.sha256((args.dataset_dir / "dataset_summary.json").read_bytes()).hexdigest()}
    (out / "config.json").write_text(json.dumps(config,default=str,indent=2))
    for epoch in range(1,args.epochs+1):
        model.train();total=0;seen=0;start=time.perf_counter()
        progress=tqdm(loaders[0],desc=f"Epoch {epoch}/{args.epochs}")
        for local,center,gt in progress:
            local,center,gt=[v.to(device) for v in (local,center,gt)]
            opt.zero_grad();loss=torch.nn.functional.mse_loss(predict_delta(model,local,center),gt-center)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite regression loss")
            loss.backward();opt.step();total+=loss.item()*len(gt)
            seen+=len(gt)
            if hasattr(progress,"set_postfix"):
                progress.set_postfix(GPU_mem=f"{torch.cuda.memory_allocated()/2**30:.2f}G" if device.type=="cuda" else "N/A",
                                     loss=f"{loss.item():.6f}",lr=opt.param_groups[0]["lr"],
                                     samples_s=f"{seen/(time.perf_counter()-start):.1f}")
        model.eval();pred=[]
        with torch.no_grad():
            for local,center,gt in loaders[1]:
                pred.append((predict_delta(model,local.to(device),center.to(device))+center.to(device)).cpu().numpy())
        metrics=regression_metrics(np.concatenate(pred),arrays(val)[1]); vl=metrics["MSE_coord"]
        if vl<best:
            best=vl;counter=0;best_epoch=epoch;torch.save(model.state_dict(),out / "best_val_loss.pth")
        else: counter+=1
        torch.save(model.state_dict(),out / "last.pth")
        row={"epoch":epoch,"train_loss":total/len(train),"val_loss":vl,**metrics,"patience_counter":counter}
        history.append(row);write_csv(out / "training_history.csv",history)
        print(row,flush=True)
        if counter>=args.patience: break
    model.load_state_dict(torch.load(out / "best_val_loss.pth",map_location=device));model.eval();pred=[]
    with torch.no_grad():
        for local,center,gt in loaders[1]:
            pred.append((predict_delta(model,local.to(device),center.to(device))+center.to(device)).cpu().numpy())
    comparison=[{"method":"geometric_center","evaluation_mode":MODE,**regression_metrics(*arrays(val))},
                {"method":"pointnet_center","evaluation_mode":MODE,**regression_metrics(np.concatenate(pred),arrays(val)[1])}]
    write_csv(out / "center_regression_comparison.csv",comparison)
    write_csv(out / "evaluation_sample_ids.csv",[{"sample_id":v} for v in frozen])
    (out / "evaluation_summary.json").write_text(json.dumps({"evaluation_mode":MODE,"best_epoch":best_epoch,
        "actual_epochs":epoch,"variant":args.variant,"comparison":comparison},indent=2))
    update_ablation_comparison(out.parent,frozen)


if __name__=="__main__":main()
