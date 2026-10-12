#!/usr/bin/env python3
"""Six independent runs: control/a_head x full/heads/lora. Sequential by default."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--action',choices=('prepare','gate','benchmark','train','eval','all'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--regimes',choices=('full','heads','lora'),nargs='+',default=['full','heads','lora'])
    p.add_argument('--variants',choices=('control','a_head'),nargs='+',default=['control','a_head'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--benchmark-updates',type=int,default=200)
    args=p.parse_args()
    for regime in args.regimes:
        cmd=[sys.executable,str(HERE/'run_suite.py'),'--action',args.action,'--output',str(args.output/regime),
             '--variants',*args.variants,'--seeds',*map(str,args.seeds),'--epochs',str(args.epochs),
             '--device',args.device,'--num-workers',str(args.num_workers),'--train-mode',regime,'--efficiency','--benchmark-updates',str(args.benchmark_updates)]
        if args.checkpoint: cmd+=['--checkpoint',str(args.checkpoint)]
        if args.data_root: cmd+=['--data-root',str(args.data_root)]
        if args.resume: cmd+=['--resume']
        subprocess.run(cmd,check=True)


if __name__=='__main__': main()
