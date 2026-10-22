"""One-variable-at-a-time validation-only real ablations; prints, never executes."""
import argparse
import json
from pathlib import Path
import shlex
import sys


def commands(config,data,held,out,seed=0,train_per_project=0,val_limit=0,test_limit=0):
    # Shared baseline: lambda=1, LR=2e-5, length=128. Avoid duplicate baseline jobs.
    variants=[('base',1.,2e-5,128)]
    variants += [(f'lambda{v}',v,2e-5,128) for v in (0.,.01,.1)]
    variants += [(f'lr{v}',1.,v,128) for v in (1e-5,3e-5)]
    variants += [(f'length{v}',1.,2e-5,v) for v in (256,512)]
    for label,lam,lr,length in variants:
        yield [sys.executable,'scripts/run_lopo.py','--config',str(config),'--method','cc-dro',
               '--held_out',held,'--seed',str(seed),'--data_path',str(data),'--output_dir',str(Path(out)/label),
               '--train_per_project',str(train_per_project),'--val_limit',str(val_limit),'--test_limit',str(test_limit),
               '--selection_only','--override',f'cc_dro.lambda_inv={lam}',f'train.lr={lr}',f'encoder.max_length={length}']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/local_gpu.yaml');parser.add_argument('--data_path',default='data/processed/top10.parquet')
    parser.add_argument('--held_out',required=True);parser.add_argument('--out_root',default='runs/real_ablations')
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--train_per_project',type=int,default=0)
    parser.add_argument('--val_limit',type=int,default=0);parser.add_argument('--test_limit',type=int,default=0)
    args=parser.parse_args()
    for command in commands(args.config,args.data_path,args.held_out,args.out_root,args.seed,args.train_per_project,args.val_limit,args.test_limit):
        print(shlex.join(command))

if __name__=='__main__':main()
