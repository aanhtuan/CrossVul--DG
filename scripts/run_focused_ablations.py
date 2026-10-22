"""Print matched sequential single-fold ablation commands; --execute runs them.

Use only after tiny-real and one-fold ERM/CC-DRO checks pass. No test tuning.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossvul_dg.resources import require_headroom


def commands(config,data_path,held_out,out_root,seed=0):
    for length in (256,512):
        for method in ('erm','cc-dro'):
            for balanced in ([True,False] if method=='cc-dro' else [False]):
                for lam in ([0,.01,.1,1.] if method=='cc-dro' else [0]):
                    name=f'{method}_len{length}_balanced{balanced}_inv{lam}_seed{seed}'
                    yield [sys.executable,'scripts/run_lopo.py','--config',str(config),'--method',method,
                           '--held_out',held_out,'--seed',str(seed),'--data_path',str(data_path),
                           '--output_dir',str(Path(out_root)/name),'--override',f'encoder.max_length={length}',
                           f'cc_dro.lambda_inv={lam}',f'cc_dro.balanced_batches={str(balanced).lower()}']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/colab.yaml');p.add_argument('--data_path',default='data/processed/top10.parquet')
    p.add_argument('--held_out',required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out_root',default='runs/focused');p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    for command in commands(args.config,args.data_path,args.held_out,args.out_root,args.seed):
        print(shlex.join(command),flush=True)
        if args.execute:
            require_headroom();subprocess.run(command,check=True)

if __name__=='__main__': main()
