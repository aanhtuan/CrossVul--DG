"""Select a compatible configuration using validation AP only; never read test metrics."""
import argparse
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossvul_dg.provenance import signature
from crossvul_dg.utils import save_json


def select(directory,method):
    candidates=[];cohort=None
    for path in sorted(Path(directory).rglob('selection.json')):
        run=json.loads(path.read_text())
        if run.get('status')!='validation_complete' or run['args']['method']!=method:continue
        key=signature(dict(data=run['data_provenance'],split=run['split_audit']['compute_budget'],
                           source=run['software']['source_signature'],backbone=run['args']['resolved_config']['encoder']['name'],
                           revision=run['args']['resolved_config']['encoder']['revision'],seed=run['args']['seed'],held_out=run['args']['held_out']))
        if cohort is not None and key!=cohort:raise ValueError('Ablations have incompatible datasets/splits/source/backbone/seed')
        cohort=key
        value=run['val'].get('pr_auc')
        if not isinstance(value,(int,float)) or not __import__('math').isfinite(value):raise ValueError('Undefined validation AP')
        candidates.append((float(value),str(path),run))
    if not candidates:raise ValueError('No compatible validation-only runs')
    best=sorted(candidates,key=lambda item:(-item[0],item[1]))[0]
    return dict(policy='maximum validation AP; deterministic path tie-break; test metrics not accessed',
                method=method,validation_cohort_signature=cohort,best_selection_path=best[1],best_validation_ap=best[0],
                best_config=best[2]['args']['resolved_config'],candidates=[dict(validation_ap=v,path=p) for v,p,r in candidates])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs_dir',type=Path,required=True);parser.add_argument('--method',default='cc-dro')
    parser.add_argument('--out',type=Path,default=Path('data/manifests/validation_selection.json'))
    args=parser.parse_args();chosen=select(args.runs_dir,args.method);save_json(chosen,args.out)
    print(json.dumps(chosen,indent=2))

if __name__=='__main__':main()
