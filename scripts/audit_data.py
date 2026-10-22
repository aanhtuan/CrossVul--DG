"""Audit protocol-v2 real parquet and LOPO folds without loading a model.

Near-duplicate screening is bounded and lexical, not an exhaustive semantic proof.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pyarrow.parquet as pq
from crossvul_dg.data import load_processed,make_lopo_partitions,validate_lopo_frames
from crossvul_dg.provenance import verify_processed
from crossvul_dg.utils import save_json


def _simhash(text):
    tokens=re.findall(r'\w+|[^\w\s]',text[:8192])[:256]
    shingles=[' '.join(tokens[i:i+5]) for i in range(max(1,len(tokens)-4))]
    vote=[0]*64
    for shingle in set(shingles):
        h=int.from_bytes(hashlib.blake2b(shingle.encode(),digest_size=8).digest(),'big')
        for bit in range(64): vote[bit]+=1 if h&(1<<bit) else -1
    return sum((v>=0)<<bit for bit,v in enumerate(vote))


def near_duplicate_candidates(path,max_records=2000,max_comparisons=100000):
    buckets=defaultdict(list);seen=compared=0;pairs=[];capped=False
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['func','project','code_hash']):
        for row in batch.to_pylist():
            if seen>=max_records or compared>=max_comparisons:
                capped=True;break
            seen+=1;h=_simhash(row['func']);candidates={}
            for band in range(4):
                key=(band,(h>>(16*band))&65535)
                for old in buckets[key]: candidates[old[1]]=old
            for oldhash,oldid,project,length in candidates.values():
                if compared>=max_comparisons: capped=True;break
                compared+=1
                if (h^oldhash).bit_count()<=3 and min(length,len(row['func']))/max(length,len(row['func']),1)>=.8:
                    if len(pairs)<1000:
                        pairs.append(dict(left=oldid,right=row['code_hash'],cross_project=project!=row['project']))
            for band in range(4):
                key=(band,(h>>(16*band))&65535)
                if len(buckets[key])<8:
                    buckets[key].append((h,row['code_hash'],row['project'],len(row['func'])))
        if capped: break
    return dict(method='64-bit token-5-shingle SimHash, four 16-bit bands, Hamming<=3, length ratio>=0.8',
                examined_rows=seen,comparisons=compared,capped=capped,bucket_cap=8,candidates=pairs,
                note='Diagnostic candidates require review. First bounded rows in timestamp order, first 256 tokens per function; absence is not an exhaustive leakage guarantee.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path',type=Path,default=Path('data/processed/top10.parquet'))
    p.add_argument('--out',type=Path,default=Path('data/manifests/fold_audit.json'))
    p.add_argument('--near_max_records',type=int,default=2000)
    args=p.parse_args();provenance=verify_processed(args.data_path)
    df=load_processed(args.data_path,metadata_only=True);folds={}
    for project in sorted(df.project.unique()):
        try:
            frames,counts=make_lopo_partitions(df,project)
            train,val,target,test=frames
            counts['source_cells']=validate_lopo_frames(train,val,test,require_source_cells=True)
            counts['partitions']={name:dict(n=len(frame),positive=int(frame.label.sum()),negative=int((frame.label==0).sum()),
                positive_rate=float(frame.label.mean()),min_date=str(frame.commit_date.min()),max_date=str(frame.commit_date.max()))
                for name,frame in zip(('train','validation','target','test'),frames)}
            counts['original_rows']=len(df)
            counts['retained_rows']=sum(map(len,frames))
            counts['excluded_rows']=len(df)-counts['retained_rows']
            counts['status']='verified'
        except ValueError as exc: counts=dict(status='blocked',reason=str(exc))
        folds[project]=counts
        print(json.dumps(dict(project=project,status=counts['status'],retained_rows=counts.get('retained_rows'))),flush=True)
    result=dict(data=provenance,folds=folds,near_duplicates=near_duplicate_candidates(args.data_path,args.near_max_records))
    save_json(result,args.out)
    print(json.dumps({p:v['status'] for p,v in folds.items()},indent=2))
    if any(v['status']=='blocked' for v in folds.values()): raise SystemExit(2)

if __name__=='__main__': main()
