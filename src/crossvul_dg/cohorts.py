"""Explicit compute-budget cohorts. Quality exclusions and budget caps are distinct.

Validation/test selection is hash-based without label inspection. Training quotas
preserve prevalence as closely as a small budget permits, retaining both classes.
These are real-data diagnostics, never silently represented as full-population LOPO.
"""
import hashlib
import duckdb
import pandas as pd
from .provenance import signature


def _ordered(frame,seed):
    hashes=frame.code_hash.map(lambda value:hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest())
    return frame.loc[hashes.sort_values(kind='stable').index]


def bounded_frames(frames,*,train_per_project=0,val_limit=0,test_limit=0,target_limit=0,seed=0):
    if any(x<0 for x in (train_per_project,val_limit,test_limit,target_limit)):
        raise ValueError('Compute-budget limits must be nonnegative')
    train,val,target,test=frames
    before=[len(frame) for frame in frames]
    if train_per_project:
        if train_per_project<2: raise ValueError('Source project budget must permit both classes')
        chunks=[]
        for project,group in train.groupby('project',sort=True):
            if len(group)<=train_per_project:
                chunks.append(group);continue
            positive=group[group.label==1];negative=group[group.label==0]
            if positive.empty or negative.empty: raise ValueError(f'Source project {project} lacks a class')
            n_pos=min(len(positive),max(1,round(train_per_project*len(positive)/len(group))))
            n_neg=min(len(negative),train_per_project-n_pos)
            n_pos=min(len(positive),train_per_project-n_neg)
            chunks.extend([_ordered(positive,seed).head(n_pos),_ordered(negative,seed).head(n_neg)])
        train=pd.concat(chunks).sort_values(['commit_date','project','code_hash']).reset_index(drop=True)
    def cap(frame,limit):
        return (_ordered(frame,seed).head(limit).sort_values(['commit_date','project','code_hash']).reset_index(drop=True)
                if limit and len(frame)>limit else frame.reset_index(drop=True))
    val,target,test=cap(val,val_limit),cap(target,target_limit),cap(test,test_limit)
    out=[train,val,target,test]
    bounded=any((train_per_project,val_limit,test_limit,target_limit))
    audit=dict(protocol='bounded-real-lopo-v1' if bounded else 'full-real-lopo-v2',seed=seed,
               train_per_project=train_per_project,val_limit=val_limit,target_limit=target_limit,test_limit=test_limit,
               full_partition_rows=before,evaluated_partition_rows=[len(frame) for frame in out],
               selection_policy='Training per-project proportional binary quotas; validation/target/test hash ranking without labels',
               hashes={name:signature(sorted(frame.code_hash.tolist())) for name,frame in zip(('train','validation','target','test'),out)})
    return out,audit


def materialize(path,metadata,memory_limit='128MB'):
    """Read only chosen function bodies from a potentially large parquet."""
    if metadata.empty:
        return metadata.assign(func=pd.Series(dtype=str))
    con=duckdb.connect(config={'memory_limit':memory_limit,'threads':1})
    try:
        con.register('chosen_hashes',metadata[['code_hash']])
        result=con.execute('SELECT p.code_hash,p.func FROM read_parquet(?) p INNER JOIN chosen_hashes c USING(code_hash)',[str(path)]).df()
    finally: con.close()
    result=metadata.merge(result,on='code_hash',how='left',validate='one_to_one')
    if len(result)!=len(metadata) or result.func.isna().any(): raise ValueError('Selected functions could not be materialized')
    return result
