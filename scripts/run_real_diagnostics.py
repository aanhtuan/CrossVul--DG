"""Tiny real-data overfit diagnostic; requires verified prepared MegaVul.

No synthetic fallback. Does not run a sweep or use test data for model selection.
An overlap between fitting/evaluation is intentional ONLY for this tiny overfit
check. It must never be reported as validation/generalization performance.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import argparse
import json
from pathlib import Path
import sys
import time
import resource
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from crossvul_dg.resources import require_headroom
from crossvul_dg.resources import training_preflight
from crossvul_dg.provenance import verify_processed
from crossvul_dg.data import MegaVulDataset
from crossvul_dg.models import EncoderClassifier
from crossvul_dg.evaluate import compute_metrics
from crossvul_dg.utils import load_config,set_seed,save_json
from torch.utils.data import DataLoader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path',type=Path,default=Path('data/processed/top10.parquet'))
    p.add_argument('--config',type=Path,default=Path('configs/colab.yaml'))
    p.add_argument('--per_class',type=int,default=8);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--out',type=Path,default=Path('experiments/outputs/tiny_real.json'))
    args=p.parse_args()
    if not 1<=args.per_class<=32 or not 1<=args.steps<=100:
        raise ValueError('Diagnostic bounds: 1..32 rows/class, 1..100 steps')
    provenance=verify_processed(args.data_path)
    started=time.monotonic();cfg=load_config(args.config)
    resources=training_preflight(cfg);set_seed(0);torch.set_num_threads(1)
    if cfg['train']['device']=='cuda':torch.cuda.reset_peak_memory_stats()
    # Deterministic bounded reservoir per class (does not materialize the full parquet).
    import random
    rng=random.Random(0);chosen={0:[],1:[]};seen={0:0,1:0}
    for batch in pq.ParquetFile(args.data_path).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            label=row['label'];seen[label]+=1
            if len(chosen[label])<args.per_class: chosen[label].append(row)
            else:
                index=rng.randrange(seen[label])
                if index<args.per_class: chosen[label][index]=row
    if any(len(chosen[c])<args.per_class for c in (0,1)):
        raise ValueError('Insufficient real examples per class')
    df=pd.DataFrame(chosen[0]+chosen[1]);tokenizer=AutoTokenizer.from_pretrained(cfg['encoder']['name'],revision=cfg['encoder']['revision'])
    model=EncoderClassifier(cfg['encoder']['name'],revision=cfg['encoder']['revision'],dropout=0.,
                            gradient_checkpointing=cfg['encoder'].get('gradient_checkpointing',False))
    device=cfg['train']['device'];model.to(device)
    loader=DataLoader(MegaVulDataset(df,tokenizer,min(cfg['encoder']['max_length'],256)),batch_size=2,shuffle=True,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'],foreach=False)
    amp=cfg['train'].get('amp',False);scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=1024.)
    losses=[];iterator=iter(loader)
    for step in range(args.steps):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader);batch=next(iterator)
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            logits=model(batch['input_ids'].to(device),batch['attention_mask'].to(device)).float()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch['label'].float().to(device))
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not amp)
        scaler.step(optimizer);scaler.update()
        losses.append(loss.item())
        if (step+1)%10==0: print(json.dumps(dict(step=step+1,loss=loss.item())),flush=True)
    model.eval();ys=[];ps=[]
    with torch.no_grad():
        for batch in loader:
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                logits=model(batch['input_ids'].to(device),batch['attention_mask'].to(device)).float()
            ys.extend(batch['label'].tolist());ps.extend(torch.sigmoid(logits).cpu().tolist())
    result=dict(kind='real-training-sample-overfit-diagnostic-only',data=provenance,resources=resources,seed=0,
                selected_hashes=df.code_hash.tolist(),losses=losses,fit_sample_metrics=compute_metrics(ys,ps),
                config=cfg,runtime_seconds=time.monotonic()-started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,
                note='No generalization claim. Failure to overfit requires diagnosis; success does not establish LOPO performance.')
    save_json(result,args.out)

if __name__=='__main__': main()
