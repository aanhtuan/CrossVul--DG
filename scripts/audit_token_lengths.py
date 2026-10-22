"""Streaming exact CodeBERT token-length histogram across clean real data."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from crossvul_dg.provenance import verify_processed
from crossvul_dg.utils import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_paths',nargs='+',default=['data/processed/top10.parquet','data/processed/tail.parquet'])
    parser.add_argument('--out',type=Path,default=Path('data/manifests/token_lengths.json'))
    args=parser.parse_args();name='microsoft/codebert-base';revision='3b0952feddeffad0063f274080e3c23d75e7eb39'
    tokenizer=AutoTokenizer.from_pretrained(name,revision=revision)
    tokenizer.model_max_length=10**9
    histogram=Counter();by_project={};total=0;provenances=[]
    for path in args.data_paths:
        provenance=verify_processed(path);provenances.append(dict(path=str(path),**provenance))
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32,columns=['func','project']):
            rows=batch.to_pylist()
            result=tokenizer([row['func'] for row in rows],padding=False,truncation=False,return_length=True,return_attention_mask=False)
            for row,length in zip(rows,result['length']):
                histogram[int(length)]+=1;total+=1
                stats=by_project.setdefault(row['project'],dict(n=0,tokens=0,over128=0,over256=0,over512=0,max=0))
                stats['n']+=1;stats['tokens']+=length;stats['max']=max(stats['max'],length)
                for limit in (128,256,512):stats[f'over{limit}']+=int(length>limit)
            if total%25000<32:print(json.dumps(dict(processed=total)),flush=True)
    def quantile(q):
        rank=max(1,__import__('math').ceil(total*q));cumulative=0
        for length,count in sorted(histogram.items()):
            cumulative+=count
            if cumulative>=rank:return length
    result=dict(data=provenances,model=name,revision=revision,n=total,scope='all clean top and tail rows; includes special tokens',
                quantiles={str(q):quantile(q) for q in (.5,.9,.95,.99)},max=max(histogram),
                mean=sum(length*count for length,count in histogram.items())/total,
                truncation_rates={str(limit):sum(count for length,count in histogram.items() if length>limit)/total for limit in (128,256,512)},
                histogram={str(k):v for k,v in sorted(histogram.items())},per_project=by_project)
    save_json(result,args.out);print(json.dumps({k:v for k,v in result.items() if k not in ('histogram','per_project')},indent=2))

if __name__=='__main__':main()
