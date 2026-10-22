"""Synthetic fixtures only. These are never used as released research data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pandas as pd
import torch
from torch import nn
from crossvul_dg.data import build_env_loaders,MegaVulDataset
from torch.utils.data import DataLoader

class Tokenizer:
    def __call__(self,text,max_length=12,**kwargs):
        ids=[ord(c)%64+1 for c in text[:max_length]]
        ids += [0]*(max_length-len(ids))
        x=torch.tensor([ids])
        return dict(input_ids=x,attention_mask=(x!=0).long())

class TinyModel(nn.Module):
    def __init__(self,dropout=.2):
        super().__init__()
        self.model_name='synthetic-test-only'
        self.emb=nn.Embedding(66,8)
        self.classifier=nn.Sequential(nn.Linear(8,8),nn.Tanh(),nn.Dropout(dropout),nn.Linear(8,1))
    def encode(self,input_ids,attention_mask):
        x=self.emb(input_ids)
        mask=attention_mask.unsqueeze(-1)
        return (x*mask).sum(1)/mask.sum(1).clamp_min(1)
    def forward(self,input_ids,attention_mask):
        return self.classifier(self.encode(input_ids,attention_mask)).squeeze(-1)


def frame(project,start=0,n=12):
    return pd.DataFrame([dict(func=f'{"unsafe" if i%2 else "safe"}_{project}_{i}',label=i%2,
                             project=project,commit_date=pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(days=start+i))
                         for i in range(n)])


def loaders(balanced=False):
    train=pd.concat([frame('a'),frame('b')],ignore_index=True)
    val=frame('a',start=100,n=8)
    val['func']=val.func+'_validation'
    mapping={'a':0,'b':1,'held':2}
    tok=Tokenizer()
    env=build_env_loaders(train,tok,4,12,0,mapping,balanced,7)
    valid=DataLoader(MegaVulDataset(val,tok,12,mapping),batch_size=4)
    target=DataLoader(MegaVulDataset(frame('held',200,8),tok,12,mapping),batch_size=4)
    return env,valid,target
