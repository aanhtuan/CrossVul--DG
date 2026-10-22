"""Pseudo-domains from source-eligible tail projects only; labels unused in clustering."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from .data import MegaVulDataset


@torch.no_grad()
def project_mean_embedding(model,df_proj,tokenizer,max_length=512,batch_size=2,max_samples=1024,device='cpu'):
    if max_samples<1 or batch_size<1 or df_proj.empty:
        raise ValueError('Pseudo-domain embeddings require nonempty bounded samples')
    if len(df_proj)>max_samples:
        df_proj=df_proj.sample(n=max_samples,random_state=0).reset_index(drop=True)
    ds=MegaVulDataset(df_proj,tokenizer,max_length,{p:0 for p in df_proj.project.unique()})
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    model.to(device); model.eval()
    total=None;count=0
    for batch in loader:
        features=model.encode(batch['input_ids'].to(device),batch['attention_mask'].to(device))
        subtotal=features.double().sum(0).cpu()
        total=subtotal if total is None else total+subtotal
        count+=len(features)
    return (total/count).numpy()


def cluster_tail_projects(model,tail_df,tokenizer,n_clusters=5,max_samples_per_proj=256,device='cpu',max_length=512,batch_size=2):
    projects=sorted(tail_df.project.unique())
    if n_clusters<1 or len(projects)<n_clusters:
        raise ValueError('Not enough tail projects for pseudo-domain clustering')
    model.to(device)
    embeddings=np.stack([project_mean_embedding(model,tail_df[tail_df.project==project],tokenizer,
                        max_length,batch_size,max_samples_per_proj,device) for project in projects])
    km=KMeans(n_clusters=n_clusters,random_state=0,n_init=10).fit(embeddings)
    if len(set(km.labels_))!=n_clusters:
        raise ValueError('Degenerate tail embeddings yield fewer pseudo-domains than requested')
    return {project:1000+int(label) for project,label in zip(projects,km.labels_)}
