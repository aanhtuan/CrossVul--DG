"""Pseudo-domain construction by clustering tail-project embeddings."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from .data import MegaVulDataset
from .models import CodeBERTClassifier


@torch.no_grad()
def project_mean_embedding(model: CodeBERTClassifier, df_proj: pd.DataFrame,
                           tokenizer, max_length: int = 512,
                           batch_size: int = 16, max_samples: int = 1024,
                           device: str = "cuda") -> np.ndarray:
    if len(df_proj) > max_samples:
        df_proj = df_proj.sample(n=max_samples, random_state=0).reset_index(drop=True)
    ds = MegaVulDataset(df_proj, tokenizer, max_length,
                        {p: 0 for p in df_proj["project"].unique()})
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats = []
    model.eval()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am  = batch["attention_mask"].to(device)
        feats.append(model.encode(ids, am).cpu().numpy())
    return np.concatenate(feats, axis=0).mean(axis=0)


def cluster_tail_projects(model: CodeBERTClassifier, tail_df: pd.DataFrame,
                          tokenizer, n_clusters: int = 5,
                          max_samples_per_proj: int = 256,
                          device: str = "cuda") -> Dict[str, int]:
    """Return a mapping ``project_name -> cluster_id`` (cluster_id offset 1000
    so it doesn't collide with top-K integer project ids).
    """
    projects = tail_df["project"].unique().tolist()
    embeds = []
    for p in projects:
        sub = tail_df[tail_df["project"] == p]
        if len(sub) == 0:
            continue
        emb = project_mean_embedding(
            model, sub, tokenizer, batch_size=16,
            max_samples=max_samples_per_proj, device=device,
        )
        embeds.append(emb)
    X = np.stack(embeds, axis=0)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
    return {p: 1000 + int(c) for p, c in zip(projects, km.labels_)}
