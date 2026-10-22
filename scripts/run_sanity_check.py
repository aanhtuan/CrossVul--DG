"""End-to-end sanity check on synthetic data (no CodeBERT download, ~1 min CPU).

Substitutes a tiny randomly-initialised Transformer and a char-level tokenizer
for CodeBERT, then runs the REAL trainers from ``crossvul_dg.trainers`` so the whole
pipeline -- including the proposed CC-DRO, pos_weight BCE, PR-AUC model
selection and val threshold tuning -- is exercised. Numbers are meaningless
(tiny model + tiny data); the point is that every code path runs.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crossvul_dg.data import (
    build_env_loaders, compute_pos_weight, lopo_fold, make_synthetic,
)
from crossvul_dg.evaluate import tune_threshold
from crossvul_dg.trainers import TrainerConfig, build_trainer
from crossvul_dg.utils import set_seed


class CharTokenizer:
    pad_token_id = 0

    def __init__(self, max_length: int = 64):
        self.max_length = max_length

    def __call__(self, text, truncation=True, padding="max_length",
                 max_length=None, return_tensors="pt"):
        max_length = max_length or self.max_length
        ids = [min(ord(c), 255) for c in text[:max_length]]
        ids += [0] * (max_length - len(ids))
        attn = [1 if x != 0 else 0 for x in ids]
        return {"input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long)}


class TinyClassifier(nn.Module):
    """Mimics EncoderClassifier's interface (encode / classifier / forward)."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 64,
                 max_length: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2,
                                       dim_feedforward=128, batch_first=True),
            num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 1))

    def encode(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        return x[:, 0]

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.encode(input_ids, attention_mask)).squeeze(-1)


def main():
    torch.set_num_threads(1)
    set_seed(0)
    print("Building synthetic data...")
    df = make_synthetic(n_projects=4, per_project=120, vuln_rate=0.15, seed=0)
    held_out = "proj_3"
    train_df, val_df, test_df = lopo_fold(df, held_out, train_ratio=0.8, val_ratio=0.1)
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"(held_out={held_out})")

    tok = CharTokenizer(max_length=32)
    project_to_id = {p: i for i, p in enumerate(sorted(df["project"].unique()))}
    pw = compute_pos_weight(train_df)
    print(f"  pos_weight={pw:.2f}")

    from crossvul_dg.data import MegaVulDataset
    from torch.utils.data import DataLoader
    env_loaders = build_env_loaders(train_df, tok, batch_size=8, max_length=32,
                                    num_workers=0, project_to_id=project_to_id)
    val_loader = DataLoader(MegaVulDataset(val_df, tok, 32, project_to_id),
                            batch_size=16, shuffle=False)
    test_loader = DataLoader(MegaVulDataset(test_df, tok, 32, project_to_id),
                             batch_size=16, shuffle=False)

    target_df=make_synthetic(n_projects=1,per_project=24,seed=20)
    target_df['project']='target_only'
    target_df['label']=0
    project_to_id['target_only']=len(project_to_id)
    target_loader=DataLoader(MegaVulDataset(target_df,tok,32,project_to_id),batch_size=8)
    for method in ["erm", "irm", "dro", "cc-dro", "pd-dro", "da-coral", "da-dann"]:
        print(f"\nRunning {method.upper()}...")
        env_loaders=build_env_loaders(train_df,tok,8,32,0,project_to_id,
                                      class_balanced=method=='cc-dro',seed=0)
        cfg = TrainerConfig(method=method, epochs=1, lr=1e-3, pos_weight=1. if method=='cc-dro' else pw,
                            irm_warmup_epochs=0, cc_warmup_epochs=0, device="cpu")
        trainer = build_trainer(method, TinyClassifier(), cfg,target_loader=target_loader,
                                n_source_domains=len(env_loaders))
        trainer.fit(env_loaders, val_loader)
        yv, pv, _ = trainer.predict(val_loader)
        thr, _ = tune_threshold(yv, pv, objective="f1")
        m = trainer.evaluate(test_loader, threshold=thr, project_ids=True)
        print(f"   thr={thr:.3f}  PR-AUC={m['pr_auc']:.3f}  F1={m['f1']:.3f}  "
              f"R@P50={m['recall_at_p50']:.3f}")

    print("\nSynthetic sanity check PASSED -- seven trainer paths executed; no real-data or CUDA claim.")


if __name__ == "__main__":
    main()
