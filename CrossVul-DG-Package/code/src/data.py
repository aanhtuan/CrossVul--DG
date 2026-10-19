"""Data loading and project-aware temporal splitting for MegaVul.

The chronological 8:1:1 split is applied to the pooled top-K projects in
the order of `commit_date` (or any timestamp present in MegaVul). This is
NOT a per-year split — instead we sort all selected samples by time and
take the first 80% as train, the next 10% as val, and the last 10% as
test. Mild per-project imbalance is expected and reported.

Synthetic data is provided so that the entire pipeline can be exercised
without downloading MegaVul.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

# Columns we expect after preprocessing.
REQUIRED_COLUMNS = ("func", "label", "project", "commit_date")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_processed(path: str | Path) -> pd.DataFrame:
    """Load a pre-processed parquet produced by ``scripts/prepare_data.py``."""
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["commit_date"]).reset_index(drop=True)
    return df


def make_synthetic(n_projects: int = 5,
                   per_project: int = 200,
                   vuln_rate: float = 0.15,
                   seed: int = 0) -> pd.DataFrame:
    """Create a tiny in-memory dataset for sanity tests."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2018-01-01", tz="UTC")
    for p in range(n_projects):
        for i in range(per_project):
            label = int(rng.random() < vuln_rate)
            t = base + pd.Timedelta(days=int(rng.integers(0, 365 * 5)))
            # very short fake C-like function
            body = "vuln" if label else "safe"
            rows.append({
                "func":        f"int proj_{p}_fn_{i}() {{ return {body}; }}",
                "label":       label,
                "project":     f"proj_{p}",
                "commit_date": t,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Project selection and splitting
# ---------------------------------------------------------------------------

def select_top_projects(df: pd.DataFrame, top_k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top-k project rows, tail rows)."""
    counts = df["project"].value_counts()
    top_names = counts.head(top_k).index.tolist()
    top_df = df[df["project"].isin(top_names)].copy()
    tail_df = df[~df["project"].isin(top_names)].copy()
    return top_df.reset_index(drop=True), tail_df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame,
                   train_ratio: float = 0.8,
                   val_ratio:   float = 0.1,
                   test_ratio:  float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split across the whole dataframe.

    Sorts by ``commit_date`` and cuts at the corresponding quantiles. The
    cut-off timestamps are returned alongside so they can be re-used by
    other callers (e.g. evaluation per project).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")
    df = df.sort_values("commit_date").reset_index(drop=True)
    n = len(df)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def lopo_fold(df: pd.DataFrame, held_out_project: str,
              train_ratio: float = 0.8, val_ratio: float = 0.1
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a leave-one-project-out fold.

    The held-out project is the (full) cross-project test set. The other
    projects are jointly split chronologically into train/val using
    ``train_ratio``:``val_ratio`` (rescaled to sum to 1).
    """
    held = df[df["project"] == held_out_project].copy().reset_index(drop=True)
    rest = df[df["project"] != held_out_project].copy().reset_index(drop=True)
    if len(held) == 0:
        raise ValueError(f"No samples for project {held_out_project!r}")
    # rescale ratios
    s = train_ratio + val_ratio
    tr_ratio, va_ratio = train_ratio / s, val_ratio / s
    train, val, _ = temporal_split(rest, tr_ratio, va_ratio, 1.0 - tr_ratio - va_ratio)
    return train, val, held


def split_held_out(held_df: pd.DataFrame, target_ratio: float = 0.5
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the held-out project chronologically into (target_early, test_late).

    The DA baselines may legitimately use *unlabelled* target data, but using
    the very samples we score on is transductive leakage. We therefore expose
    the chronologically-earlier ``target_ratio`` fraction of the held-out
    project as the (label-stripped) DA target, and keep the later fraction as
    the cross-project test set. DG/ERM methods ignore the target portion and
    test on exactly the same ``test_late`` split, so the comparison is fair.
    """
    held = held_df.sort_values("commit_date").reset_index(drop=True)
    n = len(held)
    cut = int(round(n * target_ratio))
    target_early = held.iloc[:cut].reset_index(drop=True)
    test_late = held.iloc[cut:].reset_index(drop=True)
    return target_early, test_late


def compute_pos_weight(df: pd.DataFrame, label_col: str = "label",
                       cap: float = 50.0) -> float:
    """``pos_weight`` = N_neg / N_pos on the training fold, capped for stability.

    Used in the weighted BCE so the optimiser does not collapse to the
    majority (safe) class under MegaVul's ~1:19 imbalance.
    """
    pos = int((df[label_col] == 1).sum())
    neg = int((df[label_col] == 0).sum())
    if pos == 0:
        return 1.0
    return float(min(neg / pos, cap))


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------

@dataclass
class Example:
    func: str
    label: int
    project_id: int


class MegaVulDataset(Dataset):
    """Tokenises functions on the fly. Returns tensors + project id."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512,
                 project_to_id: Optional[dict[str, int]] = None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if project_to_id is None:
            project_to_id = {p: i for i, p in enumerate(sorted(df["project"].unique()))}
        self.project_to_id = project_to_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        enc = self.tokenizer(row["func"], truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(int(row["label"]), dtype=torch.long),
            "project_id":     torch.tensor(self.project_to_id[row["project"]],
                                           dtype=torch.long),
        }


class ClassBalancedBatchSampler(BatchSampler):
    """Build batches containing both classes, oversampling the rare class."""

    def __init__(self, labels: Sequence[int], batch_size: int, seed: int = 0):
        if batch_size < 2:
            raise ValueError("Class-balanced batches require batch_size >= 2")
        labels = np.asarray(labels, dtype=int)
        self.negative = np.flatnonzero(labels == 0)
        self.positive = np.flatnonzero(labels == 1)
        if not len(self.negative) or not len(self.positive):
            raise ValueError("Class-balanced batches require both classes")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.n_batches = max(1, int(np.ceil(len(labels) / batch_size)))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        n_pos = max(1, self.batch_size // 2)
        n_neg = self.batch_size - n_pos
        for _ in range(self.n_batches):
            pos = rng.choice(self.positive, size=n_pos, replace=len(self.positive) < n_pos)
            neg = rng.choice(self.negative, size=n_neg, replace=len(self.negative) < n_neg)
            batch = np.concatenate([pos, neg])
            rng.shuffle(batch)
            yield batch.tolist()


def build_dataloaders(train_df, val_df, test_df, tokenizer, batch_size: int,
                      max_length: int = 512, num_workers: int = 2
                      ) -> Tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    project_to_id = {p: i for i, p in enumerate(sorted(
        pd.concat([train_df["project"], val_df["project"], test_df["project"]]).unique()))}
    train_ds = MegaVulDataset(train_df, tokenizer, max_length, project_to_id)
    val_ds   = MegaVulDataset(val_df,   tokenizer, max_length, project_to_id)
    test_ds  = MegaVulDataset(test_df,  tokenizer, max_length, project_to_id)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
        project_to_id,
    )


def build_env_loaders(train_df: pd.DataFrame, tokenizer, batch_size: int,
                      max_length: int = 512, num_workers: int = 2,
                      project_to_id: Optional[dict[str, int]] = None,
                      class_balanced: bool = False, seed: int = 0,
                      ) -> List[DataLoader]:
    """One DataLoader per project (= environment) for DG methods."""
    if project_to_id is None:
        project_to_id = {p: i for i, p in enumerate(sorted(train_df["project"].unique()))}
    loaders = []
    for p, _ in sorted(project_to_id.items(), key=lambda kv: kv[1]):
        sub = train_df[train_df["project"] == p]
        if len(sub) == 0:
            continue
        ds = MegaVulDataset(sub, tokenizer, max_length, project_to_id)
        if class_balanced:
            sampler = ClassBalancedBatchSampler(
                sub["label"].astype(int).tolist(), batch_size, seed + len(loaders))
            loaders.append(DataLoader(ds, batch_sampler=sampler,
                                      num_workers=num_workers))
        else:
            loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers))
    return loaders


def validate_lopo_frames(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         test_df: pd.DataFrame, *, require_source_cells: bool = False
                         ) -> dict:
    """Validate that a LOPO fold can produce meaningful binary metrics."""
    errors = []
    for name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
        counts = frame["label"].value_counts().to_dict()
        if frame.empty:
            errors.append(f"{name} split is empty")
        if not {0, 1}.issubset(counts):
            errors.append(f"{name} split must contain labels 0 and 1; got {counts}")
    project_counts = {}
    for project, frame in train_df.groupby("project"):
        counts = frame["label"].value_counts().to_dict()
        project_counts[str(project)] = {"n": len(frame), "negative": counts.get(0, 0),
                                        "positive": counts.get(1, 0)}
        if require_source_cells and not {0, 1}.issubset(counts):
            errors.append(f"source project {project!r} lacks one class: {counts}")
    if errors:
        raise ValueError("Invalid LOPO fold:\n- " + "\n- ".join(errors))
    return project_counts
