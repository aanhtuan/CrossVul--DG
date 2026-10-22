"""Strict data loading and protocol-v2 LOPO partitioning.

Full held-out projects are separate domains. Source train:validation ratios are
rescaled; equal timestamps stay together. CVE/commit/code groups spanning any
partitions are quarantined. The held-out early portion is unlabelled DA target,
and all methods share the late test portion. Historical cross-project evaluation
does not guarantee source training dates precede every held-out test date.

Synthetic generation below is exclusively a test/sanity fixture facility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
import json
from .preprocessing import _code_hash

# Columns we expect after preprocessing.
REQUIRED_COLUMNS = ("func", "label", "project", "commit_date")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_processed(path: str | Path, metadata_only: bool = False) -> pd.DataFrame:
    """Load a pre-processed parquet produced by ``scripts/prepare_data.py``."""
    import pyarrow.parquet as pq
    from .resources import snapshot
    metadata=pq.ParquetFile(path).metadata
    columns=[name for name in pq.ParquetFile(path).schema_arrow.names if name!='func'] if metadata_only else None
    uncompressed=sum(metadata.row_group(i).column(j).total_uncompressed_size
                     for i in range(metadata.num_row_groups) for j in range(metadata.num_columns)
                     if columns is None or metadata.row_group(i).column(j).path_in_schema in columns)
    if uncompressed*3 > snapshot()['available_ram_bytes']:
        raise RuntimeError('Processed parquet exceeds available RAM budget; use a larger-memory host or a bounded real diagnostic')
    df = pd.read_parquet(path,columns=columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns and not (metadata_only and c=='func')]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")
    if df['commit_date'].isna().any() or not df['label'].isin([0,1]).all():
        raise ValueError('Invalid dates or labels in processed data; refusing to change population')
    if df['project'].isna().any() or (not metadata_only and (df['func'].isna().any() or (df['func'].str.strip() == '').any())):
        raise ValueError('Empty code/project in processed data')
    if 'code_hash' not in df:
        raise ValueError('Legacy parquet lacks code_hash: reprocess using protocol v2')
    return df.reset_index(drop=True)


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

    Sorts by timestamp and moves cuts to the end of tied timestamps.
    Purges groups crossing partitions and returns the three dataframes.
    Empty/single-class partitions are rejected by fold validation.
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(x < 0 for x in ratios) or abs(sum(ratios)-1) > 1e-6 or train_ratio <= 0:
        raise ValueError('Invalid temporal split ratios')
    df = df.sort_values(['commit_date', 'project'], kind='stable').reset_index(drop=True)
    if df.empty or df['commit_date'].isna().any():
        raise ValueError('Temporal split requires valid dated rows')
    n = len(df)
    # Move boundaries to the END of a timestamp, without looking at labels.
    first = max(1, min(n, int(n * train_ratio)))
    second = n if test_ratio < 1e-10 else max(first, min(n, int(n*(train_ratio+val_ratio))))
    cut1 = df.iloc[first-1]['commit_date']
    cut2 = df.iloc[second-1]['commit_date']
    masks = (df['commit_date'] <= cut1,
             (df['commit_date'] > cut1) & (df['commit_date'] <= cut2),
             df['commit_date'] > cut2)
    return tuple(x.reset_index(drop=True) for x in purge_group_overlap([df[m] for m in masks]))


def iter_metadata(frame):
    columns=[k for k in ('code_hash','provenance','cve_id','commit_hash') if k in frame]
    if 'code_hash' not in columns: columns.append('func')
    for values in frame[columns].itertuples(index=False,name=None):
        yield dict(zip(columns,values))


def _identifiers(row):
    keys = {('code_hash', row.get('code_hash') or _code_hash(row['func']))}
    provenance = row.get('provenance')
    aliases = json.loads(provenance) if isinstance(provenance, str) and provenance else {}
    for field in ('cve_id','commit_hash'):
        values = aliases.get(field, []) + ([row[field]] if isinstance(row.get(field), str) and row[field] else [])
        for value in values:
            # CVEs are globally scoped; commit SHAs are shared across forks.
            keys.add((field, value))
    return keys


def purge_group_overlap(frames):
    """Remove ALL rows in any CVE/commit/code group spanning partitions.

    Labels are never consulted. Population reductions are recorded by caller.
    """
    owners = {}
    all_keys = []
    for partition, frame in enumerate(frames):
        keys = [_identifiers(row) for row in iter_metadata(frame)]
        all_keys.append(keys)
        for row_keys in keys:
            for key in row_keys:
                owners.setdefault(key, set()).add(partition)
    bad = {key for key, parts in owners.items() if len(parts) > 1}
    return [frame.loc[[not bool(keys & bad) for keys in row_keys]].copy().reset_index(drop=True)
            for frame, row_keys in zip(frames, all_keys)]


def assert_no_leakage(frames):
    seen = set()
    for frame in frames:
        keys = set().union(*(_identifiers(row) for row in iter_metadata(frame))) if len(frame) else set()
        if seen & keys:
            raise ValueError('CVE/commit/code leakage across partitions')
        seen |= keys


def make_lopo_partitions(df, held_out, train_ratio=.8, val_ratio=.1, target_ratio=.5):
    train, val, held = lopo_fold(df, held_out, train_ratio, val_ratio)
    target, test = split_held_out(held, target_ratio)
    before = [len(f) for f in (train,val,target,test)]
    train,val,target,test = purge_group_overlap([train,val,target,test])
    assert_no_leakage([train,val,target,test])
    return (train,val,target,test), {'before_final_purge':before, 'after_final_purge':[len(f) for f in (train,val,target,test)],
                                   'source_training_after_test_start': bool(len(train) and len(test) and train.commit_date.max() > test.commit_date.min()),
                                   'chronology_scope':'strict source train<validation and target<test; cross-project test is historical LOPO, not forward-in-time evaluation'}


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
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("LOPO source ratios must be positive")
    s = train_ratio + val_ratio
    tr_ratio, va_ratio = train_ratio / s, val_ratio / s
    train, val, _ = temporal_split(rest, tr_ratio, va_ratio, 0.0)
    train, val, held = purge_group_overlap([train, val, held])
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
    if not 0 < target_ratio < 1:
        raise ValueError('target_ratio must be strictly between zero and one')
    target, test, _ = temporal_split(held_df, target_ratio, 1-target_ratio, 0)
    return target, test


def compute_pos_weight(df: pd.DataFrame, label_col: str = "label",
                       cap: float = 50.0) -> float:
    """``pos_weight`` = N_neg / N_pos on the training fold, capped for stability.

    Used in the weighted BCE so the optimiser does not collapse to the
    majority (safe) class under MegaVul's ~1:19 imbalance.
    """
    pos = int((df[label_col] == 1).sum())
    neg = int((df[label_col] == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("Supervised source training requires both classes")
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
            "sample_index":   torch.tensor(idx, dtype=torch.long),
            "label":          torch.tensor(int(row["label"]), dtype=torch.long),
            "project_id":     torch.tensor(self.project_to_id[row["project"]],
                                           dtype=torch.long),
        }


class ClassBalancedBatchSampler(BatchSampler):
    """Build batches containing both classes, oversampling the rare class."""

    def __init__(self, labels: Sequence[int], batch_size: int, seed: int = 0):
        if batch_size < 2:
            raise ValueError("Class-balanced batches require batch_size >= 2")
        labels = np.asarray(labels)
        if not np.isin(labels,[0,1]).all():
            raise ValueError('Sampler requires strictly binary labels')
        labels = labels.astype(int)
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
                      max_length: int = 512, num_workers: int = 0
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
                      max_length: int = 512, num_workers: int = 0,
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
    if set(train_df['project']) & set(test_df['project']) or set(val_df['project']) & set(test_df['project']):
        errors.append('held-out project appears in sources')
    if len(train_df) and len(val_df) and train_df.commit_date.max() >= val_df.commit_date.min():
        errors.append('source chronological violation: train must precede validation')
    assert_no_leakage([train_df,val_df,test_df])
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


def pooled_loader(loaders, *, batch_size=None):
    from torch.utils.data import ConcatDataset
    if not loaders or any(len(loader.dataset) == 0 for loader in loaders):
        raise ValueError('Empty source loader')
    return DataLoader(ConcatDataset([loader.dataset for loader in loaders]),
                      batch_size=batch_size or loaders[0].batch_size or loaders[0].batch_sampler.batch_size,
                      shuffle=True, num_workers=0)


def sampling_budget(loaders, steps):
    return [{'environment':i, 'unique_population':len(l.dataset), 'batches_per_cycle':len(l),
             'batches_per_epoch':steps, 'iterator_restarts':max(0,(steps-1)//len(l)),
             'sampling':'class-balanced independent class draws' if isinstance(l.batch_sampler,ClassBalancedBatchSampler) else 'shuffle without replacement per cycle',
             'expected_draws_upper_bound':steps*(l.batch_size or l.batch_sampler.batch_size)}
            for i,l in enumerate(loaders)]
