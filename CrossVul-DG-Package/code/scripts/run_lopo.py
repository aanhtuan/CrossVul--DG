"""Run one leave-one-project-out (LOPO) fold.

Pipeline for a fold:
  1. Build the fold: train/val from the 9 source projects (chronological),
     held-out project split into an (earlier) DA-target portion and a (later)
     test portion. ALL methods are scored on the SAME later test portion.
  2. pos_weight = N_neg/N_pos on the training fold (shared by every method).
  3. Train the chosen method (model selection on val PR-AUC).
  4. Tune the decision threshold on val (max-F1), then report test metrics at
     that threshold plus threshold-free PR-AUC / ROC-AUC / recall@precision.

Usage::

    python scripts/run_lopo.py --config configs/default.yaml \
        --method cc-dro --held_out linux --seed 0 \
        --data_path data/processed/top10.parquet \
        --tail_path data/processed/tail.parquet \
        --output_dir runs/codebert/cc-dro/linux/seed0
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer  # type: ignore[import]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    build_dataloaders, build_env_loaders, compute_pos_weight, load_processed,
    lopo_fold, split_held_out, validate_lopo_frames,
)
from src.evaluate import compute_metrics, tune_threshold
from src.models import EncoderClassifier
from src.trainers import TrainerConfig, build_trainer
from src.utils import apply_overrides, get_logger, load_config, save_json, set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--method", required=True,
                    choices=["erm", "irm", "dro", "cc-dro", "pd-dro",
                             "da-coral", "da-dann"])
    ap.add_argument("--held_out", required=True, help="held-out project name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_path", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--tail_path", type=Path, default=None,
                    help="parquet of tail projects (needed for pd-dro)")
    ap.add_argument("--backbone", default=None,
                    help="override encoder.name (codebert/unixcoder/graphcodebert HF id)")
    ap.add_argument("--override", nargs="*", default=[])
    return ap.parse_args()


def main():
    args = parse_args()
    log = get_logger()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.backbone:
        cfg["encoder"]["name"] = args.backbone
    set_seed(args.seed)

    df = load_processed(args.data_path)
    log.info(f"Total samples: {len(df)}  projects: {df['project'].nunique()}")

    train_df, val_df, held_df = lopo_fold(
        df, args.held_out,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )
    # Split held-out project: earlier -> DA target (unlabelled), later -> test.
    target_ratio = cfg["da"].get("target_ratio", 0.5)
    da_target_df, test_df = split_held_out(held_df, target_ratio=target_ratio)
    source_stats = validate_lopo_frames(
        train_df, val_df, test_df, require_source_cells=args.method == "cc-dro")
    log.info(f"train={len(train_df)} val={len(val_df)} "
             f"da_target={len(da_target_df)} test={len(test_df)} "
             f"(held_out={args.held_out})")

    # pos_weight from the training fold, shared by all methods.
    pos_weight = compute_pos_weight(train_df)
    log.info(f"pos_weight (N_neg/N_pos, capped) = {pos_weight:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["name"])

    _, val_loader, test_loader, project_to_id = build_dataloaders(
        train_df, val_df, test_df, tokenizer,
        batch_size=cfg["eval"]["batch_size"],
        max_length=cfg["encoder"]["max_length"],
        num_workers=cfg["train"]["num_workers"],
    )

    env_loaders = build_env_loaders(
        train_df, tokenizer,
        batch_size=cfg["train"]["batch_size"],
        max_length=cfg["encoder"]["max_length"],
        num_workers=cfg["train"]["num_workers"],
        project_to_id=project_to_id,
        class_balanced=(args.method == "cc-dro" and
                        cfg["cc_dro"].get("balanced_batches", True)),
        seed=args.seed,
    )
    log.info(f"Built {len(env_loaders)} environment loaders.")

    model = EncoderClassifier(
        model_name=cfg["encoder"]["name"],
        hidden_size=cfg["encoder"]["hidden_size"],
        dropout=cfg["encoder"]["dropout"],
    )

    # Pseudo-domains (pd-dro): cluster tail projects, add as extra environments.
    if args.method == "pd-dro":
        if args.tail_path is None:
            raise SystemExit("--tail_path required for pd-dro")
        from src.pseudo_domains import cluster_tail_projects
        tail_df = load_processed(args.tail_path)
        # Preserve temporal realism: auxiliary domains must not contain code
        # committed after the latest source-training sample in this fold.
        source_cutoff = train_df["commit_date"].max()
        tail_df = tail_df[tail_df["commit_date"] <= source_cutoff].reset_index(drop=True)
        if tail_df["project"].nunique() < cfg["pseudo"]["n_clusters"]:
            raise ValueError(
                "Not enough temporally eligible tail projects for pd-dro: "
                f"{tail_df['project'].nunique()} projects for "
                f"{cfg['pseudo']['n_clusters']} clusters")
        log.info(f"Clustering {tail_df['project'].nunique()} tail projects "
                 f"into {cfg['pseudo']['n_clusters']} pseudo-domains.")
        tail_map = cluster_tail_projects(
            model, tail_df, tokenizer,
            n_clusters=cfg["pseudo"]["n_clusters"],
            max_samples_per_proj=cfg["pseudo"]["embedding_subset"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        tail_df["project"] = tail_df["project"].map(
            lambda p: f"cluster_{tail_map[p]}" if p in tail_map else p)
        train_df = pd.concat([train_df, tail_df], ignore_index=True)
        project_to_id = {p: i for i, p in enumerate(sorted(train_df["project"].unique()))}
        env_loaders = build_env_loaders(
            train_df, tokenizer,
            batch_size=cfg["train"]["batch_size"],
            max_length=cfg["encoder"]["max_length"],
            num_workers=cfg["train"]["num_workers"],
            project_to_id=project_to_id,
        )
        pos_weight = compute_pos_weight(train_df)
        log.info(f"After pseudo-domains: {len(env_loaders)} envs, "
                 f"pos_weight={pos_weight:.2f}")

    effective_pos_weight = (
        pos_weight if args.method != "cc-dro" or
        cfg["cc_dro"].get("use_pos_weight", False) else 1.0
    )
    if effective_pos_weight != pos_weight:
        log.info("CC-DRO uses unweighted within-cell BCE; class balance is handled by cells.")

    tcfg = TrainerConfig(
        method=args.method,
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_steps=cfg["train"].get("warmup_steps", 0),
        grad_clip=cfg["train"]["grad_clip"],
        early_stopping_patience=cfg["train"]["early_stopping_patience"],
        monitor=cfg["train"].get("monitor", "pr_auc"),
        pos_weight=effective_pos_weight,
        irm_lambda_max=cfg["irm"]["lambda_max"],
        irm_warmup_epochs=cfg["irm"]["warmup_epochs"],
        dro_eta_q=cfg["dro"]["eta_q"],
        cc_eta_q=cfg["cc_dro"]["eta_q"],
        cc_lambda_inv=cfg["cc_dro"]["lambda_inv"],
        cc_warmup_epochs=cfg["cc_dro"]["warmup_epochs"],
        coral_weight=cfg["da"]["coral_weight"],
        dann_lambda=cfg["da"]["dann_lambda"],
        checkpoint_path=str(
            cfg["train"].get("checkpoint_path")
            or (args.output_dir / "checkpoint.pt")
        ),
        resume=cfg["train"].get("resume", True),
    )

    # DA baselines: build an unlabelled target loader from the EARLIER portion
    # of the held-out project (not the test portion).
    target_loader = None
    if args.method.startswith("da-"):
        _, target_loader, _, _ = build_dataloaders(
            da_target_df, da_target_df, da_target_df, tokenizer,
            batch_size=cfg["train"]["batch_size"],
            max_length=cfg["encoder"]["max_length"],
            num_workers=cfg["train"]["num_workers"],
        )

    trainer = build_trainer(args.method, model, tcfg,
                            target_loader=target_loader,
                            n_source_domains=len(env_loaders))
    trainer.fit(env_loaders, val_loader)

    # Tune the decision threshold on val, then evaluate test at that threshold.
    yv, pv, _ = trainer.predict(val_loader)
    best_thr, best_val_f1 = tune_threshold(yv, pv, objective="f1")
    log.info(f"Tuned threshold on val: {best_thr:.4f} (val F1={best_val_f1:.4f})")

    test_metrics = trainer.evaluate(test_loader, threshold=best_thr, project_ids=True)
    val_metrics = trainer.evaluate(val_loader, threshold=best_thr, project_ids=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json({
        "args": vars(args) | {"backbone": cfg["encoder"]["name"],
                              "train_pos_weight": pos_weight,
                              "effective_pos_weight": effective_pos_weight,
                              "pos_weight": effective_pos_weight,
                              "tuned_threshold": best_thr,
                              "project_to_id": project_to_id,
                              "source_project_counts": source_stats,
                              "resolved_config": cfg},
        "val": val_metrics,
        "test": test_metrics,
    }, args.output_dir / "results.json")
    log.info(f"Done. held_out={args.held_out}  "
             f"test PR-AUC={test_metrics['pr_auc']:.4f}  "
             f"F1@thr={test_metrics['f1']:.4f}  "
             f"recall@P50={test_metrics['recall_at_p50']:.4f}")


if __name__ == "__main__":
    main()
