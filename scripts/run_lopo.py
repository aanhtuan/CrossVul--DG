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
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import argparse
from pathlib import Path
import sys
import time
import resource

import pandas as pd
import torch
from transformers import AutoTokenizer  # type: ignore[import]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crossvul_dg.data import (
    build_dataloaders, build_env_loaders, compute_pos_weight, load_processed,
    make_lopo_partitions, validate_lopo_frames, _identifiers, iter_metadata,
)
from crossvul_dg.evaluate import compute_metrics, tune_threshold
from crossvul_dg.provenance import (verify_processed,experiment_identity,software_provenance,ensure_result_identity,signature)
from crossvul_dg.models import EncoderClassifier
from crossvul_dg.config import validate
from crossvul_dg.resources import require_headroom
from crossvul_dg.resources import training_preflight
from crossvul_dg.cohorts import bounded_frames,materialize
from crossvul_dg.trainers import TrainerConfig, build_trainer
from crossvul_dg.utils import apply_overrides, get_logger, load_config, save_json, set_seed


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
    for name in ('train_per_project','val_limit','target_limit','test_limit'):
        ap.add_argument('--'+name,type=int,default=0,help='Explicit compute-budget cap; zero keeps full partition')
    ap.add_argument('--selection_only',action='store_true',help='Write validation selection artifact without any test forward pass')
    return ap.parse_args()


def main():
    args = parse_args()
    started=time.monotonic()
    log = get_logger()

    cfg = load_config(args.config)
    original_backbone = cfg['encoder']['name']
    cfg = apply_overrides(cfg, args.override)
    cfg['compute_budget']={name:getattr(args,name) for name in ('train_per_project','val_limit','target_limit','test_limit')}
    if args.backbone:
        cfg["encoder"]["name"] = args.backbone
    if cfg['encoder']['name'] != original_backbone:
        from huggingface_hub import model_info
        cfg['encoder']['revision'] = model_info(cfg['encoder']['name']).sha
    errors = validate(cfg)
    if errors:
        raise ValueError('; '.join(errors))
    if cfg['train']['device']=='cuda' and not torch.cuda.is_available():
        raise ValueError('Configured CUDA is unavailable; choose train.device=cpu or install appropriate PyTorch')
    set_seed(args.seed)

    torch.set_num_threads(1)
    data_provenance = verify_processed(args.data_path)
    tail_provenance = verify_processed(args.tail_path) if args.method=='pd-dro' and args.tail_path else None
    software = software_provenance()
    experiment_sig, comparison_sig = experiment_identity(cfg,args.method,args.held_out,args.seed,data_provenance,tail_provenance)
    # Any executable-source change invalidates reusable checkpoints/results.
    runtime_sig=signature({'packages':software['packages'],'python':software['python']})
    experiment_sig = signature({'experiment':experiment_sig,'source':software['source_signature'],'runtime':runtime_sig})
    comparison_sig=signature({'comparison':comparison_sig,'runtime':runtime_sig})
    if ensure_result_identity(args.output_dir/'results.json',experiment_sig):
        log.info('Matching completed experiment verified; skipping training')
        return
    resource_snapshot = training_preflight(cfg) if cfg['train']['device']=='cuda' else require_headroom()
    if cfg['train']['device']=='cuda':torch.cuda.reset_peak_memory_stats()
    bounded=any(cfg['compute_budget'].values())
    df = load_processed(args.data_path,metadata_only=bounded)
    top_project_names=set(df.project)
    log.info(f"Total samples: {len(df)}  projects: {df['project'].nunique()}")
    (train_df,val_df,da_target_df,test_df), split_audit = make_lopo_partitions(
        df,args.held_out,cfg['data']['train_ratio'],cfg['data']['val_ratio'],cfg['da']['target_ratio'])
    split_audit['original_rows']=len(df)
    split_audit['retained_rows']=sum(len(x) for x in (train_df,val_df,da_target_df,test_df))
    (train_df,val_df,da_target_df,test_df),cohort_audit=bounded_frames(
        [train_df,val_df,da_target_df,test_df],seed=args.seed,**cfg['compute_budget'])
    split_audit['compute_budget']=cohort_audit
    if bounded:
        train_df=materialize(args.data_path,train_df)
        val_df=materialize(args.data_path,val_df)
        test_df=materialize(args.data_path,test_df)
        da_target_df=materialize(args.data_path,da_target_df) if args.method.startswith('da-') else da_target_df.assign(func='')
    del df
    split_audit['partition_counts']={name:{'n':len(x),'positive':int(x.label.sum()),
                                      'hash':signature(sorted(x.code_hash.tolist()))}
                                  for name,x in zip(('train','validation','target','test'),(train_df,val_df,da_target_df,test_df))}
    source_stats=validate_lopo_frames(train_df,val_df,test_df,require_source_cells=args.method=='cc-dro')
    if args.method.startswith('da-') and da_target_df.empty:
        raise ValueError('Empty DA target partition after leakage purging')
    log.info(f'train={len(train_df)} val={len(val_df)} target={len(da_target_df)} test={len(test_df)}')

    # pos_weight from the training fold, shared by all methods.
    pos_weight = compute_pos_weight(train_df)
    log.info(f"pos_weight (N_neg/N_pos, capped) = {pos_weight:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["name"], revision=cfg["encoder"]["revision"])

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
        revision=cfg['encoder']['revision'],
        hidden_size=cfg["encoder"]["hidden_size"],
        dropout=cfg["encoder"]["dropout"],
        gradient_checkpointing=cfg['encoder'].get('gradient_checkpointing',False),
    )

    auxiliary_audit = None
    # Pseudo-domains (pd-dro): cluster tail projects, add as extra environments.
    if args.method == "pd-dro":
        if args.tail_path is None:
            raise SystemExit("--tail_path required for pd-dro")
        from crossvul_dg.pseudo_domains import cluster_tail_projects
        tail_df = load_processed(args.tail_path)
        if set(tail_df.project) & top_project_names:
            raise ValueError('Tail projects must be disjoint from top-K projects')
        protected=set().union(*(_identifiers(row) for f in (train_df,val_df,da_target_df,test_df) for row in iter_metadata(f)))
        tail_df=tail_df.loc[[not bool(_identifiers(row)&protected) for row in iter_metadata(tail_df)]].reset_index(drop=True)
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
            device=cfg['train'].get('device','cpu'),
            max_length=cfg['encoder']['max_length'],batch_size=cfg['eval']['batch_size'],
        )
        auxiliary_audit={'additional_labeled_rows':len(tail_df),'tail_project_clusters':tail_map,
                         'code_hash':signature(sorted(tail_df.code_hash.tolist())),
                         'comparison_status':'uses additional labelled tail data; not a matched-supervision DG baseline'}
        tail_df["project"] = tail_df["project"].map(
            lambda p: f"cluster_{tail_map[p]}" if p in tail_map else p)
        train_df = pd.concat([train_df, tail_df], ignore_index=True)
        for project in sorted(train_df.project.unique()):
            if project not in project_to_id:
                project_to_id[project]=len(project_to_id)
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
        device=cfg['train'].get('device','cpu'),
        experiment_signature=experiment_sig,
        amp=cfg['train'].get('amp',False),
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
            or (args.output_dir / "latest.pt")
        ),
        resume=cfg["train"].get("resume", True),
    )

    # DA baselines: build an unlabelled target loader from the EARLIER portion
    # of the held-out project (not the test portion).
    target_loader = None
    if args.method.startswith("da-"):
        # Remove target labels before constructing any loader reachable by DA.
        unlabelled_target = da_target_df.assign(label=0)
        target_loader, _, _, _ = build_dataloaders(
            unlabelled_target, unlabelled_target, unlabelled_target, tokenizer,
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

    val_metrics = trainer.evaluate(val_loader, threshold=best_thr, project_ids=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload={
        'status':'validation_complete' if args.selection_only else 'completed','result_version':2,'experiment_signature':experiment_sig,
        'comparison_signature':comparison_sig,'data_provenance':data_provenance,
        'software':software,'tail_data_provenance':tail_provenance,'split_audit':split_audit,'auxiliary_data':auxiliary_audit,
        'regime':'DA' if args.method.startswith('da-') else 'DG+additional-labelled-tail' if args.method=='pd-dro' else 'DG',
        'resource_snapshot':resource_snapshot,'training_history':trainer.history,'sampling_budget':trainer.sampling,'schedule':trainer.schedule,
        'args':vars(args)|{'backbone':cfg['encoder']['name'],'train_pos_weight':pos_weight,'effective_pos_weight':effective_pos_weight,
                           'pos_weight':effective_pos_weight,'tuned_threshold':best_thr,'project_to_id':project_to_id,
                           'source_project_counts':source_stats,'resolved_config':cfg},'val':val_metrics}
    def measured():
        return dict(runtime_seconds=time.monotonic()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if cfg['train']['device']=='cuda' else None,
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if cfg['train']['device']=='cuda' else None)
    if args.selection_only:
        payload['measurements']=measured()
        save_json(payload,args.output_dir/'selection.json')
        log.info('Validation-only artifact saved; test forward pass NOT RUN')
        return
    precision_thr, _ = tune_threshold(yv,pv,objective='recall_at_precision',min_precision=.5)
    test_y,test_p,test_g=trainer.predict(test_loader)
    test_metrics=compute_metrics(test_y,test_p,best_thr,group=test_g)
    test_metrics['validation_precision_floor_threshold']=precision_thr
    test_metrics['recall_at_validation_p50_threshold']=compute_metrics(test_y,test_p,precision_thr)['recall']
    test_metrics['precision_at_validation_p50_threshold']=compute_metrics(test_y,test_p,precision_thr)['precision']
    payload['test']=test_metrics;payload['measurements']=measured()
    save_json(payload,args.output_dir/'results.json')
    log.info(f"Done. held_out={args.held_out}  "
             f"test PR-AUC={test_metrics['pr_auc']:.4f}  "
             f"F1@thr={test_metrics['f1']:.4f}  "
             f"recall@P50={test_metrics['recall_at_p50']:.4f}")


if __name__ == "__main__":
    main()
