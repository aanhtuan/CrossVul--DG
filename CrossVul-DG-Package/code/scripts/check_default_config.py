"""Validate the default config and optional processed-data directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "encoder": ["name", "max_length", "hidden_size", "dropout"],
    "train": [
        "batch_size", "lr", "weight_decay", "epochs", "warmup_steps", "grad_clip",
        "early_stopping_patience", "num_workers", "monitor",
        "checkpoint_path", "resume",
    ],
    "eval": ["batch_size"],
    "irm": ["lambda_max", "warmup_epochs"],
    "dro": ["eta_q"],
    "cc_dro": ["eta_q", "lambda_inv", "warmup_epochs",
               "balanced_batches", "use_pos_weight"],
    "da": ["coral_weight", "dann_lambda", "target_ratio"],
    "pseudo": ["n_clusters", "embedding_subset"],
    "data": ["top_k", "train_ratio", "val_ratio", "test_ratio", "dedup"],
}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def require_number(
    cfg: dict[str, Any],
    path: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    node: Any = cfg
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    if not isinstance(node, (int, float)) or isinstance(node, bool):
        fail(f"{path} must be a number", errors)
        return
    if minimum is not None and node < minimum:
        fail(f"{path} must be >= {minimum}", errors)
    if maximum is not None and node > maximum:
        fail(f"{path} must be <= {maximum}", errors)


def validate_config(path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        return {}, [], ["Config root must be a mapping"]

    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            fail(f"Missing section: {section}", errors)
            continue
        for key in keys:
            if key not in cfg[section]:
                fail(f"Missing key: {section}.{key}", errors)

    require_number(cfg, "encoder.max_length", errors, minimum=1)
    require_number(cfg, "encoder.hidden_size", errors, minimum=1)
    require_number(cfg, "encoder.dropout", errors, minimum=0.0, maximum=1.0)
    require_number(cfg, "train.batch_size", errors, minimum=1)
    require_number(cfg, "train.lr", errors, minimum=0.0)
    require_number(cfg, "train.epochs", errors, minimum=1)
    require_number(cfg, "train.early_stopping_patience", errors, minimum=1)
    require_number(cfg, "eval.batch_size", errors, minimum=1)
    require_number(cfg, "data.top_k", errors, minimum=1)

    ratios = [
        cfg.get("data", {}).get("train_ratio"),
        cfg.get("data", {}).get("val_ratio"),
        cfg.get("data", {}).get("test_ratio"),
    ]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in ratios):
        total = float(sum(ratios))
        if abs(total - 1.0) > 1e-6:
            fail(f"data split ratios must sum to 1.0, got {total:.8f}", errors)
    else:
        fail("data train/val/test ratios must be numbers", errors)

    target_ratio = cfg.get("da", {}).get("target_ratio")
    if not isinstance(target_ratio, (int, float)) or not 0.0 < target_ratio < 1.0:
        fail("da.target_ratio must be between 0 and 1", errors)

    monitor = cfg.get("train", {}).get("monitor")
    if monitor not in {"pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"}:
        warnings.append(f"train.monitor={monitor!r} may not be supported by compute_metrics")

    checkpoint_path = cfg.get("train", {}).get("checkpoint_path")
    if checkpoint_path is not None and not isinstance(checkpoint_path, str):
        fail("train.checkpoint_path must be null or string", errors)
    if cfg.get("train", {}).get("resume") is not True:
        warnings.append("train.resume is not true; interrupted Colab runs will not resume")

    return cfg, warnings, errors


def validate_processed_dir(path: Path, errors: list[str], warnings: list[str]) -> None:
    top10 = path / "top10.parquet"
    tail = path / "tail.parquet"
    splits = path / "splits.json"

    for required in [top10, tail, splits]:
        if not required.exists():
            fail(f"Missing processed file: {required}", errors)

    if splits.exists():
        with splits.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        top_counts = data.get("top_project_counts", {})
        if not isinstance(top_counts, dict) or not top_counts:
            fail(f"{splits} missing non-empty top_project_counts", errors)
        else:
            print("Projects from splits.json:")
            for project, count in top_counts.items():
                print(f"  {project}: {count}")
            if len(top_counts) != 10:
                warnings.append(f"top_project_counts has {len(top_counts)} projects, not 10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--processed_dir", type=Path, default=None)
    args = parser.parse_args()

    warnings: list[str]
    errors: list[str]
    _, warnings, errors = validate_config(args.config)

    if args.processed_dir is not None:
        validate_processed_dir(args.processed_dir, errors, warnings)

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nConfig check FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nConfig check PASSED.")


if __name__ == "__main__":
    main()
