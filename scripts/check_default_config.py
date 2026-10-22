"""Validate the default config and optional processed-data directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from crossvul_dg.config import validate


def fail(message, errors): errors.append(message)


def validate_config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    return cfg,[],validate(cfg)


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
