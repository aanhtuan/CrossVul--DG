"""Utility helpers: seeding, config loading, logging."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply CLI overrides of the form ``a.b.c=value`` to a nested dict."""
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Bad override {kv!r}, expected key=value")
        key, value = kv.split("=", 1)
        value = yaml.safe_load(value)
        if isinstance(value,str):
            try: value=float(value) if any(c in value.lower() for c in ('.','e')) else int(value)
            except ValueError: pass
        keys = key.split('.')
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k],dict):
                raise ValueError(f'Unknown config key {key!r}')
            node = node[k]
        if keys[-1] not in node:
            raise ValueError(f'Unknown config key {key!r}')
        node[keys[-1]] = value
    return cfg


def get_logger(name: str = "crossvul", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(value):
        if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [clean(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,Path): return str(value)
        return value
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(clean(obj),indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
