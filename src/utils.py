"""Shared utilities: seeding and layered YAML config loading."""
from __future__ import annotations

import copy
import os
import random
from typing import Any

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILES = ["base.yaml", "observer.yaml", "detection.yaml", "policy.yaml"]


def set_seed(seed: int) -> None:
    """Seed python and numpy. LightGBM gets `seed` passed explicitly via its params."""
    random.seed(seed)
    np.random.seed(seed)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(overlay: str | None = None, overrides: dict | None = None,
                config_dir: str | None = None) -> dict[str, Any]:
    """Merge configs/{base,observer,detection,policy}.yaml, then an optional dataset
    overlay (e.g. configs/quic.yaml), then explicit overrides."""
    config_dir = config_dir or os.path.join(ROOT, "configs")
    cfg: dict[str, Any] = {}
    for name in CONFIG_FILES:
        with open(os.path.join(config_dir, name)) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    cfg["policies"] = {}
    pdir = os.path.join(config_dir, "policies")
    for name in sorted(os.listdir(pdir)):
        if name.endswith(".yaml"):
            with open(os.path.join(pdir, name)) as f:
                cfg["policies"][name[:-5]] = yaml.safe_load(f)
    if overlay:
        path = overlay if os.path.isabs(overlay) or os.path.exists(overlay) else os.path.join(config_dir, overlay)
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def dump_config(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
