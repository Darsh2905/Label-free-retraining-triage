"""Helpers shared by the numbered scripts."""
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import dump_config  # noqa: E402


def new_run_dir(tag: str) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "results", f"run_{ts}_{tag}")
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    return d


def snapshot(cfg: dict, run_dir: str, **extra) -> None:
    snap = dict(cfg)
    snap["_run"] = {"argv": sys.argv, **extra}
    dump_config(snap, os.path.join(run_dir, "config_snapshot.yaml"))


def write_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def parse_sets(pairs: list[str] | None) -> dict:
    """--set a.b=1 --set c=[1,2]  ->  nested override dict (values parsed as YAML)."""
    import yaml
    out: dict = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        cur = out
        parts = k.split(".")
        for q in parts[:-1]:
            cur = cur.setdefault(q, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return out
