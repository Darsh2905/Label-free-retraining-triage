"""Fetch and uniformly subsample CESNET-TLS-Year22 / CESNET-QUIC22 from Zenodo.

Usage:
    python scripts/01_fetch_data.py --dataset tls-year22 --target-per-day 100000 --workers 6
    python scripts/01_fetch_data.py --dataset quic22 --target-per-day 100000 --workers 6

Resumable: days already written to data/<dataset>/days/ are skipped.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.fetch import fetch_day, list_days  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tls-year22", "quic22"], required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--target-per-day", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (fixed; data sample is shared by all runs)")
    ap.add_argument("--dates", nargs="*", help="optional subset of YYYYMMDD dates")
    a = ap.parse_args()
    base = os.path.join(a.data_root, a.dataset)
    out = os.path.join(base, "days")
    os.makedirs(out, exist_ok=True)
    url, days = list_days(a.dataset, base)
    if a.dates:
        days = [d for d in days if d.date in set(a.dates)]
    todo = [d for d in days if not os.path.exists(os.path.join(out, f"{d.date}.parquet"))]
    print(f"{a.dataset}: {len(days)} days, {len(todo)} to fetch, "
          f"{sum(d.compress_size for d in todo) / 1e9:.1f} GB compressed", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_day, url, d, out, a.target_per_day, a.seed): d for d in todo}
        for i, f in enumerate(as_completed(futs), 1):
            d = futs[f]
            try:
                f.result()
                print(f"[{i}/{len(todo)}] {d.date} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
            except Exception as ex_:
                print(f"[{i}/{len(todo)}] {d.date} FAILED: {ex_!r}", flush=True)


if __name__ == "__main__":
    main()
