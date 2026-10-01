"""Stream CESNET datasets from Zenodo and keep a label-agnostic uniform sample per day.

Why this exists instead of `cesnet-datazoo`'s own downloader: the datazoo host
(liberouter.org) accepted connections but never sent data during development, while
the same datasets' original releases on Zenodo were reachable. Zenodo serves one zip
per dataset (30.5 GB TLS-Year22, 21 GB QUIC22), with one compressed CSV per day inside.

Design decisions (all deliberate, see README "Data acquisition"):
- Rows inside a day file are time-ordered, so reading only the start of a file would
  sample the night hours. We therefore stream each day file completely and apply
  Bernoulli sampling to every flow, which keeps a uniform sample of the whole day.
- Sampling is label-agnostic (a per-flow coin flip, not a per-service cap). A
  per-service cap would read labels on the operational stream and would also flatten
  the class prior, destroying the prior-shift signal the predicted-class divergence
  monitor exists to detect.
- The per-day sampling rate p_d = min(1, target / total_d) is stored so the true
  daily volume V can be recovered (total_d also comes from the day's stats json).
- Identifying columns (IPs, SNI, JA3, ASN, ports) are dropped: SNI is the labelling
  source and would leak the label into the features.
"""
from __future__ import annotations

import io
import json
import lzma
import os
import struct
import time
import zipfile
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

ZENODO = {
    "tls-year22": "https://zenodo.org/api/records/10608607/files/CESNET-TLS-Year22.zip/content",
    "quic22": "https://zenodo.org/api/records/7963302/files/cesnet-quic22.zip/content",
}

SCALAR_COLS = [
    "DURATION", "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV",
    "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS",
    "FLAG_CWR", "FLAG_CWR_REV", "FLAG_ECE", "FLAG_ECE_REV", "FLAG_URG", "FLAG_URG_REV",
    "FLAG_ACK", "FLAG_ACK_REV", "FLAG_PSH", "FLAG_PSH_REV", "FLAG_RST", "FLAG_RST_REV",
    "FLAG_SYN", "FLAG_SYN_REV", "FLAG_FIN", "FLAG_FIN_REV",
    "FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_END", "FLOW_ENDREASON_OTHER",
]
PHIST_COLS = ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"]
KEEP_META = ["APP", "TIME_FIRST"]


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP range requests (for zipfile)."""

    def __init__(self, url: str):
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.url = r.url
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120).content
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


@dataclass
class DayEntry:
    date: str             # YYYYMMDD
    flows_name: str
    stats_name: str | None
    header_offset: int
    compress_size: int


def list_days(dataset: str, cache_dir: str) -> tuple[str, list[DayEntry]]:
    """Read the zip central directory remotely (cached to disk) and return per-day entries."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "zip_index.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        return d["url"], [DayEntry(**e) for e in d["days"]]
    rf = _RangeFile(ZENODO[dataset])
    z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    stats = {os.path.basename(i.filename)[6:14]: i.filename for i in z.infolist()
             if os.path.basename(i.filename).startswith("stats-") and len(os.path.basename(i.filename)) >= 19}
    days = []
    for i in z.infolist():
        base = os.path.basename(i.filename)
        if base.startswith("flows-") and (base.endswith(".csv.xz") or base.endswith(".csv.gz")):
            date = base[6:14]
            days.append(DayEntry(date, i.filename, stats.get(date), i.header_offset, i.compress_size))
    days.sort(key=lambda e: e.date)
    json.dump({"url": rf.url, "days": [e.__dict__ for e in days]}, open(cache, "w"), indent=1)
    return rf.url, days


def _data_offset(url: str, header_offset: int) -> int:
    h = requests.get(url, headers={"Range": f"bytes={header_offset}-{header_offset + 29}"}, timeout=60).content
    n, e = struct.unpack("<HH", h[26:30])
    return header_offset + 30 + n + e


def _read_small_member(url: str, entry_name: str) -> bytes:
    rf = _RangeFile(url)
    z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
    return z.read(entry_name)


class _Inner:
    """Inner decompressor for .xz or .gz (handles multi-member gzip)."""

    def __init__(self, kind: str):
        self.kind = kind
        self.d = lzma.LZMADecompressor() if kind == "xz" else zlib.decompressobj(16 + zlib.MAX_WBITS)

    def feed(self, b: bytes) -> bytes:
        if not b or (self.kind == "xz" and self.d.eof):
            return b""
        out = self.d.decompress(b)
        if self.kind == "gz":
            while self.d.unused_data:
                rest = self.d.unused_data
                self.d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                out += self.d.decompress(rest)
        return out


def _parse_sample(header: str, lines: list[str]) -> pd.DataFrame:
    cols = header.split(",")
    usecols = [c for c in SCALAR_COLS + PHIST_COLS + KEEP_META if c in cols]
    df = pd.read_csv(io.StringIO(header + "\n" + "\n".join(lines)), usecols=usecols)
    out = {}
    for c in SCALAR_COLS:
        if c in df:
            s = df[c]
            if s.dtype == object or s.dtype == bool or str(s.dtype) in ("str", "string"):
                s = s.astype(str).map({"True": 1, "False": 0, "1": 1, "0": 0})
            out[c] = s.astype(np.float32)
    for c in PHIST_COLS:
        if c in df:
            arr = np.array([list(map(int, v.strip("[]").split(","))) for v in df[c].astype(str)], dtype=np.float32)
            for k in range(arr.shape[1]):
                out[f"{c}_{k}"] = arr[:, k]
    res = pd.DataFrame(out)
    res["APP"] = df["APP"].astype(str).values
    res["HOUR"] = pd.to_datetime(df["TIME_FIRST"], errors="coerce").dt.hour.fillna(-1).astype(np.int8).values
    return res


def fetch_day(url: str, entry: DayEntry, out_dir: str, target_per_day: int, seed: int,
              chunk: int = 1 << 20, max_retries: int = 20) -> str:
    """Stream one day, Bernoulli-sample flows, write <out_dir>/<date>.parquet (+ .json meta)."""
    out_path = os.path.join(out_dir, f"{entry.date}.parquet")
    if os.path.exists(out_path):
        return out_path
    for attempt in range(max_retries):
        try:
            stats = json.loads(_read_small_member(url, entry.stats_name)) if entry.stats_name else {}
            break
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(600, 15 * 2 ** attempt))
    total = int(stats.get("global", {}).get("total-saved", 0)) or None
    # If the stats file lacks a total, fall back to a conservative rate and record the realised count.
    p = min(1.0, target_per_day / total) if total else 0.05
    kind = "xz" if entry.flows_name.endswith(".xz") else "gz"
    rng = np.random.default_rng([seed, int(entry.date)])
    for attempt in range(max_retries):
        try:
            off = _data_offset(url, entry.header_offset)
            r = requests.get(url, headers={"Range": f"bytes={off}-{off + entry.compress_size - 1}"},
                             stream=True, timeout=300)
            r.raise_for_status()
            outer = zlib.decompressobj(-zlib.MAX_WBITS)   # zip member is deflated
            inner = _Inner(kind)
            header, rem, kept, n_seen = None, "", [], 0
            for raw in r.iter_content(chunk):
                text = rem + inner.feed(outer.decompress(raw)).decode("utf-8", errors="replace")
                lines = text.split("\n")
                rem = lines.pop()
                if header is None and lines:
                    header = lines.pop(0)
                if lines:
                    n_seen += len(lines)
                    mask = rng.random(len(lines)) < p
                    kept.extend(l for l, m in zip(lines, mask) if m)
            tail = rem + inner.feed(outer.flush()).decode("utf-8", errors="replace")
            for l in tail.split("\n"):
                if l.strip():
                    n_seen += 1
                    if rng.random() < p:
                        kept.append(l)
            break
        except Exception as ex:  # network hiccup: restart this day from scratch with a fresh RNG
            rng = np.random.default_rng([seed, int(entry.date)])
            if attempt == max_retries - 1:
                raise
            time.sleep(min(600, 15 * 2 ** attempt))   # patient: survives network outages
            print(f"[{entry.date}] retry {attempt + 1}: {ex!r}", flush=True)
    df = _parse_sample(header, kept)
    tmp = out_path + ".tmp"
    df.to_parquet(tmp, index=False)
    meta = {"date": entry.date, "rows_in_file": n_seen, "total_saved_stats": total,
            "sample_rate": p, "n_sampled": len(df), "app_counts_full_day": stats.get("apps", {})}
    json.dump(meta, open(os.path.join(out_dir, f"{entry.date}.json"), "w"))
    os.replace(tmp, out_path)
    return out_path
