#!/usr/bin/env bash
# Keeps the (resumable) fetch alive until both datasets are complete. Logs restarts.
cd "$(dirname "$0")/.."
need() { # $1 dataset -> prints number of non-empty days still missing
  .venv/bin/python - "$1" <<'PY'
import json, os, sys
ds = sys.argv[1]; base = f"data/{ds}"
idx = json.load(open(f"{base}/zip_index.json"))["days"] if os.path.exists(f"{base}/zip_index.json") else None
if idx is None: print(999); sys.exit()
print(sum(1 for e in idx if e["compress_size"] > 1000 and not os.path.exists(f"{base}/days/{e['date']}.parquet")))
PY
}
while true; do
  t=$(need tls-year22); q=$(need quic22)
  if [ "$t" = "0" ] && [ "$q" = "0" ]; then echo "$(date) all data complete"; break; fi
  if ! pgrep -f "01_fetch_data.py" >/dev/null; then
    echo "$(date) fetch not running (tls missing=$t, quic missing=$q) -> restarting"
    if [ "$t" != "0" ]; then ds=tls-year22; else ds=quic22; fi
    nohup .venv/bin/python scripts/01_fetch_data.py --dataset $ds --workers 6 >> data/fetch_${ds}_watchdog.log 2>&1 &
  fi
  sleep 300
done
