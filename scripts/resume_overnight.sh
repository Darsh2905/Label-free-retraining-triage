#!/usr/bin/env bash
# Resume the data fetch + cache warm-up after a shutdown/restart. Idempotent and safe
# to run any number of times (e.g. fired by a LaunchAgent at every login): skips
# starting anything that's already running, and the fetcher itself skips days already
# on disk while warm_cache reuses results/cache/ (keyed by a hash of the model config).
cd "$(dirname "$0")/.."
mkdir -p data results

echo "$(date) resume_overnight.sh invoked"
echo "TLS days on disk: $(ls data/tls-year22/days/*.parquet 2>/dev/null | wc -l | tr -d ' ') / 357"
echo "QUIC days on disk: $(ls data/quic22/days/*.parquet 2>/dev/null | wc -l | tr -d ' ') / 28"

if pgrep -f "fetch_watchdog.sh" >/dev/null; then
  echo "watchdog already running, not starting a duplicate"
else
  nohup scripts/fetch_watchdog.sh >> data/watchdog.log 2>&1 &
  echo "watchdog started (pid $!)"
fi

if pgrep -f "warm_cache.py" >/dev/null; then
  echo "warm_cache already running, not starting a duplicate"
else
  nohup .venv/bin/python scripts/warm_cache.py --set classifier.num_threads=4 >> results/warm_cache.log 2>&1 &
  echo "warm_cache started (pid $!)"
fi

echo "Both running in background. Check with:"
echo "  tail -f data/fetch_tls-year22_watchdog.log data/fetch_tls2.log"
echo "  tail -f results/warm_cache.log"
