#!/usr/bin/env bash
# Live demo: replays the real 2022 traffic through the real pipeline in a browser dashboard.
# Needs data/tls-year22/ (scripts/01_fetch_data.py) and results/cache/ (any earlier replay run) -- see README "Live demo".
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
exec $PY demo/server.py --open "$@"
