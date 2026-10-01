#!/usr/bin/env bash
# Reproduce every reported result from a clean checkout. Exits non-zero on any failure.
# Data fetch (step 1) is long (~2.5 h TLS + ~2 h QUIC at ~2 MB/s) and resumable.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2 3 4"}

echo "== tests (label-leakage tests run first) =="
$PY -m pytest

echo "== 1. data =="
$PY scripts/01_fetch_data.py --dataset tls-year22
$PY scripts/01_fetch_data.py --dataset quic22

echo "== 2. reference classifier + P0 degradation =="
$PY scripts/02_train_reference_classifier.py | tee /tmp/ref.log
REF=$(grep 'results ->' /tmp/ref.log | awk '{print $3}')

echo "== 3-4. D(t) + go/no-go correlation checkpoint =="
$PY scripts/02b_signal_checkpoint.py

echo "== 5-7. full replay, all policies, sweeps, 5 seeds =="
$PY scripts/03_run_replay.py --config configs/base.yaml --seeds $SEEDS | tee /tmp/replay.log
RUN=$(grep 'results ->' /tmp/replay.log | tail -1 | awk '{print $3}')

echo "== 8. QUIC cross-protocol validation =="
$PY scripts/04_run_quic_validation.py --seeds $SEEDS

echo "== 9. figures =="
$PY scripts/05_make_report_figures.py "$RUN" --per-service "$REF/p0_per_service_f1.csv"
