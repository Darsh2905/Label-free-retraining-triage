# Label-free, cost-aware retraining triage for encrypted traffic classifiers

A deployed, frozen traffic classifier (LightGBM, 50 services, CESNET-TLS-Year22) is
monitored **without ground-truth labels**. A composite drift score, change-point
detection and a cost model decide *when* retraining is worth its price. The classifier
is a fixed substrate. The contribution is the monitor-and-decide pipeline and the
harness that evaluates it against periodic, threshold and oracle policies.

> **Status:** see [Results](#results). Every number in this README is printed by a
> script in `scripts/` and stored under `results/`.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -e .     # pinned versions in pyproject.toml
./run_all.sh                                            # tests -> data -> steps 2-9
```

| step | script | output |
|---|---|---|
| tests | `pytest` (leakage tests run first) | 47+ assertions, non-zero exit on regression |
| 1 data | `scripts/01_fetch_data.py --dataset tls-year22 \| quic22` | `data/<ds>/days/YYYYMMDD.parquet` (resumable) |
| 2 reference model, P0 curve | `scripts/02_train_reference_classifier.py` | `p0_weekly.csv`, `p0_per_service_f1.csv` |
| 3-4 D(t) + go/no-go | `scripts/02b_signal_checkpoint.py` | `checkpoint_report.json` |
| 5-7 full replay | `scripts/03_run_replay.py --config configs/base.yaml` | `policy_comparison.csv`, `sensitivity.csv`, `oracle_*.csv`, ... |
| 8 QUIC | `scripts/04_run_quic_validation.py` | `quic_validation.json` |
| 9 figures | `scripts/05_make_report_figures.py <run_dir>` | `figures/*.png`, `summary_table.csv` |

Any config value can be overridden from the command line: `--set replay_end=20 --set 'sweeps.cost_ratio=[1,10]'`.

## Architecture

```
data/fetch.py      Zenodo -> uniform per-day sample      (labels + features)
data/loader.py     class subset, closed world, per-day cap (LABEL-BEARING)
        |  split in evaluation/harness.py
        +--> data/replay.py      StreamDay/WeeklyBatch: features only, forward-only next_week()
        +--> evaluation/sealed_store.py   labels only; caller + clock guarded
classifier/        LightGBM substrate; frozen_state = deployed generation
observer/          8 label-free signals -> composite D (oriented z-scores, per-generation baseline)
detection/         daily D -> STL(7) -> anchored normalisation -> PELT -> k-day persistence
decision/          g (isotonic, frozen) ; myopic rule (spec) ; cumulative rule ; P0-P5 policies
evaluation/        harness (memoised World), metrics, exact DP oracle, plots
pipeline.py        one year-run end to end
```

### How "label-free" is enforced (three layers)
1. **Static import graph** (`tests/test_no_label_leakage.py`, `tests/leakcheck.py`): no module
   in `observer/ detection/ decision/ classifier/ data/replay.py` may reach
   `evaluation/`, `data/loader.py` or `pipeline.py`, directly or transitively. The check
   covers every import form: `from pkg import mod`, relative imports, `importlib`
   and `__import__`. The checker is itself mutation-tested against 9 evasion forms.
2. **Runtime caller guard + clock** (`sealed_store.py`): accessors raise
   `LabelAccessError` unless called from `src.evaluation`/`src.pipeline`/scripts, and never
   return labels for a period after the simulation clock.
3. **Behavioural test**: P3/P4/P4c make *identical* decisions when the grading labels
   are replaced with garbage, while the oracle P5 changes, which proves the patch bites.

`decision/` never imports the store at all (stricter than the spec, which allowed
`calibration.py` to). The harness fetches calibration labels through
`get_calibration_labels` and passes plain arrays to `g.fit`.

## Design decisions that deviate from the spec, and why

Each was decided **before** the full-year results existed. Where a choice could look
tuned, the spec's literal variant is kept as a config switch and reported.

| # | Spec | Here | Why |
|---|---|---|---|
| 1 | per-service cap of 2000 flows/week | uniform, label-agnostic per-day sample | a per-service cap reads labels on the operational stream and flattens the class prior, which hides exactly the prior shift the JS monitor exists to detect |
| 2 | top-50 by whole-year volume | top-50 by **W0** volume (`class_selection: w0`) | whole-year ranking uses future information and removes "a top service fades" drift. `year` available |
| 3 | STL(period=7) on the weekly series | signals computed **daily**, STL(7) on the daily series, decisions weekly | on weekly data a 7-period STL models a meaningless 7-week cycle, and weekly aggregation already averages out the day-of-week cycle (Jančička et al.) |
| 4 | D = equal-weight sum of z-scores | z-scores **sign-oriented**; family-balanced weights (`weighting: equal` available) | max-prob/margin fall under drift while entropy/JS/KS/MMD/reject rise, so the unsigned sum partly cancels. 4 of 8 signals are collinear confidence stats |
| 5 | z-reference fixed on weeks 3-8, no mention of retrains | per-**generation** location (the signal on the model's own held-out training-window data), scale fixed from weeks 3-8 | after a retrain, D against the old model's reference is meaningless; this keeps D comparable across generations so g can transfer |
| 6 | rolling-median/MAD normalisation | `anchored` (calm-window median/MAD); `rolling` available | a trailing median is a high-pass filter: persistent steps vanish after ~W/2 and slow ramps never trigger (asserted in `test_changepoint_detection.py`) |
| 7 | persistence: k consecutive weeks above floor | k=3 **days**, plus "still active" check | weekly k=3 means at least 3 weeks' delay. Without the active check, a reverted transient re-triggers forever |
| 8 | "macro-F1 drop relative to W0's macro-F1" | reference F1 on a **held-out** 20% of the training window, at daily sample size | in-sample W0 F1 would make every week look degraded |
| 9 | isotonic g | isotonic within range + **linear extrapolation** beyond (`g_extrapolation: clip` available) | on the synthetic world, clip priced a D=23 drift at the calibration-window maximum (D~2.5), so P4 could never fire (see "Findings during development") |
| 10 | P4 = spec's myopic rule | P4 kept verbatim, **P4c** (cumulative excess cost) added as a separate policy | the myopic rule compares a *per-week* loss with a *one-off* fix cost: a persistent moderate loss never triggers |
| 11 | P5 threshold oracle = upper bound | P5 kept; **exact DP oracle** frontier added | a threshold on true F1 is not optimal. Because a retrained model depends only on its training week, the (generation x week) F1 table can be computed once, and DP gives the true Pareto frontier and cost-optimal schedule |
| 12 | compare 6 single operating points | every family swept (periodic k, P3 z, P4/P4c cost ratio, P5 target) | each policy has a knob, so single points compare knob settings, not policies |
| 13 | spec leakage test | fixed and extended (layer 1 above) | the spec's AST test checks only `node.module` for `ImportFrom`, so `from src.evaluation import sealed_store` passes it (`test_spec_literal_test_would_miss_from_pkg_import`) |
| 14 | weekly QUIC pipeline | QUIC at **daily** cadence, STL off | QUIC22 has 4 weeks: 4 weekly points cannot support any of STL / 8-week MAD / 3-week persistence |
| 15 | features standardised | log1p on counts, then standardised (W0 only, frozen) | RBF-MMD on raw byte counts is dominated by elephant flows. LightGBM and KS are invariant |

## Data acquisition

The `cesnet-datazoo` host (liberouter.org) accepted TLS connections but never
returned data during development. The same datasets' original releases on Zenodo
(TLS-Year22 30.5 GB, QUIC22 21 GB) were reachable at ~2 MB/s with range requests.
Rows inside each day file are time-ordered, and the xz/gz streams are single-block, so
there is no valid shortcut. `01_fetch_data.py` streams every day file once,
decompresses it on the fly, and Bernoulli-samples each flow with rate
`p_d = min(1, 100k / total_d)`. Integrity check: the rows streamed equal the day's
`stats.json` total (verified on 2022-01-01: 487,081 = 487,081). Identifying columns (IPs,
SNI, JA3, ASN, ports) are dropped: SNI is the label source.

## Classifier

LightGBM multiclass, 100 rounds, lr 0.1, 31 leaves, trained on at most 100k flows of W0
(held-out 20% for reference). With LightGBM defaults, 50-class training at lr=0.1
**diverged**: held-out accuracy 0.16 with mean max-prob 0.9999. Setting
`min_sum_hessian_in_leaf=1, lambda_l2=1` fixes it (acc 0.845, macro-F1 0.844). These
were chosen on W0 only. The trainer now fails loudly on the divergence signature. The
5-class synthetic tests could not have caught this.

## Results

*(filled in from the final run; see below)*
