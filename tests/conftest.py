"""Shared fixtures: a small synthetic world with planted drift, and test ordering
(the label-leakage tests always run first, spec §8)."""
import datetime as dt
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.loader import DayData, WorldData  # noqa: E402
from src.utils import load_config  # noqa: E402

N_CLASSES, N_FEAT, FLOWS, WEEKS = 5, 8, 600, 20
MILD_WEEK, BIG_WEEK = 7, 12


def make_synthetic_world(seed: int = 0) -> WorldData:
    """Gaussian class clusters. From week 7 a mild shift, from week 12 a large shift, in
    the class-conditional means of three classes: covariate plus concept drift."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1.6, (N_CLASSES, N_FEAT))
    prior = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    days = []
    start = dt.date(2022, 1, 3)                     # Monday of ISO week 1
    for i in range(WEEKS * 7):
        d = start + dt.timedelta(days=i)
        week = d.isocalendar()[1]
        y = rng.choice(N_CLASSES, FLOWS, p=prior)
        shift = np.zeros((N_CLASSES, N_FEAT))
        if week >= MILD_WEEK:
            shift[:3, :4] += 0.7
        if week >= BIG_WEEK:
            shift[:3, :4] += 2.0
        weekend = 0.3 if d.weekday() >= 5 else 0.0  # a day-of-week effect for STL to remove
        X = mu[y] + shift[y] + rng.normal(0, 1, (FLOWS, N_FEAT)) + weekend
        days.append(DayData(d.strftime("%Y%m%d"), week, X.astype(np.float32), y.astype(np.int16), 1e5))
    return WorldData("synthetic", [f"f{j}" for j in range(N_FEAT)], [f"c{c}" for c in range(N_CLASSES)], days)


def synthetic_config() -> dict:
    return load_config(overrides={
        "dataset": "synthetic", "n_classes": N_CLASSES, "flows_per_day": FLOWS, "max_train_flows": 4000,
        "replay_end": WEEKS, "sigma_window": [3, 6], "calibration_window": [3, 9], "transfer_window": [15, 17],
        "classifier": {"n_estimators": 30, "num_leaves": 7, "num_threads": 1},
        "observer": {"ks": {"n_ref": 800, "n_cur": 400}, "mmd": {"n_ref": 200, "n_cur": 200}},
        "evaluation": {"known_events": {"big_shift": BIG_WEEK}},
    })


@pytest.fixture(scope="session")
def synthetic_world():
    return {"cfg": synthetic_config(), "data": make_synthetic_world()}


def pytest_collection_modifyitems(session, config, items):
    items.sort(key=lambda it: 0 if "test_no_label_leakage" in it.nodeid else 1)


def pytest_collection_finish(session):
    """Test modules exercise the store directly; register them explicitly with the
    runtime caller guard rather than widening its production allow-list."""
    from src.evaluation.sealed_store import SealedLabelStore
    for item in session.items:
        mod = getattr(item, "module", None)
        if mod is not None:
            SealedLabelStore.allow_module(mod.__name__)
