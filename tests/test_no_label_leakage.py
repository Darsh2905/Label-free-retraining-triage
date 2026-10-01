"""CRITICAL: the label-free claim rests on these tests (spec §8). They run first
(see pytest_collection_modifyitems in conftest.py)."""
import ast
import pathlib
import textwrap

import numpy as np
import pytest

from tests.leakcheck import violations

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---- 1. the spec's own test, kept verbatim in intent (weaker; see test 2) -------------
FORBIDDEN_IMPORT = "sealed_store"
CHECKED_DIRS = ["src/observer", "src/detection"]
ALLOWED_IN_DECISION = {"calibration.py"}


def test_no_sealed_store_imports_outside_allowed_paths():
    for d in CHECKED_DIRS + ["src/decision"]:
        for f in (ROOT / d).rglob("*.py"):
            if d == "src/decision" and f.name in ALLOWED_IN_DECISION:
                continue
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [n.name for n in node.names] if isinstance(node, ast.Import) else \
                        [node.module] + [n.name for n in node.names]   # fix: spec omitted the alias names
                    assert not any(FORBIDDEN_IMPORT in (n or "") for n in names), f"{f} imports sealed_store"


# ---- 2. transitive import-graph check over every restricted module ---------------------
def test_no_restricted_module_reaches_labels_transitively():
    bad = violations(ROOT)
    assert not bad, "label-bearing module reachable from restricted code:\n" + "\n".join(bad)


# ---- 3. the checker itself must catch every evasion form (mutation test) ---------------
EVASIONS = {
    "from_pkg_import_module": "from src.evaluation import sealed_store\n",
    "relative": "from ..evaluation.sealed_store import SealedLabelStore\n",
    "relative_pkg": "from .. import evaluation\n",
    "import_as": "import src.evaluation.sealed_store as s\n",
    "importlib": "import importlib\nm = importlib.import_module('src.evaluation.sealed_store')\n",
    "dunder_import": "m = __import__('src.data.loader')\n",
    "loader": "from src.data.loader import load_world\n",
    "harness": "from src.evaluation.harness import World\n",
    "pipeline": "import src.pipeline\n",
}


def _mini_repo(tmp_path, observer_code: str, helper_code: str = "x = 1\n"):
    for pkg in ["src", "src/observer", "src/evaluation", "src/data", "src/detection", "src/decision",
                "src/classifier", "src/util"]:
        (tmp_path / pkg).mkdir(parents=True, exist_ok=True)
        (tmp_path / pkg / "__init__.py").write_text("")
    (tmp_path / "src/evaluation/sealed_store.py").write_text("class SealedLabelStore: pass\n")
    (tmp_path / "src/evaluation/harness.py").write_text("class World: pass\n")
    (tmp_path / "src/data/loader.py").write_text("def load_world(): pass\n")
    (tmp_path / "src/pipeline.py").write_text("")
    (tmp_path / "src/util/helper.py").write_text(helper_code)
    (tmp_path / "src/observer/sneaky.py").write_text(observer_code)
    return tmp_path


def test_clean_mini_repo_passes(tmp_path):
    assert violations(_mini_repo(tmp_path, "import numpy\n")) == []


@pytest.mark.parametrize("name", sorted(EVASIONS))
def test_checker_catches_evasion(tmp_path, name):
    assert violations(_mini_repo(tmp_path, EVASIONS[name])), f"checker missed evasion form: {name}"


def test_checker_catches_transitive_evasion(tmp_path):
    root = _mini_repo(tmp_path, "from src.util import helper\n",
                      helper_code="from src.evaluation import sealed_store\n")
    assert violations(root)


def test_spec_literal_test_would_miss_from_pkg_import(tmp_path):
    """Documents the bug in the spec's reference test: checking node.module alone misses this."""
    tree = ast.parse(EVASIONS["from_pkg_import_module"])
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
    assert FORBIDDEN_IMPORT not in (node.module or "")          # the spec's check sees nothing
    assert any(FORBIDDEN_IMPORT in a.name for a in node.names)  # ...but the import is there


# ---- 4. runtime guards --------------------------------------------------------------
def _store():
    from src.evaluation.sealed_store import SealedLabelStore
    labels = {f"202201{d:02d}": np.arange(10) % 3 for d in range(3, 31)}
    periods = {d: (int(d[-2:]) - 3) // 7 + 1 for d in labels}
    return SealedLabelStore(labels, periods, 3, (2, 3))


def test_runtime_caller_guard_blocks_restricted_modules():
    from src.evaluation.sealed_store import LabelAccessError
    store = _store()
    for mod in ("src.observer.fake", "src.decision.policies", "src.detection.x", "src.classifier.model"):
        g = {"__name__": mod, "store": store}
        with pytest.raises(LabelAccessError):
            exec("store.get_retrain_labels(range(1, 2))", g)
        with pytest.raises(LabelAccessError):
            exec("store.grade_counts('20220103', [0]*10)", g)


def test_clock_blocks_future_labels():
    from src.evaluation.sealed_store import LabelAccessError
    store = _store()
    store.advance_to(2)
    assert len(store.get_retrain_labels(range(1, 3))) > 0
    with pytest.raises(LabelAccessError):
        store.get_retrain_labels(range(2, 4))
    with pytest.raises(LabelAccessError):
        store.get_calibration_labels(range(2, 4))   # window not elapsed yet
    with pytest.raises(LabelAccessError):
        store.advance_to(1)                         # clock is monotone within a timeline
    store.advance_to(4)
    with pytest.raises(LabelAccessError):
        store.get_calibration_labels(range(1, 3))   # outside calibration window


def test_labels_are_read_only():
    store = _store()
    store.advance_to(2)
    arr = next(iter(store.get_retrain_labels(range(1, 2)).values()))
    with pytest.raises(ValueError):
        arr[0] = 99


def test_replay_never_carries_labels(synthetic_world):
    from src.data.replay import FrozenScaler, Replay, StreamDay
    data = synthetic_world["data"]
    with pytest.raises(TypeError):
        Replay(data.days, FrozenScaler(np.zeros(data.days[0].raw.shape[1], bool)), 3, 5)  # DayData has labels
    days = [StreamDay(d.date, d.period, d.raw, d.volume) for d in data.days]
    sc = FrozenScaler(np.zeros(days[0].raw.shape[1], bool)).fit(days[0].raw)
    r = Replay(days, sc, 3, 6)
    while r.has_next():
        b = r.next_week()
        assert b.true_labels is None


# ---- 5. behavioural: grading labels cannot influence label-free decisions --------------
def test_decisions_invariant_to_grading_labels(synthetic_world, monkeypatch):
    """Run P3/P4/P4c with the real grader, then with a grader that returns garbage. The
    decisions must be identical. The oracle P5 must change, proving the patch bites."""
    from src.evaluation.harness import World
    from src.evaluation.sealed_store import SealedLabelStore
    cfg, data = synthetic_world["cfg"], synthetic_world["data"]
    specs = [cfg["policies"][k] for k in ("p3_threshold", "p4_proposed", "p4c_cumulative", "p5_oracle")]
    w1 = World(cfg, data, seed=0)
    base = {s["name"]: w1.run_timeline(s)["retrain_weeks"] for s in specs}

    rng = np.random.default_rng(123)

    def garbage(self, date, predictions):
        k = self.n_classes
        tp = rng.integers(0, 50, k)
        return {"tp": tp, "fp": rng.integers(0, 50, k), "fn": rng.integers(0, 50, k),
                "n_known": np.array(1000), "n_correct": np.array(int(tp.sum()))}
    monkeypatch.setattr(SealedLabelStore, "grade_counts", garbage)
    w2 = World(cfg, data, seed=0)
    pert = {s["name"]: w2.run_timeline(s)["retrain_weeks"] for s in specs}
    for name in ("P3_threshold", "P4_proposed", "P4c_cumulative"):
        assert base[name] == pert[name], f"{name} decisions changed when grading labels changed"
    assert base["P5_oracle"] != pert["P5_oracle"], "sanity: garbage grader should change the oracle"
    assert any(base[n] for n in ("P3_threshold", "P4_proposed", "P4c_cumulative")), \
        "vacuous test: no label-free policy retrained on the synthetic drift"
