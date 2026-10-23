"""The demo server's control/poll API on the synthetic world (no real data, no HTTP socket needed)."""
import importlib.util
import json
import os
import threading
import time

import pytest

from src.evaluation.harness import World
from src.evaluation.live import LiveSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def server_module():
    spec = importlib.util.spec_from_file_location("demo_server", os.path.join(ROOT, "demo", "server.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def demo(server_module, synthetic_world):
    d = server_module.Demo(seed=0)
    d.session = LiveSession(World(synthetic_world["cfg"], synthetic_world["data"], seed=0), {})
    d.phase = "ready"
    threading.Thread(target=d.loop, daemon=True).start()
    return d


def _wait(demo, cond, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_start_run_to_end_and_poll_is_json(demo):
    assert demo.control({"action": "pace", "pace": "fast"})["ok"]
    assert demo.control({"action": "start"})["ok"]
    assert _wait(demo, lambda: demo.session.done)
    out = demo.poll(0)
    json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o))     # must serialise
    assert out["status"]["done"] and out["ledger"]["monitor_label_reads"] == 0
    assert out["config"]["total_days"] == demo.session.total_days
    n = out["n"]
    assert demo.poll(n)["records"] == []                                                # incremental polling returns only new records


def test_restart_bumps_epoch_and_clears_records(demo):
    demo.control({"action": "pace", "pace": "fast"})
    demo.control({"action": "start"})
    _wait(demo, lambda: len(demo.session.records) > 5)
    e0 = demo.epoch
    assert demo.control({"action": "reset", "policy": "p3", "ratio": 10})["ok"]
    assert demo.epoch == e0 + 1 and demo.poll(0)["n"] == 0 and not demo.running


def test_pause_stops_progress(demo):
    demo.control({"action": "pace", "pace": "fast"})
    demo.control({"action": "start"})
    _wait(demo, lambda: len(demo.session.records) > 3)
    demo.control({"action": "pause"})
    time.sleep(0.4)
    n = len(demo.session.records)
    time.sleep(0.4)
    assert len(demo.session.records) == n


def test_inject_and_ratio_controls(demo):
    assert demo.control({"action": "ratio", "ratio": 100})["ok"] and demo.session.ratio == 100
    assert demo.control({"action": "inject", "factor": "severe"})["ok"] and demo.session.inject == {"factor": 5.0}
    assert demo.control({"action": "inject", "factor": None})["ok"] and demo.session.inject is None
    assert not demo.control({"action": "nonsense"})["ok"]


def test_dashboard_html_has_every_panel():
    html = open(os.path.join(ROOT, "demo", "static", "index.html"), encoding="utf-8").read()
    for el in ("svgF1", "svgD", "svgZ", "svgTop", "flowBody", "ledMon", "decText", "bStart", "rRange", "realBox", "log"):
        assert f'id="{el}"' in html, f"dashboard lost panel {el}"
