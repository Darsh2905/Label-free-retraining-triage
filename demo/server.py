"""Live demo server: replays the real 2022 traffic through the real pipeline, day by day, in a browser dashboard.

    python demo/server.py                # then open http://localhost:8800
    python demo/server.py --open         # also opens the browser
    python demo/server.py --port 8811 --seed 0

No extra dependencies (standard-library HTTP server; the dashboard is one HTML file with inline SVG charts, so it
works offline). Start-up loads the 357 sampled days and calibrates once (~40 s); the page shows progress meanwhile.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.live import POLICIES, LiveSession  # noqa: E402
from src.pipeline import build_world  # noqa: E402
from src.utils import load_config  # noqa: E402

PACE = {"fast": 0.0, "normal": 0.4, "slow": 1.2}
INJECT_PRESETS = {"moderate": 2.0, "severe": 5.0}


class Demo:
    def __init__(self, seed: int):
        self.seed = seed
        self.phase, self.msg = "loading", "Starting…"
        self.session: LiveSession | None = None
        self.running = False
        self.pace = "normal"
        self.lock = threading.Lock()                 # held while a day is being processed
        self.cv = threading.Condition()
        self.error = None
        self.epoch = 0                                # bumped on every restart so the page knows to refetch from record 0

    # -- start-up ---------------------------------------------------------------------------
    def load(self):
        try:
            t0 = time.time()
            self.msg = "Loading 357 sampled days of 2022 flows…"
            cfg = load_config()
            world = build_world(cfg, self.seed)
            self.msg = "Calibrating once: signal scales on weeks 3–8, loss map on weeks 3–10 (the only labeled reads outside retraining)…"
            world.calibration()
            self.session = LiveSession(world, self._p0_reference())
            self.phase, self.msg = "ready", f"Ready ({time.time() - t0:.0f} s)"
            print(f"[demo] ready in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:                          # surface start-up problems in the page instead of a blank screen
            self.phase, self.msg, self.error = "error", f"{type(e).__name__}: {e}", traceback.format_exc()
            print(self.error, flush=True)

    def _p0_reference(self):
        """Weekly macro-F1 of the never-retrained model, precomputed offline (grey dashed reference line only)."""
        try:
            run = sorted(glob.glob(os.path.join(ROOT, "results", "run_*_replay_tls-year22")))[-1]
            tl = pickle.load(open(os.path.join(run, f"timelines_seed{self.seed}.pkl"), "rb"))["timelines"]["P0_never"]["rows"]
            return {int(r["week"]): float(r["macro_f1"]) for r in tl}
        except Exception:
            return {}

    # -- worker -----------------------------------------------------------------------------
    def loop(self):
        while True:
            with self.cv:
                while not (self.running and self.session is not None):
                    self.cv.wait()
            s = self.session
            if s.done:
                self.running = False
                continue
            with self.lock:
                try:
                    s.step()
                except Exception:
                    self.running = False
                    self.error = traceback.format_exc()
                    print(self.error, flush=True)
            if s.done:
                self.running = False
            delay = PACE[self.pace]
            if delay:
                time.sleep(delay)

    # -- API --------------------------------------------------------------------------------
    def control(self, body: dict) -> dict:
        a = body.get("action")
        s = self.session
        if s is None:
            return {"ok": False, "error": "still loading"}
        if a == "start":
            if s.done:
                return {"ok": False, "error": "finished: press Reset"}
            self.running = True
            with self.cv:
                self.cv.notify_all()
        elif a == "pause":
            self.running = False
        elif a == "reset":
            self.running = False
            with self.lock:
                s.reset(body.get("policy", "p4c"), float(body.get("ratio", 10)))
                self.epoch += 1
        elif a == "ratio":
            s.set_ratio(float(body["ratio"]))
        elif a == "inject":
            f = body.get("factor")
            f = INJECT_PRESETS.get(f, f) if isinstance(f, str) else f
            s.set_injection(None if f is None else float(f))
        elif a == "pace":
            if body.get("pace") in PACE:
                self.pace = body["pace"]
        else:
            return {"ok": False, "error": f"unknown action {a!r}"}
        return {"ok": True}

    def poll(self, since: int) -> dict:
        s = self.session
        out = {"phase": self.phase, "msg": self.msg, "running": self.running, "pace": self.pace, "error": self.error, "epoch": self.epoch}
        if s is None:
            return out
        recs = s.records
        out.update({"n": len(recs), "records": recs[since:], "status": s.status(), "busy": s.busy, "ledger": s.ledger()})
        if since == 0:
            out["config"] = s.config()
            out["inject_presets"] = INJECT_PRESETS
        return out


def make_handler(demo: Demo):
    static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                      # keep the console quiet
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._send(200, open(os.path.join(static, "index.html"), "rb").read(), "text/html; charset=utf-8")
            elif u.path == "/api/poll":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                body = json.dumps(demo.poll(since), default=lambda o: o.item() if hasattr(o, "item") else str(o))
                self._send(200, body.encode())
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if urlparse(self.path).path != "/api/control":
                return self._send(404, b'{"error":"not found"}')
            n = int(self.headers.get("Content-Length", 0))
            try:
                res = demo.control(json.loads(self.rfile.read(n) or b"{}"))
            except Exception as e:
                res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(res).encode())

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    a = ap.parse_args()
    demo = Demo(a.seed)
    threading.Thread(target=demo.load, daemon=True).start()
    threading.Thread(target=demo.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(demo))
    url = f"http://localhost:{a.port}"
    print(f"[demo] dashboard: {url}   (loading data in the background; the page shows progress)", flush=True)
    if a.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopped")


if __name__ == "__main__":
    main()
