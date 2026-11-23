"""The viewer and control room: a page that redraws every 2 seconds from the runs' scoreboards, and starts, steers and
stops worlds.

    python -m src.viewer [--runs runs] [--port 8765]        then open http://localhost:8765

Live shows the worlds in progress (or says plainly that nothing is running) with Start, Stop, the Auto / Manual switch
and a slider for every knob; History keeps every past run. The server listens on this machine only, and its two
writing requests accept JSON only, which another website cannot send to it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .control import HUNTERS, MODES, Control, knob_values, new_run_folder, read_control, settings_from, write_control
from .dial import KNOBS, value
from .evolve import SPECIES
from .scoreboard import ROW_KEYS, read_rows
from .settings import START, Settings

PAGE = Path(__file__).with_name("viewer.html")
REPO = Path(__file__).resolve().parent.parent
RUNNING_WITHIN = 180.0  # seconds: a run with no end that wrote its scoreboard this recently is still running
MAX_RUNNING = 2  # the box has 4 cores
MAX_BODY = 10_000  # bytes a request may send
KNOB_LABELS = {  # what the page calls each knob, and how far one notch of its slider moves it
    "max_food": ("Food pieces at most", 1.0),
    "food_energy": ("Energy per food", 0.5),
    "food_per_step": ("New food per step", 0.05),
    "move_cost": ("Cost of moving (per unit)", 0.01),
    "step_cost": ("Cost of living (per step)", 0.005),
    "lifespan": ("Lifespan (steps)", 50.0),
}
Launch = Callable[[List[str], Path], object]  # starts a world: its command and its folder; returns its process


def runs(folder: Path) -> List[Dict]:
    """Every run under `folder`, newest first: its record (seed, kind, steps asked for or None, when it started), its
    rows as numbers (an empty cell is None), the seconds it took once finished, its mode (Auto or Manual), and its
    status: "running" while it keeps writing, "finished" once it has ended, "stopped" when it ended without finishing."""
    found = []
    for run in folder.iterdir():
        scoreboard, record, timing = run / "scoreboard.csv", run / "run.json", run / "time.txt"
        if not record.exists():
            continue
        rows = read_rows(scoreboard) if scoreboard.exists() else []
        if rows and tuple(rows[0]) != ROW_KEYS:
            continue  # a scoreboard of an older design
        written = (scoreboard if scoreboard.exists() else record).stat().st_mtime
        if timing.exists():
            status = "finished"
        elif time.time() - written <= RUNNING_WITHIN:
            status = "running"
        else:
            status = "stopped"
        control = read_control(run)
        found.append({
            **json.loads(record.read_text()), "name": run.name, "status": status,
            "mode": control.mode if control is not None else "auto",
            "rows": [{key: _number(row[key]) for key in ROW_KEYS} for row in rows],
            "seconds": float(timing.read_text()) if timing.exists() else None,
        })
    return sorted(found, key=lambda run: run["started"], reverse=True)


def _number(cell: str) -> Optional[float]:
    if cell == "":
        return None
    value = float(cell)
    return int(value) if value.is_integer() else value


def knobs() -> Dict:
    """The sliders the page shows: every knob's ends and starting value, and the hunters switch."""
    sliders = []
    for knob in KNOBS:
        label, step = KNOB_LABELS[knob.name]
        sliders.append({"name": knob.name, "label": label, "easiest": knob.easiest, "hardest": knob.hardest,
                        "start": value(START, knob.name), "step": step})
    return {"knobs": sliders, HUNTERS: START.attacks}


def _launch(command: List[str], folder: Path) -> subprocess.Popen:
    environment = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
    with open(folder / "run.log", "w") as log:  # the world keeps its own copy of the handle
        return subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=environment)


class Refused(Exception):
    """A request the server will not carry out: its HTTP status and the reason in plain words."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


class Controls:
    """Starting, steering and stopping worlds under `folder`."""

    def __init__(self, folder: Path, launch: Launch) -> None:
        self.folder = folder
        self.launch = launch
        self.started: Dict[str, object] = {}  # this server's worlds by name: a world just started has no rows yet

    def _running(self) -> Dict[str, Dict]:
        return {run["name"]: run for run in runs(self.folder) if run["status"] == "running"}

    def start(self, body: Dict) -> Dict:
        species, seed, mode = body.get("species"), body.get("seed"), body.get("mode")
        if species not in SPECIES:
            raise Refused(400, f"species must be one of {', '.join(SPECIES)}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise Refused(400, "seed must be a whole number, 0 or more")
        if mode not in MODES:
            raise Refused(400, f"mode must be one of {', '.join(MODES)}")
        settings = self._settings(body.get("settings"), START) if body.get("settings") is not None else None
        alive = set(self._running()) | {name for name, process in self.started.items() if process.poll() is None}
        if len(alive) >= MAX_RUNNING:
            raise Refused(409, f"{MAX_RUNNING} worlds are running already: stop one first")
        folder = new_run_folder(self.folder, seed, species)
        folder.mkdir(parents=True)
        write_control(folder, Control(mode, settings, False))
        command = [sys.executable, "-m", "src.evolve", "--seed", str(seed), "--forever", "--species", species,
                   "--mode", mode, "--out", str(folder.resolve())]
        self.started[folder.name] = self.launch(command, folder)
        return {"name": folder.name}

    def control(self, body: Dict) -> Dict:
        running = self._running()
        name = body.get("run")
        if name not in running:
            raise Refused(404, f"no world called {name!r} is running")
        folder = self.folder / name
        current = read_control(folder) or Control("auto", None, False)
        mode = body.get("mode", current.mode)
        if mode not in MODES:
            raise Refused(400, f"mode must be one of {', '.join(MODES)}")
        settings = current.settings if mode == "manual" else None
        if body.get("settings") is not None:
            if mode != "manual":
                raise Refused(409, "switch to Manual to turn the knobs")
            settings = self._settings(body["settings"], settings or self._now(running[name]))
        stop = body.get("stop", current.stop)
        if not isinstance(stop, bool):
            raise Refused(400, "stop must be true or false")
        write_control(folder, Control(mode, settings, stop))
        return {"mode": mode, "settings": None if settings is None else knob_values(settings), "stop": stop}

    @staticmethod
    def _settings(given: object, base: Settings) -> Settings:
        if not isinstance(given, dict):
            raise Refused(400, "settings must be an object of knob values")
        try:
            return settings_from(given, base)
        except ValueError as problem:
            raise Refused(400, str(problem)) from problem

    @staticmethod
    def _now(run: Dict) -> Settings:
        """The world's settings as its last row shows them (the start, before its first row)."""
        if not run["rows"]:
            return START
        last = run["rows"][-1]
        return settings_from({**{k.name: last[k.name] for k in KNOBS}, HUNTERS: bool(last["hunters"])}, START)


def make_server(folder: Path, port: int, launch: Launch = _launch) -> ThreadingHTTPServer:
    controls = Controls(folder, launch)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/data"):
                self._send(200, runs(folder))
            elif self.path.startswith("/knobs"):
                self._send(200, knobs())
            else:
                body = PAGE.read_bytes()
                self._raw(200, body, "text/html; charset=utf-8")

        def do_POST(self) -> None:
            actions = {"/start": controls.start, "/control": controls.control}
            try:
                if self.path not in actions:
                    raise Refused(404, f"no such request: {self.path}")
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    raise Refused(415, "send JSON (Content-Type: application/json)")
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    raise Refused(413, "the request is too large")
                try:
                    body = json.loads(self.rfile.read(length) or b"null")
                except json.JSONDecodeError as problem:
                    raise Refused(400, f"not valid JSON: {problem}") from problem
                if not isinstance(body, dict):
                    raise Refused(400, "send a JSON object")
                self._send(200, actions[self.path](body))
            except Refused as refused:
                self._send(refused.status, {"error": str(refused)})

        def _send(self, status: int, answer: object) -> None:
            self._raw(status, json.dumps(answer).encode(), "application/json")

        def _raw(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # quiet: the page asks every 2 seconds
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the live viewer and control room of the runs.")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    args.runs.mkdir(parents=True, exist_ok=True)
    server = make_server(args.runs, args.port)
    print(f"life viewer on http://localhost:{args.port} (runs: {args.runs})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
