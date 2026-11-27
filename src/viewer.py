"""The viewer and control room: a page that redraws every 2 seconds from the runs' scoreboards, and starts, steers and
stops worlds.

    python -m src.viewer [--runs runs] [--port 8765]        then open http://localhost:8765

Live shows the worlds in progress (or says plainly that nothing is running) with Start, Stop, Pause, the Auto / Manual
switch and a slider for every knob; History keeps every past run. Any world's save can be resumed as a new world. The server listens on this machine only, and its two
writing requests accept JSON only, which another website cannot send to it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from .control import (HUNTERS, MODES, WATCH_LEASE, Control, knob_values, new_run_folder, read_control,
                      settings_from, watch_speed_from, write_control, write_json)
from .dial import KNOBS, value
from .evolve import SPECIES
from .saves import FIRST_PLANET, FLAT_WORLD, SaveError, save_format, save_header, save_path, saves_of
from .rules import GENE_MEANS, GROUPS, RULES, check, check_genes, means, starting_rules
from .scoreboard import OLDER_LAYOUTS, ROW_KEYS, read_rows
from .settings import START, Settings

PAGE = Path(__file__).with_name("viewer.html")
STATIC = Path(__file__).with_name("static")  # three.js and the page's own scripts, served as they are
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".html": "text/html; charset=utf-8", ".txt": "text/plain; charset=utf-8"}
REPO = Path(__file__).resolve().parent.parent
RUNNING_WITHIN = 180.0  # seconds: a run with no end that wrote its scoreboard this recently is still running
MAX_RUNNING = 2  # the box has 4 cores
MAX_BODY = 10_000  # bytes a request may send
NEXT_WORLD = "next_world.json"  # in the runs folder: the owner's changes to the rules the next new world starts on
HANDOVER_WAIT = 300.0  # seconds Update code waits for a world to save and end before giving up
HANDOVER_LOOK = 0.2
KNOB_LABELS = {  # what the page calls each knob, and how far one notch of its slider moves it
    "max_food": ("Food pieces at most", 1.0),
    "food_energy": ("Energy per food", 0.5),
    "food_per_step": ("New food per step", 0.05),
    "fruit_rate": ("Fruit a tree drops per step", 0.002),
    "move_cost": ("Cost of moving (per unit)", 0.01),
    "step_cost": ("Cost of living (per step)", 0.005),
    "lifespan": ("Lifespan (steps)", 50.0),
    "sprint_power": ("Cost of sprinting (power of speed ÷ cruising)", 0.01),
    "eat_miss": ("Cost of eating at nothing", 0.005),
    "attack_cost": ("Cost of an attack (× size)", 0.01),
    "attack_damage": ("Damage of a hit (× the hitter's size)", 0.05),
    "bump_cost": ("Cost of bumping into a rock", 0.005),
    "vision_cost": ("Cost of seeing (per square unit, per step)", 0.00001),
    "hearing_cost": ("Cost of hearing (per square unit, per step)", 0.00001),
    "neuron_cost": ("Cost of each neuron (per step)", 0.00001),
    "stealth_cost": ("Cost of stealth (living × 1 + stealth × this)", 0.005),
    "armour_cost": ("Cost of armour (living × 1 + armour × this)", 0.005),
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
        if rows and tuple(rows[0]) not in (ROW_KEYS,) + OLDER_LAYOUTS:
            continue  # a scoreboard of another design altogether
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
            "paused": control.paused if control is not None else False,
            "saves": len(saves_of(run)),
            "rows": [{key: _number(row[key]) if key in row else None for key in ROW_KEYS}  # a newer column: empty
                     for row in rows],
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


def _on_its_own() -> Dict:
    """How a world is started so it does not belong to the page server: restarting the server leaves it running."""
    if os.name == "nt":
        return {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_BREAKAWAY_FROM_JOB}
    return {"start_new_session": True}


def _launch(command: List[str], folder: Path) -> subprocess.Popen:
    environment = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
    with open(folder / "run.log", "a") as log:  # the world keeps its own copy of the handle
        return subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                **_on_its_own())


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
        """A new world: of a species and seed, or resumed from a save (`from: {run, step}`)."""
        mode = body.get("mode")
        if mode not in MODES:
            raise Refused(400, f"mode must be one of {', '.join(MODES)}")
        settings = self._settings(body.get("settings"), START) if body.get("settings") is not None else None
        if body.get("from") is not None:
            save = self._save(body["from"])
            try:
                header = save_header(save)
            except SaveError as problem:  # a flat world from before the sphere, or a broken save: nothing starts
                raise Refused(409, str(problem)) from problem
            seed, kind = header["seed"], header["kind"]
            how = ["--resume", str(save.resolve())]
        else:
            species, seed = body.get("species"), body.get("seed")
            if species not in SPECIES:
                raise Refused(400, f"species must be one of {', '.join(SPECIES)}")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise Refused(400, "seed must be a whole number, 0 or more")
            kind, how = species, ["--seed", str(seed), "--species", species]
        alive = set(self._running()) | {name for name, process in self.started.items() if process.poll() is None}
        if len(alive) >= MAX_RUNNING:
            raise Refused(409, f"{MAX_RUNNING} worlds are running already: stop one first")
        folder = new_run_folder(self.folder, seed, kind)
        folder.mkdir(parents=True)
        rules = None if body.get("from") is not None else (self._next_rules() or None)  # a new world: the owner's rules
        write_control(folder, Control(mode, settings, False, rules=rules))
        command = [sys.executable, "-m", "src.evolve", *how, "--forever", "--mode", mode, "--out", str(folder.resolve())]
        self.started[folder.name] = self.launch(command, folder)
        return {"name": folder.name}

    def _run_folder(self, name: object) -> Path:
        if not isinstance(name, str) or name not in {run.name for run in self.folder.iterdir()}:
            raise Refused(404, f"no world called {name!r}")
        return self.folder / name

    def _save(self, source: object) -> Path:
        if not isinstance(source, dict):
            raise Refused(400, "from must say the world and the step: {run, step}")
        run, step = self._run_folder(source.get("run")), source.get("step")
        if step not in saves_of(run):
            raise Refused(404, f"{run.name} has no save at step {step}")
        return save_path(run, step)

    def saves(self, name: Optional[str]) -> List[int]:
        return saves_of(self._run_folder(name))

    def now(self, name: Optional[str]) -> bytes:
        """The world's numbers right now (it writes them every 25 steps); null before its first ones (or for a world
        made by older code, which writes none)."""
        numbers = self._run_folder(name) / "now.json"
        return numbers.read_bytes() if numbers.exists() else b"null"

    def _next_rules(self) -> Dict[str, float]:
        """The owner's changes to the rules of the next new world (kept in the runs folder)."""
        path = self.folder / NEXT_WORLD
        known = starting_rules()  # a rule that has since become a knob (a cost, 2026-11-26) is dropped
        return {name: value for name, value in json.loads(path.read_text())["rules"].items() if name in known} if path.exists() else {}

    def next_rules(self, body: Dict) -> Dict:
        """Changes the rules the next new world starts on, each checked like a running world's."""
        given = body.get("rules")
        if not isinstance(given, dict):
            raise Refused(400, "rules must map each rule to its new value")
        values, changes = {**starting_rules(), **self._next_rules()}, self._next_rules()
        for name, value in given.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise Refused(400, f"{name} must be a number, not {value!r}")
            problem = check(values, name, float(value))
            if problem:
                raise Refused(400, problem)
            values[name] = changes[name] = float(value)
        write_json(self.folder / NEXT_WORLD, {"rules": changes})
        return {"rules": changes}

    def rules(self, name: Optional[str]) -> Dict:
        """Every rule with its value in the world now (its owner's change waiting for the world counts already); with
        no world named, the rules the next new world starts on."""
        values = self._world_rules(self._run_folder(name)) if name else {**starting_rules(), **self._next_rules()}
        return {"groups": [{"name": name, "title": title} for name, title in GROUPS], "genes": GENE_MEANS, "next": not name,
                "rules": [{"name": r.name, "group": r.group, "words": r.words, "means": means(r.name), "start": r.start,
                           "low": r.low, "high": r.high, "value": values.get(r.name, r.start)} for r in RULES]}

    def _world_rules(self, folder: Path) -> Dict[str, float]:
        numbers = folder / "now.json"
        now = json.loads(numbers.read_text()) if numbers.exists() else None
        values = dict(now["rules"]) if now and "rules" in now else starting_rules()
        pending = read_control(folder)
        return {**values, **(pending.rules if pending and pending.rules else {})}

    def _rule_changes(self, folder: Path, given: object, current: Optional[Dict[str, float]]) -> Dict[str, float]:
        """The owner's rule changes so far with `given` added, each checked against the world's rules."""
        if not isinstance(given, dict):
            raise Refused(400, "rules must map each rule to its new value")
        values, changes = self._world_rules(folder), dict(current or {})
        for name, value in given.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise Refused(400, f"{name} must be a number, not {value!r}")
            problem = check(values, name, float(value))
            if problem:
                raise Refused(400, problem)
            values[name] = changes[name] = float(value)
        return changes

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
        hunters = current.hunters if mode == "auto" else None
        if body.get("settings") is not None:
            given = body["settings"]
            if mode != "manual" and isinstance(given, dict) and set(given) == {HUNTERS}:
                if not isinstance(given[HUNTERS], bool):
                    raise Refused(400, "hunters must be true or false")
                hunters = given[HUNTERS]  # the hunters switch works in Auto too
            elif mode != "manual":
                raise Refused(409, "switch to Manual to turn the knobs")
            else:
                settings = self._settings(given, settings or self._now(running[name]))
        stop, paused = body.get("stop", current.stop), body.get("paused", current.paused)
        if not isinstance(stop, bool):
            raise Refused(400, "stop must be true or false")
        if not isinstance(paused, bool):
            raise Refused(400, "paused must be true or false")
        watch_until, watch_speed = current.watch_until, current.watch_speed
        if "watch_speed" in body:
            try:
                watch_speed = watch_speed_from(body["watch_speed"])
            except ValueError as problem:
                raise Refused(400, str(problem)) from problem
        if "watch" in body:
            if not isinstance(body["watch"], bool):
                raise Refused(400, "watch must be true or false")
            watch_until = time.time() + WATCH_LEASE if body["watch"] else None  # the page renews it while it watches
        rules = current.rules
        if "rules" in body:
            rules = self._rule_changes(folder, body["rules"], current.rules)
        edit = current.edit
        if "edit" in body:
            given = body["edit"]
            if not isinstance(given, dict) or isinstance(given.get("id"), bool) or not isinstance(given.get("id"), int):
                raise Refused(400, "edit must name a creature's id and its genes")
            problem = check_genes(self._world_rules(folder), given.get("genes"))
            if problem:
                raise Refused(400, problem)
            edit = {"seq": time.time_ns(), "id": given["id"], "genes": given["genes"]}
        update = body.get("update", False)
        if not isinstance(update, bool):
            raise Refused(400, "update must be true or false")
        if update:  # the world saves and ends; then it goes on from that save with today's code, in the same run
            saved = saves_of(folder)
            if saved and 0 < save_format(save_path(folder, saved[-1])) < FIRST_PLANET:
                raise Refused(409, f"{name}: {FLAT_WORLD} (it keeps running on the code it started with)")
            stop, paused = True, False
        write_control(folder, Control(mode, settings, stop, watch_until, watch_speed, paused, hunters, rules, edit))
        if update:
            threading.Thread(target=self._hand_over, args=(folder,), daemon=True).start()
        return {"mode": mode, "settings": None if settings is None else knob_values(settings), "stop": stop,
                "paused": paused, "watching": watch_until is not None, "watch_speed": watch_speed, "updating": update}

    def _hand_over(self, folder: Path) -> None:
        """Update code: waits for the world to save and end, then starts it again from its last save, same run."""
        ended = folder / "time.txt"
        deadline = time.monotonic() + HANDOVER_WAIT
        while not ended.exists():
            if time.monotonic() > deadline:
                print(f"Update code: {folder.name} did not end within {HANDOVER_WAIT:.0f} s; nothing was restarted",
                      flush=True)
                return
            time.sleep(HANDOVER_LOOK)
        save = save_path(folder, saves_of(folder)[-1])
        ended.unlink()
        current = read_control(folder)
        write_control(folder, Control(current.mode, current.settings, False, current.watch_until, current.watch_speed,
                                      False, current.hunters, current.rules))  # the owner's rule changes wait on
        steps = json.loads((folder / "run.json").read_text())["steps"]
        length = ["--forever"] if steps is None else ["--steps", str(steps)]
        command = [sys.executable, "-m", "src.evolve", "--resume", str(save.resolve()), "--continue", *length,
                   "--out", str(folder.resolve())]
        self.started[folder.name] = self.launch(command, folder)

    def view(self, name: Optional[str]) -> bytes:
        """The picture a watched world last drew (it is drawn 10 to 30 times a second, so no scoreboard is read)."""
        folder = self._run_folder(name)
        control = read_control(folder)
        picture = folder / "view.json"
        if control is None or not control.watched(time.time()) or not picture.exists():
            raise Refused(404, "that world is not being watched")
        return picture.read_bytes()

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
        last = run["rows"][-1]  # a knob a row from before it has no number for keeps its start
        known = {k.name: last[k.name] for k in KNOBS if last.get(k.name) is not None}
        return settings_from({**known, HUNTERS: bool(last["hunters"])}, START)


def make_server(folder: Path, port: int, launch: Launch = _launch) -> ThreadingHTTPServer:
    controls = Controls(folder, launch)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/data"):
                self._send(200, runs(folder))
            elif self.path.startswith("/knobs"):
                self._send(200, knobs())
            elif self.path.startswith(("/view", "/saves", "/now", "/rules")):
                try:
                    name = parse_qs(urlparse(self.path).query).get("run", [None])[0]
                    if self.path.startswith("/view"):
                        self._raw(200, controls.view(name), "application/json")
                    elif self.path.startswith("/now"):
                        self._raw(200, controls.now(name), "application/json")
                    elif self.path.startswith("/rules"):
                        self._send(200, controls.rules(name))
                    else:
                        self._send(200, controls.saves(name))
                except Refused as refused:
                    self._send(refused.status, {"error": str(refused)})
            elif self.path.startswith("/static/"):
                name = unquote(urlparse(self.path).path[len("/static/"):])
                path = STATIC / name
                if "/" in name or "\\" in name or name.startswith(".") or not path.is_file() or path.suffix not in STATIC_TYPES:
                    self._send(404, {"error": f"no such file: {name}"})  # only the files in src/static, never a path out of it
                else:
                    self._raw(200, path.read_bytes(), STATIC_TYPES[path.suffix])
            else:
                body = PAGE.read_bytes()
                self._raw(200, body, "text/html; charset=utf-8")

        def do_POST(self) -> None:
            actions = {"/start": controls.start, "/control": controls.control, "/next-rules": controls.next_rules}
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
