"""Run one living world, headless with zero delay, and keep its scoreboard.

    python -m src.evolve --seed 1 [--steps 100000 | --forever] --species learning|instinct [--mode auto|manual]
                         [--inherit learned|inborn] [--level N] [--max-level M] [--out DIR]
    python -m src.evolve --seed 1 --steps 20000 --policy random|rules [--level N]      (a yardstick world)
    python -m src.evolve --resume runs/<run>/saves/step-0016000.npz [--steps N | --forever] [--mode auto|manual]
    python -m src.evolve --resume runs/<run>/saves/step-0016000.npz --continue --out runs/<run>   (Update code: the
                         same run goes on from its save with the code as it is now)

The world is never restarted and no creature is ever ranked or picked: births, deaths and the soup are the world's own
rules (src/world.py). This runner starts the world, takes the live page's control file every few steps (src/control.py:
the mode, the owner's knobs, Stop, Pause, a jump-in), lets Auto review the world every 1,000 steps (src/dial.py), holds
the intelligence exam as a measurement, writes the scoreboard, and saves the whole world every 2,000 steps
(src/saves.py). A world resumed from a save is a new run that begins with the old run's history up to that step.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import deque
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .control import MODES, Control, new_run_folder, read_control, write_control, write_json
from .creature import Creature
from .dial import Dial, rates
from .exam import ExamResult, run_exam
from .hall_of_fame import HallOfFame
from .saves import (SaveError, load_world, random_state, save_header, save_path, save_world,
                    set_random_state)
from .terrain import make_terrain
from .scoreboard import ROW_KEYS, Scoreboard, Value, migrate_scoreboard, read_rows
from .rules import ROOM_SLOTS, starting_rules
from .world import INHERIT, World

DEFAULT_STEPS = 100000
PAUSE_LOOK = 0.25  # seconds between two looks at the control file while paused
NOW_EVERY = 25  # steps between two looks at whether now.json is due (the live page's every-second numbers)
SPECIES = {"learning": "learner", "instinct": "brain"}  # the name on the command line -> the creatures' policy
YARDSTICKS = ("random", "rules")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one living world and keep its scoreboard.")
    parser.add_argument("--seed", type=int, default=None)
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--steps", type=int, default=None, help=f"how long to run (default {DEFAULT_STEPS:,})")
    length.add_argument("--forever", action="store_true", help="run until Stop is pressed on the live page")
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--species", choices=tuple(SPECIES))
    kind.add_argument("--policy", choices=YARDSTICKS, help="a yardstick world instead of a species")
    kind.add_argument("--resume", help="a save to go on from, as a new run (its seed and kind come from the save)")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="with --resume: go on in the save's own run (after an update of the code)")
    parser.add_argument("--mode", choices=MODES, default="auto",
                        help="auto: the world turns its own knobs; manual: only the owner does (from the live page)")
    parser.add_argument("--level", type=float, default=None,
                        help="the difficulty the world starts at (nudges harder than the easy start)")
    parser.add_argument("--max-level", type=int, default=None, help="Auto never makes the world harder than this")
    parser.add_argument("--inherit", choices=INHERIT, default=None,
                        help="what a learning baby gets: the parent's brain as it is now, or as the parent was born")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.continue_run and not args.resume:
        parser.error("--continue goes with --resume")
    if args.resume:
        if args.seed is not None or args.level is not None or args.inherit is not None:
            parser.error("a resumed world takes its seed, difficulty and inheritance from its save")
        args.kind = None  # read from the save when the run starts
    elif args.seed is None:
        parser.error("--seed is required")
    args.level = 0.0 if args.level is None else args.level
    args.inherit = "learned" if args.inherit is None else args.inherit
    if args.forever:
        args.steps = None
    elif args.steps is None:
        args.steps = DEFAULT_STEPS
    if args.species:
        args.kind = args.species if args.inherit == "learned" else f"{args.species}-{args.inherit}"
    elif args.policy:
        args.kind = f"{args.policy}-L{args.level:g}"
    if args.out is None and not args.resume:  # every run gets its own dated folder, so history is never overwritten
        args.out = new_run_folder(Path("runs"), args.seed, args.kind).as_posix()
    return args


def _exam_brains(world: World, sample: Sequence[Creature]) -> List[Any]:
    """What the exam tests: a learner creature's current (learned) network, otherwise its brain."""
    return [world.learner.current(c.slot) if c.policy == "learner" else c.brain for c in sample]


def world_view(world: World) -> Dict[str, Any]:
    """Where everything is, for the page's picture of a watched world."""
    return {
        "step": world.step_count, "radius": world.radius,
        "creatures": [{"id": c.id, "p": [round(float(v), 6) for v in c.p], "radius": round(c.radius, 3), "size": round(c.size, 3),
                       "energy": round(max(c.energy, 0.0) / c.max_energy, 3),
                       "heading": [round(v, 3) for v in c.intended_vector], "did": c.last_action, "intent": c.intent,
                       "target": c.target, "baby": c.had_baby_at == world.step_count,
                       "generation": c.generation, "age": c.age, "lifespan": c.lifespan,
                       "earned": round(c.earned, 2), "spent": round(c.spent, 2),
                       "facing": [round(float(v), 4) for v in c.facing], "vision_width": round(c.vision_width, 1),
                       "vision_reach": round(c.vision_reach, 2), "hearing": round(c.hearing, 2),
                       "fear_of_death": round(c.fear_of_death, 2), "love_of_babies": round(c.love_of_babies, 2),
                       "learns_from_others": round(c.learns_from_others, 3), "neurons": c.neurons,
                       "layers": c.layers, "moved": c.moved, "bumped": c.bumped, "mutations": c.mutations,
                       "mutation_size": round(c.mutation_size, 4), "diet": round(c.diet, 3), "calls": c.calls, "kin": round(c.kin, 3), "water": round(c.water, 3), "stealth": round(c.stealth, 3), "armour": round(c.armour, 3),
                       "call": None if c.called is None else [c.called[0], round(c.called[1], 2)],
                       "brain": world.learner.brain_size(c.slot) if c.policy == "learner" else None,
                       "learning_speed": world.learner.lr(c.slot) if c.policy == "learner" else None} for c in world.creatures],
        "foods": [{"p": [round(float(v), 6) for v in f.p], "radius": round(f.radius, 3), "fruit": f.fruit, "meat": f.meat}
                  for f in world.foods],
        "terrain": world.terrain.state(),
    }


def world_now(world: World) -> Dict[str, Any]:
    """The world's numbers right now, for the live page's tiles."""
    alive = world.creatures
    return {
        "step": world.step_count, "alive": len(alive),
        "energy": sum(max(c.energy, 0.0) / c.max_energy for c in alive) / len(alive) if alive else None,
        "food": len(world.foods), "births": world.births, "deaths": sum(world.deaths.values()), "time": time.time(),
        "rules": dict(world.rules), "season": world.season(),
        "genes": {key: _spread([getattr(c, key) for c in alive if key != "neurons" or c.layers]) for key in GENE_TILES},
    }


GENE_TILES = ("size", "neurons", "vision_width", "vision_reach", "hearing", "fear_of_death", "love_of_babies",
              "learns_from_others", "mutations", "mutation_size", "diet", "calls", "kin", "water", "stealth", "armour")


def _spread(values: List[float]) -> Optional[List[float]]:
    """Lowest, average, highest of a gene among the living (None with nobody to measure)."""
    return [min(values), sum(values) / len(values), max(values)] if values else None


def _format(row: Dict[str, Value]) -> str:
    def show(value: Value) -> str:
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    return " | ".join(f"{key} {show(row[key])}" for key in ROW_KEYS)


class _Knobs:
    """Who turns the knobs: the owner's settings from the control file are applied once each, in Manual."""

    def __init__(self, control: Control) -> None:
        self.control = control
        self._applied = None
        self._applied_hunters = None
        self._applied_rules = None
        self._applied_edit = None

    def take(self, control: Control, dial: Dial, world: World) -> bool:
        """Takes the newest control: the owner's rule changes reach the world at once (in either mode); returns whether
        the owner's settings changed the world's."""
        self.control = control
        if control.rules is not None and control.rules != self._applied_rules:
            before = self._applied_rules or {}  # only what changed since: a cost Auto turned since stays as it is
            self._applied_rules = dict(control.rules)
            world.change_rules({k: v for k, v in control.rules.items() if before.get(k) != v})
        if control.edit is not None and control.edit["seq"] != self._applied_edit:
            self._applied_edit = control.edit["seq"]  # each change once (applying it again would change nothing)
            world.edit_genes(control.edit["id"], control.edit["genes"])
        if control.mode != "manual" or control.settings is None:
            self._applied = None  # back in Auto: the same settings sent again later are a new choice
            if control.mode == "auto" and control.hunters is not None and control.hunters != self._applied_hunters:
                self._applied_hunters = control.hunters  # the owner's hunters switch, in Auto
                dial.set_hunters(control.hunters)
                return True
            return False
        if control.settings == self._applied:
            return False
        self._applied = control.settings
        dial.set(control.settings)
        return True


def _resumed_from(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """For a resumed world: the old run and the save's step (and the run's seed and kind, from the save)."""
    if not args.resume:
        return None
    header = save_header(Path(args.resume))
    args.seed, args.kind = header["seed"], header["kind"]
    args.species = args.kind.split("-")[0]
    if args.species not in SPECIES:
        raise SaveError(f"{args.resume} is of a {args.kind} world; only a species world can be resumed")
    old_run = Path(args.resume).parent.parent
    if args.continue_run:
        if args.out is not None and Path(args.out).resolve() != old_run.resolve():
            raise SaveError(f"--continue goes on in the save's own run ({old_run}), not in {args.out}")
        args.out = str(old_run)
    elif args.out is None:
        args.out = new_run_folder(Path("runs"), args.seed, args.kind).as_posix()
    migrate_scoreboard(old_run / "scoreboard.csv")  # a run from before the chase, attacks, senses, values or brain columns
    return {"run": old_run, "step": header["step"], "history": _history(old_run, header["step"])}


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    source = _resumed_from(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.continue_run:  # the same run: its record gains the update
        record = json.loads((out / "run.json").read_text())
        record["steps"] = args.steps
        record.setdefault("updates", []).append({"step": source["step"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    else:
        record = {"seed": args.seed, "kind": args.kind, "steps": args.steps, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if source is not None and not args.continue_run:
        record["from"] = {"run": source["run"].name, "step": source["step"]}
    (out / "run.json").write_text(json.dumps(record))  # the run's record, for the viewer's Live and History
    control = read_control(out)  # the live page may have written it before starting the run
    if control is None:
        control = Control(args.mode, None, False)
        write_control(out, control)
    exam_rng = random.Random(args.seed + 2)  # who takes the exam: its own stream, so an exam never changes the world
    if source is None:
        world, dial = _new_world(args, control.rules or {})
        first_creatures = world.newcomers  # the soup's newcomers are counted from here
    else:
        world, dial, extra = load_world(Path(args.resume), learner_seed=args.seed)
        dial.max_level = args.max_level
        set_random_state(exam_rng, extra["exam_rng"])
        first_creatures = 0  # the first creatures were counted in the old run's windows
    dial.rules = world.rules  # Auto's target and when hunters come are the world's rules
    if world.terrain.water is not None and world.terrain.water.any_water():
        (out / "water.png").write_bytes(world.terrain.water.picture())  # the page draws the seas, lakes and rivers from it
    if control.settings is not None:  # the world starts from the settings the page asked for, in either mode
        dial.set(control.settings)
    knobs = _Knobs(control)
    knobs.take(control, dial, world)
    world.set_settings(dial.settings, round(dial.level, 2))

    board = Scoreboard()
    if source is not None:
        board.start_at(world)  # a new world's first row counts its first creatures as newcomers; a resumed one's does not
    exam: Optional[ExamResult] = None
    saved = set()

    def save() -> None:
        save_world(save_path(out, world.step_count), world, dial,
                   {"seed": args.seed, "kind": args.kind, "exam_rng": random_state(exam_rng)})
        saved.add(world.step_count)
    since_review: List[Dict[str, Value]] = []  # the rows since Auto's last look (their newcomers)
    recent = deque(maxlen=400)  # the latest rows; Auto measures its rates over the rules' last steps of them
    next_tick = time.perf_counter()  # while watched: when the next step may start
    now_written = -world.rules["now_seconds"]  # when now.json was last written

    csv_path = out / "scoreboard.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS)
        writer.writeheader()
        if source is not None:  # the story so far: the old run's rows up to the save (read before this file was opened)
            history = source["history"]
            writer.writerows(history)
            f.flush()
            recent.extend({key: _cell(row[key]) for key in ("step", "population", "energy")} for row in history)
        for step in itertools.count(world.step_count + 1):
            if args.steps is not None and step > args.steps:
                break
            watched = knobs.control.watched(time.time())
            r = world.rules  # the runner's numbers too: changed on the Rules page, they apply at once
            if watched or step % round(r["control_every"]) == 0:  # watched: every step, so Jump out is instant
                found = read_control(out)
                while found is not None and found.paused and not found.stop:  # paused: no step until Continue
                    os.utime(csv_path)  # still alive: the page keeps a paused world in Live
                    time.sleep(PAUSE_LOOK)
                    found = read_control(out)
                if found is not None:
                    if found.stop:
                        break
                    if knobs.take(found, dial, world):
                        world.set_settings(dial.settings, round(dial.level, 2))

            world.step(board.observe)
            if knobs.control.watched(time.time()):  # the owner is watching: draw it, then wait (the world is unchanged)
                write_json(out / "view.json", world_view(world))
                next_tick = max(next_tick, time.perf_counter()) + 1.0 / knobs.control.watch_speed
                time.sleep(max(0.0, next_tick - time.perf_counter()))

            if step % NOW_EVERY == 0 and time.perf_counter() - now_written >= r["now_seconds"]:  # it never changes the world
                write_json(out / "now.json", world_now(world))
                now_written = time.perf_counter()

            exam_every = round(r["exam_every"])
            if args.species and step % exam_every == 0:  # a measurement only: it never changes the world
                sample = exam_rng.sample(world.creatures, min(round(r["exam_brains"]), len(world.creatures)))
                exam_seed = args.seed * 100_000 + step // exam_every
                exam = run_exam(_exam_brains(world, sample), [c.size for c in sample], exam_seed, r)

            if step % round(r["row_every"]) == 0:
                row = board.row(world, exam, knobs.control.mode == "manual")
                exam = None  # each exam appears in exactly one row
                writer.writerow(row)
                f.flush()
                print(_format(row), flush=True)
                since_review.append(row)
                recent.append(row)

            every = round(world.rules["auto_every"])  # steps between two of Auto's looks
            if step % every == 0:
                if args.species and knobs.control.mode == "auto":  # a yardstick world is held at its level
                    soup = sum(r["newcomers"] for r in since_review) - (first_creatures if step == every else 0)
                    measured = [row for row in recent if row["step"] > step - r["auto_measure"]]
                    alive_rate, energy_rate = rates(measured)
                    dial.review(len(world.creatures), alive_rate, energy_rate, soup)
                    world.set_settings(dial.settings, round(dial.level, 2))
                since_review = []

            if step % round(r["save_every"]) == 0:
                save()


    write_json(out / "now.json", world_now(world))  # its last numbers
    if world.step_count not in saved and (source is None or world.step_count > source["step"]):
        save()  # stopped, or at its last step
    best = HallOfFame()
    for creature, brain in zip(world.creatures, _exam_brains(world, world.creatures)):
        if brain is not None:
            best.offer(brain, creature.food_eaten)
    best.save(out / "best")  # the brains of the living creatures that have eaten the most
    (out / "time.txt").write_text(f"{time.perf_counter() - started:.1f}\n")  # seconds the run took
    return csv_path


def _new_world(args: argparse.Namespace, rules: Dict[str, float]) -> Tuple[World, Dial]:
    """A new world on the starting rules with the owner's changes (from the Rules page), then its first creatures."""
    policy = SPECIES[args.species] if args.species else args.policy
    learner = None
    if policy == "learner":
        from .learner import Learner  # TensorFlow is loaded only for learning animals

        learner = Learner(ROOM_SLOTS, args.seed)
    dial = Dial(max_level=args.max_level)
    dial.turn_to(args.level)
    size = {**starting_rules(), **rules}  # the starting size unless the owner changed it
    world = World(size["planet_radius"], dial.settings, policy, random.Random(args.seed),
                  np.random.default_rng(args.seed), random.Random(args.seed + 1), learner=learner,
                  inherit=args.inherit)
    world.change_rules(rules)
    world.terrain = make_terrain(world.rules, world.rng, world.radius)  # its rocks, water (seas, islands, rivers), trees
    for _ in range(round(world.rules["first_creatures"])):
        world.add_newcomer()
    return world, dial


def _cell(text: str) -> Optional[float]:
    return None if text == "" else float(text)


def _history(old_run: Path, step: int) -> List[Dict[str, str]]:
    """The old run's scoreboard rows up to `step`."""
    rows = read_rows(old_run / "scoreboard.csv")
    if rows and tuple(rows[0]) != ROW_KEYS:
        raise SaveError(f"the scoreboard of {old_run} has other columns than this code writes")
    return [row for row in rows if int(row["step"]) <= step]


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
