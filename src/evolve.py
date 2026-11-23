"""Run one living world, headless with zero delay, and keep its scoreboard.

    python -m src.evolve --seed 1 [--steps 100000 | --forever] --species learning|instinct [--mode auto|manual]
                         [--inherit learned|inborn] [--level N] [--max-level M] [--window] [--out DIR]
    python -m src.evolve --seed 1 --steps 20000 --policy random|rules [--level N]      (a yardstick world)

The world is never restarted and no creature is ever ranked or picked: births, deaths and the soup are the world's own
rules (src/world.py). This runner starts the world, takes the live page's control file every few steps (src/control.py:
the mode, the owner's knobs, Stop, a jump-in), lets Auto review the world every 1,000 steps (src/dial.py), holds the intelligence
exam as a measurement, and writes the scoreboard.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .control import MODES, Control, new_run_folder, read_control, write_control, write_json
from .creature import Creature
from .dial import Dial
from .exam import ExamResult, run_exam
from .hall_of_fame import HallOfFame
from .scoreboard import ROW_KEYS, Scoreboard, Value
from .settings import MAX_CREATURES
from .world import INHERIT, World

WIDTH, HEIGHT = 50, 30
FIRST_CREATURES = 30
DEFAULT_STEPS = 100000
CONTROL_EVERY = 25  # steps between two looks at the control file: a knob moved on the page arrives within a second
ROW_EVERY = 250  # steps between two rows of the scoreboard (a new point on every chart)
REVIEW_EVERY = 1000  # steps between two of Auto's reviews
EXAM_EVERY = 2000
EXAM_BRAINS = 20
SPECIES = {"learning": "learner", "instinct": "brain"}  # the name on the command line -> the creatures' policy
YARDSTICKS = ("random", "rules")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one living world and keep its scoreboard.")
    parser.add_argument("--seed", type=int, required=True)
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--steps", type=int, default=None, help=f"how long to run (default {DEFAULT_STEPS:,})")
    length.add_argument("--forever", action="store_true", help="run until Stop is pressed on the live page")
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--species", choices=tuple(SPECIES))
    kind.add_argument("--policy", choices=YARDSTICKS, help="a yardstick world instead of a species")
    parser.add_argument("--mode", choices=MODES, default="auto",
                        help="auto: the world turns its own knobs; manual: only the owner does (from the live page)")
    parser.add_argument("--level", type=float, default=0,
                        help="the difficulty the world starts at (nudges harder than the easy start)")
    parser.add_argument("--max-level", type=int, default=None, help="Auto never makes the world harder than this")
    parser.add_argument("--inherit", choices=INHERIT, default="learned",
                        help="what a learning baby gets: the parent's brain as it is now, or as the parent was born")
    parser.add_argument("--window", action="store_true", help="draw every step (no added delay)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.forever:
        args.steps = None
    elif args.steps is None:
        args.steps = DEFAULT_STEPS
    if args.species:
        args.kind = args.species if args.inherit == "learned" else f"{args.species}-{args.inherit}"
    else:
        args.kind = f"{args.policy}-L{args.level:g}"
    if args.out is None:  # every run gets its own dated folder, so history is never overwritten
        args.out = new_run_folder(Path("runs"), args.seed, args.kind).as_posix()
    return args


def _exam_brains(world: World, sample: Sequence[Creature]) -> List[Any]:
    """What the exam tests: a learner creature's current (learned) network, otherwise its brain."""
    return [world.learner.current(c.slot) if c.policy == "learner" else c.brain for c in sample]


def world_view(world: World) -> Dict[str, Any]:
    """Where everything is, for the page's picture of a watched world."""
    return {
        "step": world.step_count,
        "creatures": [{"x": round(c.x, 3), "y": round(c.y, 3), "radius": round(c.radius, 3),
                       "energy": round(max(c.energy, 0.0) / c.max_energy, 3),
                       "heading": [round(v, 3) for v in c.intended_vector], "did": c.last_action,
                       "generation": c.generation} for c in world.creatures],
        "foods": [{"x": round(f.x, 3), "y": round(f.y, 3), "radius": round(f.radius, 3)} for f in world.foods],
    }


def _format(row: Dict[str, Value]) -> str:
    def show(value: Value) -> str:
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    return " | ".join(f"{key} {show(row[key])}" for key in ROW_KEYS)


class _Knobs:
    """Who turns the knobs: the owner's settings from the control file are applied once each, in Manual."""

    def __init__(self, control: Control) -> None:
        self.control = control
        self._applied = None

    def take(self, control: Control, dial: Dial) -> bool:
        """Takes the newest control; returns whether the owner's settings changed the world's."""
        self.control = control
        if control.mode != "manual" or control.settings is None:
            self._applied = None  # back in Auto: the same settings sent again later are a new choice
            return False
        if control.settings == self._applied:
            return False
        self._applied = control.settings
        dial.set(control.settings)
        return True


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({  # the run's record, for the viewer's Live and History
        "seed": args.seed, "kind": args.kind, "steps": args.steps, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }))
    control = read_control(out)  # the live page may have written it before starting the run
    if control is None:
        control = Control(args.mode, None, False)
        write_control(out, control)
    policy = SPECIES[args.species] if args.species else args.policy
    learner = None
    if policy == "learner":
        from .learner import Learner  # TensorFlow is loaded only for learning animals

        learner = Learner(MAX_CREATURES, args.seed)
    dial = Dial(max_level=args.max_level)
    dial.turn_to(args.level)
    if control.settings is not None:  # the world starts from the settings the page asked for, in either mode
        dial.set(control.settings)
    knobs = _Knobs(control)
    knobs.take(control, dial)
    world = World(WIDTH, HEIGHT, dial.settings, policy, random.Random(args.seed),
                  np.random.default_rng(args.seed), random.Random(args.seed + 1), learner=learner,
                  inherit=args.inherit)
    world.set_settings(dial.settings, round(dial.level, 2))
    for _ in range(FIRST_CREATURES):
        world.add_newcomer()
    first_creatures = world.newcomers  # the soup's newcomers are counted from here

    board = Scoreboard()
    exam: Optional[ExamResult] = None
    exam_rng = random.Random(args.seed + 2)  # who takes the exam: its own stream, so an exam never changes the world
    since_review: List[Dict[str, Value]] = []
    next_tick = time.perf_counter()  # while watched: when the next step may start
    viz = _open_window() if args.window else None

    csv_path = out / "scoreboard.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS)
        writer.writeheader()
        for step in itertools.count(1):
            if args.steps is not None and step > args.steps:
                break
            watched = knobs.control.watched(time.time())
            if watched or step % CONTROL_EVERY == 0:  # watched: every step, so Jump out is instant
                found = read_control(out)
                if found is not None:
                    if found.stop:
                        break
                    if knobs.take(found, dial):
                        world.set_settings(dial.settings, round(dial.level, 2))

            world.step(board.observe)
            if knobs.control.watched(time.time()):  # the owner is watching: draw it, then wait (the world is unchanged)
                write_json(out / "view.json", world_view(world))
                next_tick = max(next_tick, time.perf_counter()) + 1.0 / knobs.control.watch_speed
                time.sleep(max(0.0, next_tick - time.perf_counter()))

            if args.species and step % EXAM_EVERY == 0:  # a measurement only: it never changes the world
                sample = exam_rng.sample(world.creatures, min(EXAM_BRAINS, len(world.creatures)))
                exam_seed = args.seed * 100_000 + step // EXAM_EVERY
                exam = run_exam(_exam_brains(world, sample), [c.size for c in sample], exam_seed)

            if step % ROW_EVERY == 0:
                row = board.row(world, exam, knobs.control.mode == "manual")
                exam = None  # each exam appears in exactly one row
                writer.writerow(row)
                f.flush()
                print(_format(row), flush=True)
                since_review.append(row)

            if step % REVIEW_EVERY == 0:
                if args.species and knobs.control.mode == "auto":  # a yardstick world is held at its level
                    soup = sum(r["newcomers"] for r in since_review) - (first_creatures if step == REVIEW_EVERY else 0)
                    energy = [r["energy"] for r in since_review if r["energy"] is not None]
                    dial.review(len(world.creatures), mean(energy) if energy else None, soup)
                    world.set_settings(dial.settings, round(dial.level, 2))
                since_review = []

            if viz is not None:
                viz.render(world)
                viz.fig.canvas.draw_idle()
                viz.fig.canvas.flush_events()

    best = HallOfFame()
    for creature, brain in zip(world.creatures, _exam_brains(world, world.creatures)):
        if brain is not None:
            best.offer(brain, creature.food_eaten)
    best.save(out / "best")  # the brains of the living creatures that have eaten the most
    (out / "time.txt").write_text(f"{time.perf_counter() - started:.1f}\n")  # seconds the run took
    if viz is not None:
        _keep_window_open()
    return csv_path


def _open_window():
    import matplotlib.pyplot as plt

    from .visualization import Visualizer

    plt.ion()
    viz = Visualizer(WIDTH, HEIGHT)
    plt.show(block=False)
    return viz


def _keep_window_open() -> None:
    import matplotlib.pyplot as plt

    plt.ioff()
    plt.show()


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
