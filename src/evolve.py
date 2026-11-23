"""Run one living world, headless with zero delay, and keep its scoreboard.

    python -m src.evolve --seed 1 --steps 100000 --species learning|instinct [--inherit learned|inborn]
                         [--level N] [--max-level M] [--window] [--out DIR]
    python -m src.evolve --seed 1 --steps 20000 --policy random|rules [--level N]      (a yardstick world)

The world is never restarted and no creature is ever ranked or picked: births, deaths and the soup are the world's own
rules (src/world.py). This runner starts the world, lets the difficulty dial review each window of steps (src/dial.py),
holds the intelligence exam as a measurement, and writes the scoreboard.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .creature import Creature
from .dial import Dial
from .exam import ExamResult, run_exam
from .hall_of_fame import HallOfFame
from .scoreboard import ROW_KEYS, Scoreboard, Value
from .settings import MAX_CREATURES
from .world import INHERIT, World

WIDTH, HEIGHT = 50, 30
FIRST_CREATURES = 30
WINDOW = 1000  # steps between two rows of the scoreboard, and between two reviews by the dial
EXAM_EVERY = 2000
EXAM_BRAINS = 20
SPECIES = {"learning": "learner", "instinct": "brain"}  # the name on the command line -> the creatures' policy
YARDSTICKS = ("random", "rules")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one living world and keep its scoreboard.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--species", choices=tuple(SPECIES))
    kind.add_argument("--policy", choices=YARDSTICKS, help="a yardstick world instead of a species")
    parser.add_argument("--level", type=float, default=0,
                        help="the difficulty the world starts at (nudges harder than the easy start); a yardstick world stays there")
    parser.add_argument("--max-level", type=int, default=None, help="the dial never turns past this level")
    parser.add_argument("--inherit", choices=INHERIT, default="learned",
                        help="what a learning baby gets: the parent's brain as it is now, or as the parent was born")
    parser.add_argument("--window", action="store_true", help="draw every step (no added delay)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.species:
        args.kind = args.species if args.inherit == "learned" else f"{args.species}-{args.inherit}"
    else:
        args.kind = f"{args.policy}-L{args.level:g}"
    if args.out is None:  # every run gets its own dated folder, so history is never overwritten
        args.out = f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{args.seed}-{args.kind}"
    return args


def _exam_brains(world: World, sample: Sequence[Creature]) -> List[Any]:
    """What the exam tests: a learner creature's current (learned) network, otherwise its brain."""
    return [world.learner.current(c.slot) if c.policy == "learner" else c.brain for c in sample]


def _format(row: Dict[str, Value]) -> str:
    def show(value: Value) -> str:
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    return " | ".join(f"{key} {show(row[key])}" for key in ROW_KEYS)


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({  # the run's record, for the viewer's Live and History
        "seed": args.seed, "kind": args.kind, "steps": args.steps, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }))
    policy = SPECIES[args.species] if args.species else args.policy
    learner = None
    if policy == "learner":
        from .learner import Learner  # TensorFlow is loaded only for learning animals

        learner = Learner(MAX_CREATURES, args.seed)
    dial = Dial(max_level=args.max_level)
    dial.turn_to(args.level)
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
    viz = _open_window() if args.window else None

    csv_path = out / "scoreboard.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS)
        writer.writeheader()
        for step in range(1, args.steps + 1):
            world.step(board.observe)

            if args.species and step % EXAM_EVERY == 0:  # a measurement only: it never changes the world
                sample = exam_rng.sample(world.creatures, min(EXAM_BRAINS, len(world.creatures)))
                exam_seed = args.seed * 100_000 + step // EXAM_EVERY
                exam = run_exam(_exam_brains(world, sample), [c.size for c in sample], exam_seed)

            if step % WINDOW == 0:
                row = board.row(world, exam)
                exam = None  # each exam appears in exactly one row
                writer.writerow(row)
                f.flush()
                print(_format(row), flush=True)
                soup = row["newcomers"] - (first_creatures if step == WINDOW else 0)
                if args.species:  # a yardstick world is held at its level
                    dial.review(len(world.creatures), row["energy"], soup)
                    world.set_settings(dial.settings, round(dial.level, 2))

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
