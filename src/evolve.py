"""Run one living world, headless with zero delay, and keep its scoreboard.

    python -m src.evolve --seed 1 --steps 100000 --species instinct [--stage N] [--max-stage M] [--window] [--out DIR]
    python -m src.evolve --seed 1 --steps 20000 --policy random|rules [--stage N]      (a yardstick world)

The world is never restarted and no creature is ever ranked or picked: births, deaths and the soup are the world's own
rules (src/world.py). This runner only starts the world, holds the intelligence exam, moves the stage up when it is
passed, and writes the scoreboard.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .creature import Creature
from .exam import run_exam
from .hall_of_fame import HallOfFame
from .scoreboard import ROW_KEYS, Scoreboard, Value
from .stages import STAGES, stage
from .world import World

WIDTH, HEIGHT = 50, 30
FIRST_CREATURES = 30
ROW_EVERY = 1000
EXAM_EVERY = 2000
EXAM_BRAINS = 20
PASS_RATIO = 1.5
PASS_MIN_POPULATION = 10
SPECIES = {"instinct": "brain"}  # the species name on the command line -> the creatures' policy
FIRST_STAGE = {"instinct": 1}  # without learning and without death nothing can change, so instinct starts mortal
YARDSTICKS = ("random", "rules")
YARDSTICK_STAGE = 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one living world and keep its scoreboard.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--species", choices=tuple(SPECIES))
    kind.add_argument("--policy", choices=YARDSTICKS, help="a yardstick world instead of a species")
    parser.add_argument("--stage", type=int, default=None, help="the stage to start at; a yardstick world stays at it")
    parser.add_argument("--max-stage", type=int, default=STAGES[-1].number, help="the last stage a species may reach")
    parser.add_argument("--window", action="store_true", help="draw every step (no added delay)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.stage is None:
        args.stage = FIRST_STAGE[args.species] if args.species else YARDSTICK_STAGE
    if args.out is None:
        args.out = (f"runs/{args.seed}-{args.species}" if args.species
                    else f"runs/{args.seed}-{args.policy}-S{args.stage}")
    return args


def should_unlock(ratios: List[Optional[float]], population: int, stage_number: int, max_stage: int) -> bool:
    """The next stage opens when the last two exam scores are both at least 1.5 with at least 10 alive."""
    last_two = ratios[-2:]
    return (len(last_two) == 2 and all(r is not None and r >= PASS_RATIO for r in last_two)
            and population >= PASS_MIN_POPULATION and stage_number < max_stage)


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
    policy = SPECIES[args.species] if args.species else args.policy
    world = World(WIDTH, HEIGHT, stage(args.stage), policy, random.Random(args.seed),
                  np.random.default_rng(args.seed), random.Random(args.seed + 1))
    for _ in range(FIRST_CREATURES):
        world.add_newcomer()

    board = Scoreboard()
    ratios: List[Optional[float]] = []
    exam_ratio: Optional[float] = None
    viz = _open_window() if args.window else None

    csv_path = out / "scoreboard.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS)
        writer.writeheader()
        for step in range(1, args.steps + 1):
            world.step(board.observe)

            if args.species and step % EXAM_EVERY == 0:
                sample = world.rng.sample(world.creatures, min(EXAM_BRAINS, len(world.creatures)))
                exam_seed = args.seed * 100_000 + step // EXAM_EVERY
                exam_ratio = run_exam(_exam_brains(world, sample), [c.size for c in sample], exam_seed).ratio
                ratios.append(exam_ratio)
                if should_unlock(ratios, len(world.creatures), world.stage.number, args.max_stage):
                    world.set_stage(stage(world.stage.number + 1))
                    ratios = []  # each stage is passed by two exams of its own

            if step % ROW_EVERY == 0:
                row = board.row(world, exam_ratio)
                exam_ratio = None  # each exam appears in exactly one row
                writer.writerow(row)
                f.flush()
                print(_format(row), flush=True)

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
