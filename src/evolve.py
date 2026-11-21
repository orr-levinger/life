"""Run a seeded simulation headless with zero delay: evolve brains (or run a baseline), write the scoreboard.

    python -m src.evolve --seed 1 --steps 100000 [--policy brain|random|rules] [--level N] [--max-level M]
                         [--window] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .brain import Brain
from .creature import Creature
from .exam import run_exam
from .hall_of_fame import HallOfFame
from .levels import LEVELS, Level
from .scoreboard import ROW_KEYS, Scoreboard, Value
from .world import World

WIDTH, HEIGHT = 50, 30
START_CREATURES = 20
START_ENERGY = (50.0, 100.0)
ROW_EVERY = 1000
EXAM_EVERY = 2000
EXAM_BRAINS = 20
UNLOCK_RATIO = 1.5
UNLOCK_MIN_POPULATION = 10
RESTART_MUTATION = (0.1, 0.2)  # rate, scale


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve creature brains, or run a baseline, and keep a scoreboard.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--policy", choices=Creature.POLICIES, default="brain")
    parser.add_argument("--level", type=int, default=1, help="start level; baselines stay at it")
    parser.add_argument("--max-level", type=int, default=5, help="highest level a brain run may unlock")
    parser.add_argument("--window", action="store_true", help="draw every step (no added delay)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = (f"runs/{args.seed}-brain" if args.policy == "brain"
                    else f"runs/{args.seed}-{args.policy}-L{args.level}")
    return args


def should_unlock(ratios: List[Optional[float]], population: int, level: int, max_level: int) -> bool:
    """Up a level when the last two exam ratios are both at least 1.5 with at least 10 alive."""
    last_two = ratios[-2:]
    return (len(last_two) == 2 and all(r is not None and r >= UNLOCK_RATIO for r in last_two)
            and population >= UNLOCK_MIN_POPULATION and level < max_level)


def _spawn(world: World, policy: str, brains: Sequence[Optional[Brain]]) -> None:
    for brain in brains:
        x, y = world.rng.uniform(1.0, WIDTH - 1.0), world.rng.uniform(1.0, HEIGHT - 1.0)
        size = world.rng.uniform(Creature.MIN_SIZE, Creature.MAX_SIZE)
        world.add_creature(Creature(world.next_id(), x, y, size, world.rng.uniform(*START_ENERGY), policy, brain))


def restart_population(world: World, hall: HallOfFame, policy: str) -> None:
    """After an extinction: brains restart from mutated hall-of-fame brains; baselines start fresh."""
    if policy != "brain":
        _spawn(world, policy, [None] * START_CREATURES)
        return
    parents = hall.brains()
    if not parents:
        raise RuntimeError("extinction with an empty hall of fame: every dead brain should have been offered")
    rate, scale = RESTART_MUTATION
    _spawn(world, policy, [parents[i % len(parents)].mutate(world.np_rng, rate, scale) for i in range(START_CREATURES)])


def _format(row: Dict[str, Value]) -> str:
    def show(value: Value) -> str:
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    return " | ".join(f"{key} {show(row[key])}" for key in ROW_KEYS)


def run(args: argparse.Namespace, levels: Sequence[Level] = LEVELS) -> Path:
    by_number = {lv.number: lv for lv in levels}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = World(WIDTH, HEIGHT, by_number[args.level], random.Random(args.seed), np.random.default_rng(args.seed),
                  random.Random(args.seed + 1))
    first_brains = ([Brain.random(world.np_rng) for _ in range(START_CREATURES)] if args.policy == "brain"
                    else [None] * START_CREATURES)
    _spawn(world, args.policy, first_brains)

    hall = HallOfFame()
    board = Scoreboard()
    restarts = 0
    ratios: List[Optional[float]] = []
    exam_ratio: Optional[float] = None
    viz = _open_window() if args.window else None

    csv_path = out / "scoreboard.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS)
        writer.writeheader()
        for step in range(1, args.steps + 1):
            world.step(board.observe)
            for dead in world.dead_this_step:
                if dead.brain is not None:
                    hall.offer(dead.brain, dead.food_eaten)
            if not world.creatures:
                restart_population(world, hall, args.policy)
                restarts += 1

            if args.policy == "brain" and step % EXAM_EVERY == 0:
                sample = world.rng.sample(world.creatures, min(EXAM_BRAINS, len(world.creatures)))
                exam_seed = args.seed * 100_000 + step // EXAM_EVERY
                exam_ratio = run_exam([c.brain for c in sample], [c.size for c in sample], world.level, exam_seed).ratio
                ratios.append(exam_ratio)
                if should_unlock(ratios, len(world.creatures), world.level.number, args.max_level):
                    world.set_level(by_number[world.level.number + 1])
                    ratios = []  # the next level is unlocked by two exams at the current one

            if step % ROW_EVERY == 0:
                row = board.row(world, exam_ratio, restarts)
                exam_ratio = None  # each exam appears in exactly one row
                writer.writerow(row)
                f.flush()
                print(_format(row), flush=True)

            if viz is not None:
                viz.render(world)
                viz.fig.canvas.draw_idle()
                viz.fig.canvas.flush_events()

    hall.save(out / "best")
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
