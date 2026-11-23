"""Run a seeded simulation headless with zero delay: a team of brains in rounds, or a baseline, with a scoreboard.

    python -m src.evolve --seed 1 --steps 100000 [--team evolution] [--policy random|rules] [--level N]
                         [--max-level M] [--window] [--out DIR]
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
from .rounds import ROUND_SIZE, Member, next_generation
from .scoreboard import ROW_KEYS, Scoreboard, Value
from .world import World

WIDTH, HEIGHT = 50, 30
START_ENERGY = (50.0, 100.0)
ROW_EVERY = 1000
ROUND_STEPS = 1000
EXAM_EVERY = 2000
EXAM_BRAINS = 20
UNLOCK_RATIO = 1.5
UNLOCK_MIN_POPULATION = 10
TEAMS = ("evolution",)
BASELINES = ("random", "rules")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a team of brains in rounds, or a baseline, with a scoreboard.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--team", choices=TEAMS, default="evolution")
    parser.add_argument("--policy", choices=BASELINES, default=None, help="run a baseline instead of a team")
    parser.add_argument("--level", type=int, default=1, help="start level; baselines stay at it")
    parser.add_argument("--max-level", type=int, default=5, help="highest level a team may unlock")
    parser.add_argument("--window", action="store_true", help="draw every step (no added delay)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = (f"runs/{args.seed}-{args.policy}-L{args.level}" if args.policy is not None
                    else f"runs/{args.seed}-{args.team}")
    return args


def should_unlock(ratios: List[Optional[float]], population: int, level: int, max_level: int) -> bool:
    """Up a level when the last two exam ratios are both at least 1.5 with at least 10 alive."""
    last_two = ratios[-2:]
    return (len(last_two) == 2 and all(r is not None and r >= UNLOCK_RATIO for r in last_two)
            and population >= UNLOCK_MIN_POPULATION and level < max_level)


def _spot(world: World) -> tuple[float, float]:
    return world.rng.uniform(1.0, WIDTH - 1.0), world.rng.uniform(1.0, HEIGHT - 1.0)


def _newborn(world: World, policy: str) -> Creature:
    x, y = _spot(world)
    size = world.rng.uniform(Creature.MIN_SIZE, Creature.MAX_SIZE)
    return Creature(world.next_id(), x, y, size, world.rng.uniform(*START_ENERGY), policy)


def _start_round(world: World, members: Sequence[Member]) -> List[Creature]:
    """A fresh world at the current level, with one creature per member."""
    world.creatures.clear()
    world.foods.clear()
    world.prefill_food()
    creatures = []
    for member in members:
        x, y = _spot(world)
        creatures.append(Creature(member.id, x, y, member.size, world.rng.uniform(*START_ENERGY), "brain",
                                  member.genome, parent_id=member.parent_id, generation=member.generation))
    world.creatures.extend(creatures)
    return creatures


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
    baseline = args.policy

    if baseline is not None:
        for _ in range(ROUND_SIZE):
            world.add_creature(_newborn(world, baseline))
        round_number: Optional[int] = None
    else:
        members = [Member(Brain.random(world.np_rng), world.rng.uniform(Creature.MIN_SIZE, Creature.MAX_SIZE), None,
                          world.next_id()) for _ in range(ROUND_SIZE)]
        round_creatures = _start_round(world, members)
        round_number = 1

    hall = HallOfFame()
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
            if baseline is not None:
                for _ in world.dead_this_step:
                    world.add_creature(_newborn(world, baseline))

            if baseline is None and step % EXAM_EVERY == 0:
                sample = world.rng.sample(world.creatures, min(EXAM_BRAINS, len(world.creatures)))
                exam_seed = args.seed * 100_000 + step // EXAM_EVERY
                exam_ratio = run_exam([c.brain for c in sample], [c.size for c in sample], world.level, exam_seed).ratio
                ratios.append(exam_ratio)
                if should_unlock(ratios, len(world.creatures), world.level.number, args.max_level):
                    world.set_level(by_number[world.level.number + 1])
                    ratios = []  # the next level is unlocked by two exams at the current one

            if step % ROW_EVERY == 0:
                row = board.row(world, exam_ratio, round_number)
                exam_ratio = None  # each exam appears in exactly one row
                writer.writerow(row)
                f.flush()
                print(_format(row), flush=True)

            if baseline is None and step % ROUND_STEPS == 0:
                scored = [(member, creature.food_eaten) for member, creature in zip(members, round_creatures)]
                for member, fitness in scored:
                    hall.offer(member.genome, fitness)
                if step < args.steps:
                    members = next_generation(scored, world.rng, world.np_rng, world.next_id)
                    round_creatures = _start_round(world, members)
                    round_number += 1

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
