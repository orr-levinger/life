"""The exam: brains and random walkers in two identical copies of a fresh world, so crowding can't skew the comparison."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .brain import Brain
from .creature import Creature
from .levels import Level
from .world import World

WIDTH, HEIGHT = 50, 30
START_ENERGY_SHARE = 0.5  # creatures start half full


@dataclass(frozen=True)
class ExamResult:
    brain_bites_per_step: float
    walker_bites_per_step: float
    ratio: Optional[float]  # None when the walkers never bite


def _exam_world(policy: str, brains: Sequence[Optional[Brain]], sizes: Sequence[float], level: Level,
                seed: int) -> World:
    world = World(WIDTH, HEIGHT, level, random.Random(seed + 2), np.random.default_rng(seed), random.Random(seed + 1),
                  evolving=False)
    world.prefill_food()
    spots = random.Random(seed)
    for brain, size in zip(brains, sizes):
        x, y = spots.uniform(1.0, WIDTH - 1.0), spots.uniform(1.0, HEIGHT - 1.0)
        world.add_creature(Creature(world.next_id(), x, y, size, size * 100.0 * START_ENERGY_SHARE, policy, brain))
    return world


def exam_worlds(brains: List[Brain], sizes: List[float], level: Level, seed: int) -> Tuple[World, World]:
    """The brains in one copy; random walkers of the same sizes at the same spots in the other; the same food."""
    return (_exam_world("brain", brains, sizes, level, seed),
            _exam_world("random", [None] * len(sizes), sizes, level, seed))


def _bites_per_step(world: World, steps: int) -> float:
    for _ in range(steps):
        world.step()
    return world.total_bites / world.creature_steps if world.creature_steps else 0.0


def run_exam(brains: List[Brain], sizes: List[float], level: Level, seed: int, steps: int = 500) -> ExamResult:
    with_brains, with_walkers = exam_worlds(brains, sizes, level, seed)
    brains_score = _bites_per_step(with_brains, steps)
    walkers_score = _bites_per_step(with_walkers, steps)
    ratio = brains_score / walkers_score if walkers_score > 0 else None
    return ExamResult(brains_score, walkers_score, ratio)


def walker_score(level: Level, seed: int, steps: int = 500, n: int = 20) -> float:
    """Random walkers' bites per creature-step at this level: the yardstick the levels are tuned against."""
    sizes_rng = random.Random(seed)
    sizes = [sizes_rng.uniform(Creature.MIN_SIZE, Creature.MAX_SIZE) for _ in range(n)]
    return _bites_per_step(_exam_world("random", [None] * n, sizes, level, seed), steps)
