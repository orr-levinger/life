"""The intelligence exam: brains and random wanderers in two identical copies of the fixed test world.

The test world never changes, whatever stage the living world is at, so scores compare across time, and nobody is
born or dies in it, so crowding cannot skew the comparison.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .brain import Brain
from .creature import Creature
from .stages import TEST_WORLD
from .world import World

WIDTH, HEIGHT = 50, 30
START_ENERGY_SHARE = 0.5  # creatures start half full


@dataclass(frozen=True)
class ExamResult:
    brain_bites_per_step: float
    walker_bites_per_step: float
    ratio: Optional[float]  # None when the wanderers never bite


def _exam_world(policy: str, brains: Sequence[Any], sizes: Sequence[float], seed: int) -> World:
    """`brains` are Brains for policy "brain", NetWeights for "learner" (their learning is paused), None otherwise."""
    learner = None
    if policy == "learner":
        from .learner import Learner  # imported only when learner networks are examined: TensorFlow stays out otherwise

        learner = Learner(len(sizes), seed)
        for slot, weights in enumerate(brains):
            learner.install(slot, weights, lr=0.0)
    world = World(WIDTH, HEIGHT, TEST_WORLD, policy, random.Random(seed + 2), np.random.default_rng(seed),
                  random.Random(seed + 1), evolving=False, learner=learner)
    world.prefill_food()
    spots = random.Random(seed)
    for slot, (brain, size) in enumerate(zip(brains, sizes)):
        x, y = spots.uniform(1.0, WIDTH - 1.0), spots.uniform(1.0, HEIGHT - 1.0)
        world.add_creature(Creature(world.next_id(), x, y, size, size * 100.0 * START_ENERGY_SHARE, policy,
                                    brain if policy == "brain" else None,
                                    slot=slot if policy == "learner" else None))
    return world


def exam_worlds(brains: Sequence[Any], sizes: List[float], seed: int) -> Tuple[World, World]:
    """The brains (Brains or learner NetWeights) in one copy; random wanderers of the same sizes at the same spots in
    the other; the same food."""
    policy = "brain" if all(isinstance(brain, Brain) for brain in brains) else "learner"
    return (_exam_world(policy, brains, sizes, seed), _exam_world("random", [None] * len(sizes), sizes, seed))


def _bites_per_step(world: World, steps: int) -> float:
    for _ in range(steps):
        world.step()
    return world.total_bites / world.creature_steps if world.creature_steps else 0.0


def run_exam(brains: Sequence[Any], sizes: List[float], seed: int, steps: int = 500) -> ExamResult:
    with_brains, with_walkers = exam_worlds(brains, sizes, seed)
    brains_score = _bites_per_step(with_brains, steps)
    walkers_score = _bites_per_step(with_walkers, steps)
    ratio = brains_score / walkers_score if walkers_score > 0 else None
    return ExamResult(brains_score, walkers_score, ratio)


def walker_score(seed: int, steps: int = 500, n: int = 20) -> float:
    """Random wanderers' bites per creature-step in the test world: the yardstick an exam score is measured against."""
    sizes_rng = random.Random(seed)
    sizes = [sizes_rng.uniform(Creature.MIN_SIZE, Creature.MAX_SIZE) for _ in range(n)]
    return _bites_per_step(_exam_world("random", [None] * n, sizes, seed), steps)
