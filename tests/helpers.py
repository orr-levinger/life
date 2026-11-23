"""Shared setup for the world, policy and visualization tests."""
import random
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from src.brain import Brain
from src.creature import Creature
from src.levels import LEVELS, Level
from src.world import World

NO_FOOD = replace(LEVELS[0], max_food=0, food_per_step=0)
NO_FOOD_ATTACKS = replace(LEVELS[3], max_food=0, food_per_step=0)


def make_world(level: Level = NO_FOOD, seed: int = 0, evolving: bool = True, learner=None) -> World:
    return World(50, 30, level, random.Random(seed), np.random.default_rng(seed), random.Random(seed + 1000), evolving,
                 learner)


def fixed_brain(outputs: Sequence[float]) -> Brain:
    """A brain whose 4 outputs are always `outputs` (±1 becomes ±0.999999)."""
    b2 = np.arctanh(np.clip(np.asarray(outputs, dtype=float), -0.999999, 0.999999))
    return Brain(np.zeros((12, 16)), np.zeros(16), np.zeros((16, 4)), b2)


def add(world: World, x: float, y: float, size: float = 1.0, energy: float = 50.0, policy: str = "random",
        brain: Optional[Brain] = None, parent_id: int = 0, slot: Optional[int] = None) -> Creature:
    creature = Creature(world.next_id(), x, y, size, energy, policy, brain, parent_id=parent_id, slot=slot)
    world.add_creature(creature)
    return creature


def populate(world: World, n: int = 20) -> None:
    for _ in range(n):
        add(world, world.rng.uniform(1, 49), world.rng.uniform(1, 29), world.rng.uniform(0.5, 2.0),
            world.rng.uniform(50, 100), "brain", Brain.random(world.np_rng))
