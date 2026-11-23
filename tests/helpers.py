"""Shared setup for the world, policy and visualization tests."""
import random
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from src.brain import Brain
from src.creature import Creature
from src.stages import STAGES, Stage
from src.world import World

GARDEN_NO_FOOD = replace(STAGES[0], max_food=0, food_per_step=0)
NO_FOOD = replace(STAGES[1], max_food=0, food_per_step=0)  # a mortal stage with nothing to eat
NO_FOOD_ATTACKS = replace(STAGES[3], max_food=0, food_per_step=0)


def make_world(stage: Stage = NO_FOOD, seed: int = 0, evolving: bool = True, learner=None, species: str = "brain",
               inherit: str = "learned") -> World:
    return World(50, 30, stage, species, random.Random(seed), np.random.default_rng(seed), random.Random(seed + 1000),
                 evolving, learner, inherit)


def fixed_brain(outputs: Sequence[float]) -> Brain:
    """A brain whose 4 outputs (move x, move y, eat, attack) are always `outputs` (±1 becomes ±0.999999)."""
    b2 = np.arctanh(np.clip(np.asarray(outputs, dtype=float), -0.999999, 0.999999))
    return Brain(np.zeros((12, 16)), np.zeros(16), np.zeros((16, 4)), b2)


def add(world: World, x: float, y: float, size: float = 1.0, energy: float = 50.0, policy: str = "random",
        brain: Optional[Brain] = None, parent_id: int = 0, slot: Optional[int] = None) -> Creature:
    creature = Creature(world.next_id(), x, y, size, energy, policy, brain, parent_id=parent_id, slot=slot)
    world.add_creature(creature)
    return creature


def populate(world: World, n: int = 20) -> None:
    for _ in range(n):
        world.add_newcomer()
