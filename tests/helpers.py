"""Shared setup for the world, policy and visualization tests."""
import math
import random
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from src.brain import Brain
from src.creature import Creature
from src.rules import rule, starting_rules
from src.sphere import exp_map, local_frame
from src.settings import START, TODAY_SENSES, Settings
from src.world import World

TEST_RADIUS = 1.0e6  # a test planet: flat to ~1e-11 over the few units a test uses (above the rule's highest, on purpose)
SMALL_WORLD = {"planet_radius": 10.9, "room": 80.0, "first_creatures": 30.0, "soup": 10.0,
               "obstacles": 6.0, "trees": 8.0, "auto_target": 64.0,
               "save_every": 400.0, "row_every": 50.0, "auto_every": 100.0, "auto_measure": 200.0, "exam_every": 400.0,
               "exam_steps": 100.0, "control_every": 5.0}  # its runner 5× quicker: short test runs  # a test world stays small and quick, whatever a
# new world's starting rules are (the owner, 2026-11-26: "we should never run full life simulations as tests!")
BASE = np.array([0.36, 0.48, 0.8])  # where a test's local map is centred: any point, none special on a sphere
EAST, NORTH = local_frame(BASE)  # the map's x and y at its centre


def at(x: float, y: float, R: float = TEST_RADIUS) -> np.ndarray:
    """The point (x, y) of the local map: x along EAST, y along NORTH, as one straight walk from its centre."""
    return exp_map(BASE, EAST, NORTH, x, y, R)


def cap(x: float, y: float, r: float, R: float = TEST_RADIUS) -> tuple:
    """A round patch of ground of arc radius r centred at the map's (x, y): one cap of an obstacle."""
    return tuple(map(float, at(x, y, R))), float(r)


def tree(x: float, y: float, canopy: float, R: float = TEST_RADIUS):
    """A tree at the map's (x, y)."""
    from src.terrain import Tree
    return Tree(tuple(map(float, at(x, y, R))), float(canopy))


def face(creature: Creature, fx: float, fy: float) -> None:
    """Turn a creature to the map direction (fx, fy) at its place."""
    d = fx * EAST + fy * NORTH
    d = d - creature.p * float(d @ creature.p)
    creature.facing = d / np.linalg.norm(d)


NO_FOOD = replace(START, max_food=0, food_per_step=0)  # the easy start with nothing to eat
NO_FOOD_ATTACKS = replace(NO_FOOD, attacks=True)
WIDTH, REACH, HEARING = TODAY_SENSES
SENSES = (rule("vision_cost").start * WIDTH / 360.0 * math.pi * REACH ** 2
          + rule("hearing_cost").start * math.pi * HEARING ** 2)  # a step of today's senses
LIVING = NO_FOOD.step_cost + SENSES  # a step alive, size 1, today's senses


def living(size: float) -> float:
    """A step alive for a body of `size` with today's senses: a bigger body pays more, less than its size."""
    return NO_FOOD.step_cost * size ** rule("living_power").start + SENSES


def make_world(settings: Settings = NO_FOOD, seed: int = 0, evolving: bool = True, learner=None,
               species: str = "brain", inherit: str = "learned") -> World:
    return World(TEST_RADIUS, settings, species, random.Random(seed), np.random.default_rng(seed),
                 random.Random(seed + 1000), evolving, learner, inherit,
                 rules={**starting_rules(), **SMALL_WORLD, "planet_radius": TEST_RADIUS})  # hand-placed tests: a flat map


def fixed_brain(outputs: Sequence[float]) -> Brain:
    """A brain whose 4 outputs (move x, move y, eat, attack) are always `outputs` (±1 becomes ±0.999999)."""
    b2 = np.arctanh(np.clip(np.asarray(outputs, dtype=float), -0.999999, 0.999999))
    return Brain(np.zeros((Brain.N_IN, 16)), np.zeros(16), np.zeros((16, 4)), b2)


def add(world: World, x: float, y: float, size: float = 1.0, energy: float = 50.0, policy: str = "random",
        brain: Optional[Brain] = None, parent_id: int = 0, slot: Optional[int] = None) -> Creature:
    creature = Creature(world.next_id(), at(x, y, world.radius), size, energy, policy, brain, parent_id=parent_id,
                        slot=slot)
    world.add_creature(creature)
    face(creature, 1.0, 0.0)  # facing east on the map, as a creature made by hand did
    return creature


def populate(world: World, n: int = 20) -> None:
    for _ in range(n):
        world.add_newcomer()



def small_world():
    """While a test runs a world: a new world starts on the small test world's rules, as if they were the starting
    rules (the run itself is left as it is: its control, its saves, its Auto)."""
    from contextlib import ExitStack
    from unittest import mock

    small = lambda: {**starting_rules(), **SMALL_WORLD}  # noqa: E731
    stack = ExitStack()
    for where in ("src.evolve", "src.world", "src.dial"):
        stack.enter_context(mock.patch(f"{where}.starting_rules", small))
    return stack
