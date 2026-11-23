"""The world's settings: how hard survival is right now, and the rules of life that never change.

The living world starts at START, the easy start. The dial (src/dial.py) nudges one setting at a time while nobody is
dying and takes a nudge back when too many die. The exam always happens in TEST_WORLD, which never changes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_food: float  # food pieces at most at once
    food_energy: tuple[float, float]  # a new food's energy is drawn uniformly from this range
    food_per_step: float  # new food candidates per step, on average
    move_cost: float  # energy per unit of distance moved
    step_cost: float  # energy every step, whatever the creature does
    lifespan: tuple[float, float]  # a creature's own lifespan in steps is drawn from this range when it is born
    attacks: bool  # whether creatures can hunt each other


# Starting values from the design; the plan tunes them by measurement.
START = Settings(max_food=200, food_energy=(20, 40), food_per_step=3.0, move_cost=0.5, step_cost=0.05,
                 lifespan=(4000, 6000), attacks=False)

# The intelligence exam always happens here, whatever the living world's settings, so scores compare across time.
TEST_WORLD = Settings(max_food=80, food_energy=(10, 25), food_per_step=3.0, move_cost=1.0, step_cost=0.10,
                      lifespan=(4000, 6000), attacks=False)

BIRTH_AT = 0.7  # a creature with this share of its maximum energy has a baby
BABY_SHARE = 0.4  # the share of the parent's energy the baby gets
LOW_ENERGY = 0.3  # past its lifespan, a creature with less than this share of its maximum energy dies of old age
MAX_CREATURES = 80  # a safety limit that only pauses births, to keep the simulation fast
SOUP_MIN = 10  # random newcomers appear while fewer than this many live
START_ENERGY = (50.0, 100.0)  # a newcomer's energy
HUNTER_SHARE = 0.8  # share of the energy an attacker takes that it keeps
EAT_MISS_COST = 0.2  # energy lost choosing to eat with no food within reach
MUTATION_RATE = 0.1  # share of a baby's brain weights that change
MUTATION_SCALE = 0.2  # how much a changed weight moves (standard deviation)
SIZE_MUTATION = 0.2  # a baby's size is its parent's × (1 ± up to this)
LR_MUTATION = 0.2  # a learning baby's learning speed is its parent's × e^N(0, this)
