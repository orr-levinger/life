"""The world's stages, from a Garden where nobody dies to a harsh world with hunters, and the rules of life they share.

Two rules hold through any tuning: food never increases from one stage to the next, and no stage is tuned to make an
exam score reachable (the exam has its own fixed test world).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    mortal: bool  # whether anything can die here
    food_energy: tuple[float, float]  # a new food's energy is drawn uniformly from this range
    max_food: int
    food_per_step: int  # food spawn candidates drawn every step
    move_cost: float  # energy per unit of distance moved
    step_cost: float  # energy every step, whatever the creature does
    attacks: bool  # whether creatures can hunt each other


# Starting numbers from the design; tuned by measurement in the plan's Task 5.
STAGES: tuple[Stage, ...] = (
    Stage(0, "Garden", False, (20, 40), 200, 3, 0.5, 0.05, False),
    Stage(1, "Mortal", True, (20, 40), 200, 3, 0.5, 0.05, False),
    Stage(2, "Scarcity", True, (10, 25), 100, 3, 1.0, 0.10, False),
    Stage(3, "Hunters", True, (10, 25), 100, 3, 1.0, 0.10, True),
    Stage(4, "Harsh", True, (5, 15), 50, 3, 1.5, 0.15, True),
)

# The intelligence exam always happens here, whatever the living world's stage, so scores compare across time.
TEST_WORLD = Stage(-1, "Test world", False, (10, 25), 80, 3, 1.0, 0.10, False)

BIRTH_AT = 0.7  # a creature with this share of its maximum energy has a baby
BABY_SHARE = 0.4  # the share of the parent's energy the baby gets
MAX_AGE = 5000  # steps a creature lives at most, in a mortal stage
MAX_CREATURES = 80  # a safety limit that only pauses births, to keep the simulation fast
SOUP_MIN = 10  # in a mortal stage, random newcomers appear while fewer than this many live
START_ENERGY = (50.0, 100.0)  # a newcomer's energy
HUNTER_SHARE = 0.8  # share of the energy an attacker takes that it keeps
EAT_MISS_COST = 0.2  # energy lost choosing to eat with no food within reach
MUTATION_RATE = 0.1  # share of a baby's brain weights that change
MUTATION_SCALE = 0.2  # how much a changed weight moves (standard deviation)
SIZE_MUTATION = 0.2  # a baby's size is its parent's × (1 ± up to this)
LR_MUTATION = 0.2  # a learning baby's learning speed is its parent's × e^N(0, this)


def stage(number: int) -> Stage:
    for candidate in STAGES:
        if candidate.number == number:
            return candidate
    raise ValueError(f"no stage {number}; stages are 0 to {len(STAGES) - 1}")
