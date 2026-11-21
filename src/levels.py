"""The world's levels, from forgiving to harsh, and the world rules every level shares."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Level:
    number: int
    food_energy: tuple[float, float]  # a new food's energy is drawn uniformly from this range
    max_food: int
    food_per_step: int  # food spawn candidates drawn every step
    move_cost: float  # energy per unit of distance moved
    step_cost: float  # energy every step, whatever the action
    attacks: bool  # whether an attack does anything at this level


# food_per_step = 3 is today's spawn rate: int(50 * 30 * 0.002).
LEVELS: tuple[Level, ...] = (
    Level(1, (20, 40), 200, 3, 0.5, 0.05, False),
    Level(2, (10, 25), 150, 3, 1.0, 0.10, False),
    Level(3, (10, 25), 80, 3, 1.0, 0.10, False),
    Level(4, (10, 25), 80, 3, 1.0, 0.10, True),
    Level(5, (5, 15), 50, 3, 1.5, 0.15, True),
)

HUNTER_SHARE = 0.8  # share of the energy an attacker takes that it keeps
MAX_AGE = 3000  # steps a creature lives at most
EAT_MISS_COST = 0.2  # energy lost trying to eat with no food in reach


def level(number: int) -> Level:
    for candidate in LEVELS:
        if candidate.number == number:
            return candidate
    raise ValueError(f"no level {number}; levels are 1 to {len(LEVELS)}")
