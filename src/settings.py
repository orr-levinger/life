"""The world's settings: how hard survival is right now, and the rules of life that never change.

The living world starts at START, the easy start. The dial (src/dial.py) nudges one setting at a time while nobody is
dying and takes a nudge back when too many die. The exam's world is a set of rules (src/rules.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SEEING = 0.05 / (math.pi * 3.0 ** 2)  # today's price of a square unit of senses: 0.05 a step for a reach of 3


@dataclass(frozen=True)
class Settings:
    max_food: float  # grass pieces at most at once (plants; fruit has its own room)
    food_energy: tuple[float, float]  # a new food's energy is drawn uniformly from this range
    food_per_step: float  # new food candidates per step, on average
    lifespan: tuple[float, float]  # a creature's own lifespan in steps is drawn from this range when it is born
    attacks: bool  # whether creatures can hunt each other
    fruit_rate: float  # fruit each tree drops per step, on average
    max_fruit: float  # fruit pieces at most at once: the trees' own room (the owner, 2026-11-28: "total fruts should also be nobed")
    # what everything costs is not here: the world's rules (src/rules.py, COSTS), the owner's to turn, never Auto's


# Starting values from the design; the plan tunes them by measurement.
START = Settings(max_food=400, food_energy=(20, 40), food_per_step=6.0, lifespan=(4000, 6000), attacks=False,
                 fruit_rate=0.02, max_fruit=300.0)  # the costs are the world's rules (src/rules.py, COSTS)

TODAY_SENSES = (360.0, 3.0, 1.5)  # width, reach, hearing of a creature no world gave senses to (the exam's; older saves)
