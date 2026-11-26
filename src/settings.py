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
    max_food: float  # food pieces at most at once
    food_energy: tuple[float, float]  # a new food's energy is drawn uniformly from this range
    food_per_step: float  # new food candidates per step, on average
    move_cost: float  # energy per unit of distance moved
    step_cost: float  # energy every step, whatever the creature does
    lifespan: tuple[float, float]  # a creature's own lifespan in steps is drawn from this range when it is born
    attacks: bool  # whether creatures can hunt each other
    fruit_rate: float  # fruit each tree drops per step, on average
    # every other cost (the owner, 2026-11-26: "Anything that costs must be part of the auto mechanism"); a world that
    # never had them as knobs (an old save, the exam) pays today's values
    sprint_power: float = 2.0  # faster than cruising costs the move × (speed ÷ cruising) to this power
    eat_miss: float = 0.2  # eating at nothing
    attack_cost: float = 1.0  # an attack, hit or miss, × size
    attack_damage: float = 5.0  # being hit: energy lost, × the attacker's size
    bump_cost: float = 0.5  # running into a rock
    vision_cost: float = SEEING  # seeing, per square unit of its cone, per step
    hearing_cost: float = SEEING  # hearing, per square unit of its circle, per step
    neuron_cost: float = 0.0005  # each neuron, per step
    stealth_cost: float = 0.3  # stealth: living × (1 + stealth × this)
    armour_cost: float = 0.3  # armour: living × (1 + armour × this)


# Starting values from the design; the plan tunes them by measurement.
START = Settings(max_food=400, food_energy=(20, 40), food_per_step=6.0, move_cost=0.05, step_cost=0.005,
                 lifespan=(4000, 6000), attacks=False, fruit_rate=0.02,  # every cost a tenth of today's: a gentle start
                 sprint_power=0.2, eat_miss=0.02, attack_cost=0.1, attack_damage=0.5, bump_cost=0.05,
                 vision_cost=SEEING / 10.0, hearing_cost=SEEING / 10.0, neuron_cost=0.00005, stealth_cost=0.03,
                 armour_cost=0.03)

TODAY_SENSES = (360.0, 3.0, 1.5)  # width, reach, hearing of a creature no world gave senses to (the exam's; older saves)
