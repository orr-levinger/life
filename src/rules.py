"""Every rule of the world, once: its words, its table on the Rules page, its starting value and its allowed range.

A world holds its own values (`World.rules`, a dict from rule name to value), changed on the Rules page and kept in its
saves; every module reads its numbers from there. Auto's six knobs (food, its energy and rate, the costs of moving and of
being alive, the lifespan) are the world's settings (`src/settings.py`), turned by Auto or by hand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

Rules = Dict[str, float]


@dataclass(frozen=True)
class Rule:
    name: str
    table: str  # "costs", "values" or "genes": the table it is on
    words: str  # what it is, in plain words
    start: float  # a new world's value
    low: float  # the allowed range
    high: float


def _gene(name: str, words: str, low: float, high: float, limits: tuple, change: float, start: Optional[float] = None,
          new: Optional[tuple] = None) -> tuple:
    """A gene's rules: its lowest and highest (each changeable within `limits`), what a newcomer gets (a start, or a
    range it draws from) and how much a baby's copy may change (× (1 ± up to this))."""
    rules = [Rule(f"gene.{name}.low", "genes", f"{words}: lowest", low, *limits),
             Rule(f"gene.{name}.high", "genes", f"{words}: highest", high, *limits)]
    if start is not None:
        rules.append(Rule(f"gene.{name}.start", "genes", f"{words}: a newcomer gets", start, *limits))
    if new is not None:
        rules += [Rule(f"gene.{name}.new_low", "genes", f"{words}: a newcomer gets at least", new[0], *limits),
                  Rule(f"gene.{name}.new_high", "genes", f"{words}: a newcomer gets at most", new[1], *limits)]
    rules.append(Rule(f"gene.{name}.change", "genes", f"{words}: a baby's copy changes by up to (share)", change, 0.0, 1.0))
    return tuple(rules)


RULES = (
    Rule("eat_miss", "costs", "Eating at nothing", 0.2, 0.0, 5.0),
    Rule("attack_cost", "costs", "An attack, hit or miss, × size", 1.0, 0.0, 10.0),
    Rule("attack_damage", "costs", "Being hit: energy lost, × the attacker's size", 5.0, 0.0, 50.0),
    Rule("baby_share", "costs", "Having a baby: the share of its energy given to the baby", 0.4, 0.05, 0.9),
    Rule("sense_cost", "costs", "Senses, per square unit, each step", 0.05 / (math.pi * 3.0 ** 2), 0.0, 0.01),
    Rule("world_width", "values", "The world's width", 50.0, 20.0, 200.0),
    Rule("world_height", "values", "The world's height", 30.0, 12.0, 120.0),
    Rule("room", "values", "Room for this many creatures (births pause when full)", 80.0, 10.0, 80.0),
    Rule("soup", "values", "Random newcomers appear while fewer than this many live", 10.0, 0.0, 50.0),
    Rule("newcomer_energy_low", "values", "A newcomer's energy: at least", 50.0, 1.0, 500.0),
    Rule("newcomer_energy_high", "values", "A newcomer's energy: at most", 100.0, 1.0, 500.0),
    Rule("tank", "values", "Energy tank, × size", 100.0, 20.0, 500.0),
    Rule("body_radius", "values", "Body radius, × size", 0.2, 0.05, 1.0),
    Rule("cruise_speed", "values", "Cruising speed, ÷ size", 0.25, 0.05, 1.0),
    Rule("sprint_speed", "values", "Sprint top speed, × √size", 0.4, 0.1, 1.5),
    Rule("birth_at", "values", "Has a baby at this share of its tank", 0.7, 0.3, 1.0),
    Rule("low_energy", "values", "Past its lifespan it dies below this share of its tank", 0.3, 0.0, 0.9),
    Rule("eat_range", "values", "Eating reach, × both radii", 3.0, 1.0, 10.0),
    Rule("attack_range", "values", "Attack reach, × both radii", 3.0, 1.0, 10.0),
    Rule("hunter_share", "values", "A hunter keeps this share of what it takes", 0.8, 0.0, 1.0),
    Rule("corpse_energy", "values", "A corpse's food, × the prey's size", 8.0, 0.0, 50.0),
    Rule("corpse_steps", "values", "A corpse lasts this many steps", 5.0, 1.0, 100.0),
    Rule("brain_change_share", "values", "Share of a baby's brain weights that change", 0.1, 0.0, 1.0),
    Rule("brain_change_size", "values", "How much a changed weight moves", 0.2, 0.0, 2.0),
    Rule("auto_target", "values", "Auto aims at this many alive", 64.0, 10.0, 80.0),
    Rule("auto_hunters_at", "values", "Auto switches hunters on at this difficulty", 24.0, 0.0, 200.0),
    Rule("auto_every", "values", "Auto looks every this many steps", 500.0, 100.0, 5000.0),
) + _gene("size", "Size", 0.5, 2.0, (0.1, 5.0), 0.2, new=(0.5, 2.0)) \
  + _gene("vision_width", "Vision width (degrees)", 30.0, 360.0, (1.0, 360.0), 0.2, new=(30.0, 360.0)) \
  + _gene("vision_reach", "Vision reach", 1.0, 8.0, (0.1, 20.0), 0.2, new=(1.0, 8.0)) \
  + _gene("hearing", "Hearing", 0.0, 4.0, (0.0, 20.0), 0.2, new=(0.0, 4.0)) \
  + _gene("fear", "Fear of death", 0.1, 100.0, (0.0, 1000.0), 0.2, start=10.0) \
  + _gene("love", "Love of babies", 0.1, 100.0, (0.0, 1000.0), 0.2, start=10.0) \
  + _gene("learns", "Learning from others", 0.01, 1.0, (0.0, 1.0), 0.2, start=0.5) \
  + _gene("learning_speed", "Learning speed", 1e-5, 1e-2, (1e-6, 0.1), 0.2, start=3e-4)

_BY_NAME = {rule.name: rule for rule in RULES}
ROOM_SLOTS = int(_BY_NAME["room"].high)  # a learner keeps a slot for each creature there may ever be room for


def rule(name: str) -> Rule:
    return _BY_NAME[name]


def starting_rules() -> Rules:
    """A new world's rules."""
    return {r.name: r.start for r in RULES}


def check(rules: Rules, name: str, value: float) -> Optional[str]:
    """Why `value` cannot be rule `name` in a world with `rules` (outside its range, or a gene's lowest above its
    highest), in plain words; None when it can."""
    found = _BY_NAME.get(name)
    if found is None:
        return f"{name} is an unknown rule"
    if value < found.low:
        return f"{found.words} must be at least {found.low:g}"
    if value > found.high:
        return f"{found.words} can be at most {found.high:g}"
    for low, high in ((".low", ".high"), (".new_low", ".new_high")):
        for mine, other, below in ((low, high, True), (high, low, False)):
            if name.startswith("gene.") and name.endswith(mine):
                partner = rules.get(name[:-len(mine)] + other)
                if partner is not None and (value > partner if below else value < partner):
                    return f"{found.words.split(':')[0]}: the lowest must not be above the highest"
    return None


def gene_limits(rules: Rules, gene: str) -> tuple:
    return rules[f"gene.{gene}.low"], rules[f"gene.{gene}.high"]
