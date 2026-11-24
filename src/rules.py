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
    group: str  # its group on the Rules page (GROUPS)
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
    Rule("sprint_power", "costs", "Sprinting: faster than cruising costs the move × (speed ÷ cruising) to this power", 2.0,
         1.0, 4.0),
    Rule("neuron_cost", "costs", "Each neuron, each step", 0.0005, 0.0, 0.01),
    Rule("vision_cost", "costs", "Vision, per square unit seen, each step", 0.05 / (math.pi * 3.0 ** 2), 0.0, 0.01),
    Rule("hearing_cost", "costs", "Hearing, per square unit heard, each step", 0.05 / (math.pi * 3.0 ** 2), 0.0, 0.01),
    Rule("world_width", "world", "The world's width", 50.0, 20.0, 200.0),
    Rule("world_height", "world", "The world's height", 30.0, 12.0, 120.0),
    Rule("room", "world", "Room for this many creatures (births pause when full)", 80.0, 10.0, 80.0),
    Rule("soup", "world", "Random newcomers appear while fewer than this many live", 10.0, 0.0, 50.0),
    Rule("newcomer_energy_low", "world", "A newcomer's energy: at least", 50.0, 1.0, 500.0),
    Rule("newcomer_energy_high", "world", "A newcomer's energy: at most", 100.0, 1.0, 500.0),
    Rule("tank", "body", "Energy tank, × size", 100.0, 20.0, 500.0),
    Rule("body_radius", "body", "Body radius, × size", 0.2, 0.05, 1.0),
    Rule("cruise_speed", "body", "Cruising speed, ÷ size", 0.25, 0.05, 1.0),
    Rule("sprint_speed", "body", "Sprint top speed, × √size", 0.4, 0.1, 1.5),
    Rule("birth_at", "birth", "Has a baby at this share of its tank", 0.7, 0.3, 1.0),
    Rule("low_energy", "birth", "Past its lifespan it dies below this share of its tank", 0.3, 0.0, 0.9),
    Rule("eat_range", "senses", "Eating reach, × both radii", 3.0, 1.0, 10.0),
    Rule("attack_range", "senses", "Attack reach, × both radii", 3.0, 1.0, 10.0),
    Rule("hunter_share", "senses", "A hunter keeps this share of what it takes", 0.8, 0.0, 1.0),
    Rule("corpse_energy", "senses", "A corpse's food, × the prey's size", 8.0, 0.0, 50.0),
    Rule("corpse_steps", "senses", "A corpse lasts this many steps", 5.0, 1.0, 100.0),
    Rule("first_creatures", "world", "A new world starts with this many creatures", 30.0, 1.0, 80.0),
    Rule("bite", "body", "A bite of food: the most it eats in one step", 1.0, 0.1, 20.0),
    Rule("threat_ratio", "senses", "Bigger means at least this many times its size (for fleeing and chasing)", 1.5, 1.05, 5.0),
    Rule("learn_rollout", "learning", "Steps of its own life between two learning rounds", 64.0, 8.0, 1024.0),
    Rule("learn_passes", "learning", "Passes over those steps in a round", 4.0, 1.0, 20.0),
    Rule("learn_clip", "learning", "How far one round may move its choices (share)", 0.2, 0.01, 1.0),
    Rule("learn_discount", "learning", "How much a reward one step later counts", 0.99, 0.5, 0.999),
    Rule("learn_credit", "learning", "How far back a reward is credited to earlier steps", 0.95, 0.0, 1.0),
    Rule("learn_explore", "learning", "How much it is rewarded for trying different things", 0.01, 0.0, 0.5),
    Rule("learn_guess", "learning", "How much learning to guess what a place is worth counts", 0.5, 0.0, 5.0),
    Rule("learn_spread", "learning", "How spread a newborn's moves are at first", 0.5, 0.01, 2.0),
    Rule("auto_per_creatures", "auto", "One nudge per this many creatures away from its target", 8.0, 1.0, 80.0),
    Rule("auto_per_alive_rate", "auto", "One nudge per this share a 1,000 steps the number alive moves", 0.10, 0.01, 1.0),
    Rule("auto_per_energy_rate", "auto", "One nudge per this share of a full tank a 1,000 steps energy falls", 0.05, 0.01, 1.0),
    Rule("auto_max_nudges", "auto", "The most nudges one look may make", 4.0, 1.0, 20.0),
    Rule("auto_measure", "auto", "It measures the rates over this many last steps", 1000.0, 250.0, 10000.0),
    Rule("auto_nudge", "auto", "One nudge changes a knob by this share (changing it rescales the difficulty number)", 0.05,
         0.01, 0.3),
    Rule("exam_every", "exam", "The exam every this many steps (changing the exam makes older scores incomparable)", 2000.0,
         250.0, 20000.0),
    Rule("exam_brains", "exam", "Creatures taking each exam", 20.0, 2.0, 80.0),
    Rule("exam_steps", "exam", "Steps an exam lasts", 500.0, 50.0, 5000.0),
    Rule("exam_max_food", "exam", "The exam world: food pieces at most", 80.0, 5.0, 400.0),
    Rule("exam_food_energy_low", "exam", "The exam world: a food's energy at least", 10.0, 1.0, 100.0),
    Rule("exam_food_energy_high", "exam", "The exam world: a food's energy at most", 25.0, 1.0, 100.0),
    Rule("exam_food_per_step", "exam", "The exam world: new food a step", 3.0, 0.1, 20.0),
    Rule("exam_move_cost", "exam", "The exam world: cost of moving, per unit", 1.0, 0.0, 5.0),
    Rule("exam_step_cost", "exam", "The exam world: cost of being alive, each step", 0.10, 0.0, 2.0),
    Rule("exam_start_share", "exam", "The exam: creatures start with this share of a full tank", 0.5, 0.05, 1.0),
    Rule("save_every", "runner", "The world saves itself every this many steps", 2000.0, 250.0, 20000.0),
    Rule("row_every", "runner", "A row of numbers (a point on the charts) every this many steps", 250.0, 50.0, 5000.0),
    Rule("control_every", "runner", "It looks at the page's changes every this many steps", 25.0, 1.0, 500.0),
    Rule("now_seconds", "runner", "Its live numbers for the tiles at most every this many seconds", 0.5, 0.1, 10.0),
    Rule("auto_target", "auto", "Auto aims at this many alive", 64.0, 10.0, 80.0),
    Rule("auto_hunters_at", "auto", "Auto switches hunters on at this difficulty", 24.0, 0.0, 200.0),
    Rule("auto_every", "auto", "Auto looks every this many steps", 500.0, 100.0, 5000.0),
) + _gene("size", "Size", 0.5, 2.0, (0.1, 5.0), 0.05, new=(0.5, 2.0)) \
  + _gene("vision_width", "Vision width (degrees)", 30.0, 120.0, (1.0, 120.0), 0.05, new=(30.0, 120.0)) \
  + _gene("vision_reach", "Vision reach", 1.0, 10.0, (0.1, 20.0), 0.05, new=(1.0, 10.0)) \
  + _gene("hearing", "Hearing", 0.0, 4.0, (0.0, 20.0), 0.05, new=(0.0, 4.0)) \
  + _gene("fear", "Fear of death", 0.1, 100.0, (0.0, 1000.0), 0.05, start=10.0) \
  + _gene("love", "Love of babies", 0.1, 100.0, (0.0, 1000.0), 0.05, start=10.0) \
  + _gene("learns", "Learning from others", 0.01, 1.0, (0.0, 1.0), 0.05, start=0.5) \
  + _gene("learning_speed", "Learning speed", 1e-5, 1e-2, (1e-6, 0.1), 0.05, start=3e-4)

RULES += (
    Rule("gene.weights.share", "genes", "Brain weights: the share that changes when they are a baby's changed gene", 0.02,
         0.0, 1.0),
    Rule("gene.weights.size", "genes", "Brain weights: how much a changed weight moves", 0.02, 0.0, 1.0),
    Rule("gene.neurons.low", "genes", "Neurons in a layer: lowest", 1.0, 1.0, 64.0),
    Rule("gene.neurons.high", "genes", "Neurons in a layer: highest", 64.0, 1.0, 64.0),
    Rule("gene.neurons.start", "genes", "Neurons in a layer: a newcomer gets", 32.0, 1.0, 64.0),
    Rule("gene.layers.low", "genes", "Layers: lowest", 1.0, 1.0, 3.0),
    Rule("gene.layers.high", "genes", "Layers: highest", 3.0, 1.0, 3.0),
    Rule("gene.layers.start", "genes", "Layers: a newcomer gets", 1.0, 1.0, 3.0),
)

GROUPS = (("costs", "Costs"), ("world", "World"), ("body", "Body and movement"), ("senses", "Senses and hunting"),
          ("birth", "Birth and death"), ("genes", "Genes"), ("learning", "Learning"), ("auto", "Auto"), ("exam", "Exam"),
          ("runner", "Runner"))

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
