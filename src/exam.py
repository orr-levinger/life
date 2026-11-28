"""The intelligence exam: brains and random wanderers in two identical copies of the fixed test world.

Intelligence is energy earned ÷ energy spent (the owner, 2026-11-23: "intelligence should be rewards / waste").

The test world never changes, however hard the living world is, so scores compare across time, and nobody is
born or dies in it, so crowding cannot skew the comparison.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .brain import Brain
from .creature import Creature, full_tank
from .sphere import random_point
from .rules import TODAY_COSTS, Rules, starting_rules
from .settings import Settings
from .world import World



EXAM_BODIES = {"tank_power": 1.0, "living_power": 1.0, "speed_power": 1.0, "lifespan_power": 0.0, "heard_power": 0.0,
               "seen_power": 0.0}  # the exam's bodies grow in straight lines with size, as before size curves


def exam_settings(rules: Rules) -> Settings:
    """The exam's own world, from the rules: always the same unless the owner changes it on the Rules page."""
    return Settings(fruit_rate=0.0, max_fruit=0.0, max_food=rules["exam_max_food"], food_energy=(rules["exam_food_energy_low"], rules["exam_food_energy_high"]),
                    food_per_step=rules["exam_food_per_step"], lifespan=(4000.0, 6000.0), attacks=False)


def exam_costs(rules: Rules) -> Rules:
    """The exam world's costs: as they were before 2026-11-27, moving and living its own rules'."""
    return {**TODAY_COSTS, "move_cost": rules["exam_move_cost"], "step_cost": rules["exam_step_cost"]}


@dataclass(frozen=True)
class ExamResult:
    """Intelligence = energy earned ÷ energy spent in the test world. None when nothing was spent (nobody took it)."""
    intelligence: Optional[float]  # the examined creatures'
    wanderers: Optional[float]  # random wanderers', in the identical copy: the yardstick


def _exam_world(policy: str, brains: Sequence[Any], sizes: Sequence[float], seed: int, rules: Rules) -> World:
    """`brains` are Brains for policy "brain", NetWeights for "learner" (their learning is paused), None otherwise."""
    learner = None
    if policy == "learner":
        from .learner import Learner  # imported only when learner networks are examined: TensorFlow stays out otherwise

        learner = Learner(len(sizes), seed)
        for weights in brains:  # a fresh learner hands out slots 0, 1, 2 … in order
            learner.allocate(weights, lr=0.0)
    exam_rules = {**starting_rules(), "planet_radius": rules["exam_radius"],
                  **EXAM_BODIES, **exam_costs(rules)}  # its own small planet, bodies and costs as they were: the test stays
    world = World(exam_rules["planet_radius"], exam_settings(rules), policy,
                  random.Random(seed + 2), np.random.default_rng(seed), random.Random(seed + 1), evolving=False,
                  learner=learner, rules=exam_rules)
    world.prefill_food()
    spots = random.Random(seed)
    for slot, (brain, size) in enumerate(zip(brains, sizes)):
        spot = random_point(spots)
        full = full_tank(size, world.rules)
        world.add_creature(Creature(world.next_id(), spot, size, full * rules["exam_start_share"], policy,
                                    brain if policy == "brain" else None,
                                    slot=slot if policy == "learner" else None, rules=world.rules))
    return world


def exam_worlds(brains: Sequence[Any], sizes: List[float], seed: int, rules: Rules) -> Tuple[World, World]:
    """The brains (Brains or learner NetWeights) in one copy; random wanderers of the same sizes at the same spots in
    the other; the same food."""
    policy = "brain" if all(isinstance(brain, Brain) for brain in brains) else "learner"
    return (_exam_world(policy, brains, sizes, seed, rules), _exam_world("random", [None] * len(sizes), sizes, seed, rules))


def _efficiency(world: World, steps: int) -> Optional[float]:
    for _ in range(steps):
        world.step()
    return world.energy_earned / world.energy_spent if world.energy_spent > 0 else None


def run_exam(brains: Sequence[Any], sizes: List[float], seed: int, rules: Rules) -> ExamResult:
    """The exam, its world and length from `rules` (the living world's)."""
    steps = round(rules["exam_steps"])
    with_brains, with_walkers = exam_worlds(brains, sizes, seed, rules)
    return ExamResult(_efficiency(with_brains, steps), _efficiency(with_walkers, steps))

