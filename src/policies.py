"""How a creature picks its action: its brain, its learning network, a random walk, or the hand-written rules.

Each step a creature chooses one action: move, eat, or attack. Only brains and learning networks are creatures of the
simulation; the random walk and the rules are yardsticks.
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np

from .creature import is_family
from .senses import THREAT_RATIO, Senses
from .sensors import ProximitySensor

if TYPE_CHECKING:
    from .creature import Creature
    from .food import Food
    from .world import World

Action = Tuple[str, Any]  # ("MOVE", (dx, dy)) | ("EAT", Food | None) | ("ATTACK", Creature)
STAY: Action = ("MOVE", (0.0, 0.0))


def _reachable_food(creature: Creature, senses: Senses) -> Optional[Food]:
    """What an eat action bites: the food the creature stands on, else the nearest food within reach, else nothing."""
    if senses.on_food is not None:
        return senses.on_food
    if senses.food is not None and creature.can_reach(senses.food):
        return senses.food
    return None


def _capped(dx: float, dy: float, top_speed: float) -> Tuple[float, float]:
    length = math.hypot(dx, dy)
    if length > top_speed:
        return dx * top_speed / length, dy * top_speed / length
    return dx, dy


def brain_action(creature: Creature, senses: Senses, attacks_on: bool) -> Action:
    """Outputs 1-2 move, 3 eat, 4 attack (only when hunting exists at this stage)."""
    out = creature.brain.forward(senses.vector)
    if attacks_on and out[3] > 0 and out[3] >= out[2] and senses.other is not None:
        return ("ATTACK", senses.other)
    if out[2] > 0:
        return ("EAT", _reachable_food(creature, senses))
    return ("MOVE", _capped(out[0] * creature.velocity, out[1] * creature.velocity, creature.velocity))


def learner_action(creature: Creature, senses: Senses, choice: int, move: Tuple[float, float]) -> Action:
    """A learner's choice (0 move, 1 eat, 2 attack) and move as an action; an attack with nobody in view does nothing."""
    if choice == 1:
        return ("EAT", _reachable_food(creature, senses))
    if choice == 2:
        return ("ATTACK", senses.other) if senses.other is not None else STAY
    return ("MOVE", _capped(move[0] * creature.velocity, move[1] * creature.velocity, creature.velocity))


def random_action(creature: Creature, senses: Senses, rng: random.Random) -> Action:
    """The yardstick wanderer: eat whenever food is within reach; otherwise a random direction at a random speed."""
    food = _reachable_food(creature, senses)
    if food is not None:
        return ("EAT", food)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    speed = rng.uniform(0.0, creature.velocity)
    return ("MOVE", (math.cos(angle) * speed, math.sin(angle) * speed))


_RULES_SENSOR = ProximitySensor()


def rules_action(creature: Creature, world: World, rng: random.Random) -> Action:
    """The original hand-written behavior, kept as a yardstick. Never used by a brain or a learner."""
    _RULES_SENSOR.sense_range = creature.SENSE_RANGE
    sightings = _RULES_SENSOR.get_reading(creature, world)
    others = [
        (obj, dist, angle) for kind, obj, dist, angle in sightings
        if kind == "creature" and not is_family(creature, obj)
        and not (1 / THREAT_RATIO < obj.size / creature.size < THREAT_RATIO)
    ]
    foods = [(obj, dist, angle) for kind, obj, dist, angle in sightings if kind == "food"]
    v = creature.velocity

    if others:
        other, dist, angle = others[0]
        if other.size >= creature.size * THREAT_RATIO:
            return ("MOVE", (math.cos(angle + math.pi) * v, math.sin(angle + math.pi) * v))
        attack_range = (creature.radius + other.radius) * creature.ATTACK_RANGE_FACTOR
        if creature.size >= other.size * THREAT_RATIO and dist <= attack_range:
            return ("ATTACK", other)
        return ("MOVE", (math.cos(angle) * v, math.sin(angle) * v))

    if foods:
        food, _, angle = foods[0]
        if creature.can_reach(food):
            return ("EAT", food)
        return ("MOVE", (math.cos(angle) * v * 0.75, math.sin(angle) * v * 0.75))

    if rng.random() < 0.5:
        angle = rng.random() * 2 * math.pi
        return ("MOVE", (math.cos(angle) * v * 0.5, math.sin(angle) * v * 0.5))
    return STAY


def decide(creature: Creature, world: World, senses: Senses, rng: random.Random) -> Action:
    if creature.policy == "brain":
        return brain_action(creature, senses, world.stage.attacks)
    if creature.policy == "random":
        return random_action(creature, senses, rng)
    if creature.policy == "rules":
        return rules_action(creature, world, rng)
    raise ValueError(f"{creature.policy!r} creatures decide together, through decide_all")


def decide_all(world: World, creatures: Sequence[Creature], senses: Sequence[Senses]) -> List[Action]:
    """Every creature's action: the learners in one call to the world's learner, the others one by one."""
    actions: List[Optional[Action]] = [None] * len(creatures)
    learners = [i for i, creature in enumerate(creatures) if creature.policy == "learner"]
    if learners:
        outputs = world.learner.act([creatures[i].slot for i in learners],
                                    np.stack([senses[i].vector for i in learners]),
                                    world.stage.attacks, learning=world.evolving)
        for i, (choice, move) in zip(learners, outputs):
            actions[i] = learner_action(creatures[i], senses[i], choice, move)
    for i, creature in enumerate(creatures):
        if actions[i] is None:
            actions[i] = decide(creature, world, senses[i], world.rng)
    return actions
