"""What a creature senses, as the brain's 12 inputs, and a display label for what it did."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np

from .creature import Creature, is_family

if TYPE_CHECKING:
    from .food import Food

SIZE_CENTER = 1.25  # middle of the 0.5–2.0 size range
SIZE_HALF_RANGE = 0.75
THREAT_RATIO = 1.5  # a creature this many times bigger counts as a threat
FOOD_ANGLE = math.pi / 4  # a move within 45° of food counts as going to it


@dataclass(frozen=True)
class Senses:
    vector: np.ndarray
    food: Optional['Food']  # nearest food in range
    other: Optional[Creature]  # nearest non-family creature in range
    on_food: Optional['Food']  # food the creature overlaps


def sense_all(world: Any, creatures: Optional[Sequence[Creature]] = None) -> List[Senses]:
    """What each of `creatures` (default: everyone in the world) senses. All distances are computed at once.
    Inputs 1–3: energy, size, on food. 4–7: nearest food in range. 8–12: nearest non-family creature in range."""
    creatures = list(world.creatures) if creatures is None else creatures
    if not creatures:
        return []
    cx = np.array([c.x for c in creatures])
    cy = np.array([c.y for c in creatures])
    foods, everyone = world.foods, world.creatures
    if foods:
        food_d = np.hypot(np.array([f.x for f in foods])[None, :] - cx[:, None],
                          np.array([f.y for f in foods])[None, :] - cy[:, None])
        food_edge = food_d - np.array([f.radius for f in foods])[None, :]  # distance to each food's edge
    other_d = np.hypot(np.array([c.x for c in everyone])[None, :] - cx[:, None],
                       np.array([c.y for c in everyone])[None, :] - cy[:, None])

    sensed = []
    for i, creature in enumerate(creatures):
        food = on_food = other = None
        if foods:
            nearest = int(np.argmin(food_d[i]))
            if food_d[i, nearest] <= Creature.SENSE_RANGE:
                food = foods[nearest]
            touching = int(np.argmin(food_edge[i]))
            if food_edge[i, touching] <= creature.radius:
                on_food = foods[touching]
        for j in np.argsort(other_d[i], kind="stable"):
            if other_d[i, j] > Creature.SENSE_RANGE:
                break
            if everyone[j] is not creature and not is_family(creature, everyone[j]):
                other = everyone[j]
                break
        sensed.append(Senses(vector=_vector(creature, food, other, on_food), food=food, other=other, on_food=on_food))
    return sensed


def sense(creature: Creature, world: Any) -> Senses:
    return sense_all(world, [creature])[0]


def _seen(creature: Creature, target: Any) -> tuple:
    """A sighting as inputs: seen, distance ÷ sense range, cos and sin of its direction."""
    dx, dy = target.x - creature.x, target.y - creature.y
    angle = math.atan2(dy, dx)
    return (1.0, math.hypot(dx, dy) / Creature.SENSE_RANGE, math.cos(angle), math.sin(angle))


def _vector(creature: Creature, food: Optional['Food'], other: Optional[Creature],
            on_food: Optional['Food']) -> np.ndarray:
    vector = np.zeros(12)
    vector[0] = creature.energy / creature.max_energy
    vector[1] = (creature.size - SIZE_CENTER) / SIZE_HALF_RANGE
    vector[2] = 1.0 if on_food is not None else 0.0
    if food is not None:
        vector[3:7] = _seen(creature, food)
    if other is not None:
        relative_size = min(max(math.log2(other.size / creature.size), -1.0), 1.0)
        vector[7:12] = _seen(creature, other) + (relative_size,)
    return vector


def _toward(creature: Creature, target: Any, dx: float, dy: float) -> float:
    """Cosine between the move (dx, dy) and the direction to the target."""
    tx, ty = target.x - creature.x, target.y - creature.y
    norm = math.hypot(dx, dy) * math.hypot(tx, ty)
    return (dx * tx + dy * ty) / norm if norm > 0 else 0.0


def display_intent(creature: Creature, action: tuple[str, Any], senses: Senses) -> str:
    """A label for the visualizer's arrow, derived from what the creature did. Display only."""
    kind, param = action
    if kind == "ATTACK":
        return "ATTACK"
    if kind != "MOVE" or math.hypot(*param) == 0:
        return "REST"
    dx, dy = param
    if senses.food is not None and _toward(creature, senses.food, dx, dy) >= math.cos(FOOD_ANGLE):
        return "GO_TO_FOOD"
    other = senses.other
    if other is not None and other.size >= THREAT_RATIO * creature.size and _toward(creature, other, dx, dy) < 0:
        return "RUN_AWAY"
    return "WANDER"
