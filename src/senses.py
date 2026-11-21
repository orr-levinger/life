"""What a creature senses, as the brain's 12 inputs, and a display label for what it did."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .creature import Creature, is_family
from .sensors import ProximitySensor

if TYPE_CHECKING:
    from .food import Food

SIZE_CENTER = 1.25  # middle of the 0.5–2.0 size range
SIZE_HALF_RANGE = 0.75
THREAT_RATIO = 1.5  # a creature this many times bigger counts as a threat
FOOD_ANGLE = math.pi / 4  # a move within 45° of food counts as going to it

_SENSOR = ProximitySensor(sense_range=Creature.SENSE_RANGE)


@dataclass(frozen=True)
class Senses:
    vector: np.ndarray
    food: Optional['Food']  # nearest food in range
    other: Optional[Creature]  # nearest non-family creature in range
    on_food: Optional['Food']  # food the creature overlaps


def sense(creature: Creature, world: Any) -> Senses:
    """Inputs 1–3: energy, size, on food. 4–7: nearest food. 8–12: nearest non-family creature."""
    sightings = _SENSOR.get_reading(creature, world)
    nearest_food = next((s for s in sightings if s[0] == "food"), None)
    nearest_other = next(
        (s for s in sightings if s[0] == "creature" and not is_family(creature, s[1])), None
    )
    on_food = next((f for f in world.foods if f.is_overlapping(creature.x, creature.y, creature.radius)), None)

    vector = np.zeros(12)
    vector[0] = creature.energy / creature.max_energy
    vector[1] = (creature.size - SIZE_CENTER) / SIZE_HALF_RANGE
    vector[2] = 1.0 if on_food is not None else 0.0
    if nearest_food is not None:
        _, _, distance, angle = nearest_food
        vector[3:7] = (1.0, distance / Creature.SENSE_RANGE, math.cos(angle), math.sin(angle))
    if nearest_other is not None:
        _, other, distance, angle = nearest_other
        relative_size = float(np.clip(math.log2(other.size / creature.size), -1.0, 1.0))
        vector[7:12] = (1.0, distance / Creature.SENSE_RANGE, math.cos(angle), math.sin(angle), relative_size)

    return Senses(
        vector=vector,
        food=nearest_food[1] if nearest_food is not None else None,
        other=nearest_other[1] if nearest_other is not None else None,
        on_food=on_food,
    )


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
