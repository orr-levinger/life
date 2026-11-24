"""What a creature senses, as the brain's 20 inputs, and a display label for what it did."""
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
FOOD_ANGLE = math.pi / 4  # a move within 45° of food counts as going to it


@dataclass(frozen=True)
class Senses:
    vector: np.ndarray
    food: Optional['Food']  # nearest food in range
    other: Optional[Creature]  # nearest non-family creature it sees
    on_food: Optional['Food']  # food the creature overlaps
    heard: Optional[Creature] = None  # nearest non-family creature it hears but does not see


def in_cone(creature: Creature, dx, dy, distance):
    """Whether what lies (dx, dy) away, at `distance`, is inside the creature's vision cone (arrays work too)."""
    if creature.vision_width >= 360.0:
        return np.ones(np.shape(distance), dtype=bool) if np.ndim(distance) else True
    fx, fy = creature.facing
    cosine = (np.asarray(dx) * fx + np.asarray(dy) * fy) / np.maximum(distance, 1e-12)
    inside = (cosine >= math.cos(math.radians(creature.vision_width / 2.0))) | (np.asarray(distance) == 0)
    return inside if np.ndim(distance) else bool(inside)


def sense_all(world: Any, creatures: Optional[Sequence[Creature]] = None) -> List[Senses]:
    """What each of `creatures` (default: everyone in the world) senses. All distances are computed at once.
    Inputs 1-3: energy, size, on food. 4-7: nearest food it sees. 8-12: nearest stranger it sees (with its size). 13-16:
    nearest stranger it hears but does not see (where and how far only: a sound tells no size). 17-20: pain, for one step
    after a hit: hit, share taken, direction to the hitter."""
    creatures = list(world.creatures) if creatures is None else creatures
    if not creatures:
        return []
    cx = np.array([c.x for c in creatures])
    cy = np.array([c.y for c in creatures])
    foods, everyone = world.foods, world.creatures
    if foods:
        food_x, food_y = np.array([f.x for f in foods]), np.array([f.y for f in foods])
        food_dx, food_dy = food_x[None, :] - cx[:, None], food_y[None, :] - cy[:, None]
        food_d = np.hypot(food_dx, food_dy)
        food_edge = food_d - np.array([f.radius for f in foods])[None, :]  # distance to each food's edge
    other_x, other_y = np.array([c.x for c in everyone]), np.array([c.y for c in everyone])
    other_d = np.hypot(other_x[None, :] - cx[:, None], other_y[None, :] - cy[:, None])

    sensed = []
    for i, creature in enumerate(creatures):
        food = on_food = other = heard = None
        if foods:
            seen = (food_d[i] <= creature.vision_reach) & in_cone(creature, food_dx[i], food_dy[i], food_d[i])
            if seen.any():
                food = foods[int(np.argmin(np.where(seen, food_d[i], np.inf)))]
            touching = int(np.argmin(food_edge[i]))
            if food_edge[i, touching] <= creature.radius:
                on_food = foods[touching]
        farthest = max(creature.vision_reach, creature.hearing)
        for j in np.argsort(other_d[i], kind="stable"):
            distance = other_d[i, j]
            if distance > farthest or (other is not None and heard is not None):
                break
            them = everyone[j]
            if them is creature or is_family(creature, them):
                continue
            sees = distance <= creature.vision_reach and in_cone(creature, them.x - creature.x, them.y - creature.y,
                                                                  distance)
            if sees and other is None:
                other = them
            elif not sees and distance <= creature.hearing and heard is None:
                heard = them
        sensed.append(Senses(vector=_vector(creature, food, other, on_food, heard), food=food, other=other,
                             on_food=on_food, heard=heard))
    return sensed


def sense(creature: Creature, world: Any) -> Senses:
    return sense_all(world, [creature])[0]


def perceived(watchers: Sequence[Creature], actors: Sequence[Creature]) -> np.ndarray:
    """[actor, watcher]: whether the watcher sees or hears the actor (family included)."""
    ax, ay = np.array([c.x for c in actors]), np.array([c.y for c in actors])
    out = np.zeros((len(actors), len(watchers)), dtype=bool)
    for j, watcher in enumerate(watchers):
        dx, dy = ax - watcher.x, ay - watcher.y
        distance = np.hypot(dx, dy)
        out[:, j] = (distance <= watcher.hearing) | ((distance <= watcher.vision_reach) &
                                                     in_cone(watcher, dx, dy, distance))
    return out


def _sighting(creature: Creature, target: Any, reach: float) -> tuple:
    """A sighting or a sound as inputs: sensed, distance ÷ how far that sense reaches, cos and sin of its direction."""
    dx, dy = target.x - creature.x, target.y - creature.y
    angle = math.atan2(dy, dx)
    return (1.0, math.hypot(dx, dy) / reach, math.cos(angle), math.sin(angle))


def _relative_size(creature: Creature, other: Creature) -> float:
    return min(max(math.log2(other.size / creature.size), -1.0), 1.0)


def _vector(creature: Creature, food: Optional['Food'], other: Optional[Creature],
            on_food: Optional['Food'], heard: Optional[Creature]) -> np.ndarray:
    vector = np.zeros(20)
    vector[0] = creature.energy / creature.max_energy
    vector[1] = (creature.size - SIZE_CENTER) / SIZE_HALF_RANGE
    vector[2] = 1.0 if on_food is not None else 0.0
    if food is not None:
        vector[3:7] = _sighting(creature, food, creature.vision_reach)
    if other is not None:
        vector[7:12] = _sighting(creature, other, creature.vision_reach) + (_relative_size(creature, other),)
    if heard is not None:
        vector[12:16] = _sighting(creature, heard, creature.hearing)
    if creature.hurt is not None:
        vector[16:20] = (1.0,) + creature.hurt
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
    other = senses.other
    bigger = creature.rules["threat_ratio"]  # at least this many times its size
    if other is not None and other.size >= bigger * creature.size and _toward(creature, other, dx, dy) < 0:
        return "RUN_AWAY"
    if other is not None and other.size * bigger <= creature.size and _toward(creature, other, dx, dy) > 0:
        return "CHASE"
    if senses.food is not None and _toward(creature, senses.food, dx, dy) >= math.cos(FOOD_ANGLE):
        return "GO_TO_FOOD"
    return "WANDER"
