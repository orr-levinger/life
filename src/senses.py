"""What a creature senses, as the brain's 20 inputs, and a display label for what it did."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np

from .brain import N_SENSES
from .creature import Creature, similarity

if TYPE_CHECKING:
    from .food import Food

SIZE_CENTER = 1.25  # middle of the 0.5–2.0 size range
SIZE_HALF_RANGE = 0.75
FOOD_ANGLE = math.pi / 4  # a move within 45° of food counts as going to it
CALL_FIRST = 3  # a learner's choices 3.. are its calls (0 move, 1 eat, 2 attack)
N_CALLS = 10  # kinds of call (the most the calls gene allows)
KIN_INPUT = 44  # how alike the nearest creature it sees is
IN_WATER = 45  # it is in a pond
WATER_SEEN = 46  # the nearest obstacle in its field of vision is a pond
CALL_INPUTS = 27  # where the heard call starts: its kind (10), how loud (1), from where (2), the voice (4)


@dataclass(frozen=True)
class Senses:
    vector: np.ndarray
    food: Optional['Food']  # nearest food in range
    other: Optional[Creature]  # nearest creature it sees
    on_food: Optional['Food']  # food the creature overlaps
    heard: Optional[Creature] = None  # nearest creature it hears but does not see


def sense_all(world: Any, creatures: Optional[Sequence[Creature]] = None) -> List[Senses]:
    """What each of `creatures` (default: everyone in the world) senses. All distances are computed at once.
    Inputs 1-3: energy, size, on food. 4-7: nearest food it sees. 8-12: nearest stranger it sees (with its size). 13-16:
    nearest stranger it hears but does not see, of those that moved last step (where and how far only: a sound tells
    no size). 17-20: pain, for one step
    after a hit: hit, share taken, direction to the hitter."""
    creatures = list(world.creatures) if creatures is None else creatures
    if not creatures:
        return []
    me = _Eyes(creatures)
    foods, everyone = world.foods, world.creatures
    if foods:
        food_x, food_y = np.array([f.x for f in foods]), np.array([f.y for f in foods])
        food_dx, food_dy = food_x[None, :] - me.x[:, None], food_y[None, :] - me.y[:, None]
        food_d = np.hypot(food_dx, food_dy)
        food_edge = food_d - np.array([f.radius for f in foods])[None, :]  # distance to each food's edge
        food_seen = (food_d <= me.reach[:, None]) & me.cone(food_dx, food_dy, food_d)
        food_seen &= ~world.terrain.sight_blocked(me.x, me.y, food_x, food_y, only=food_seen)  # nothing seen through an obstacle
        nearest_food = np.argmin(np.where(food_seen, food_d, np.inf), axis=1)
        touching = np.argmin(food_edge, axis=1)
    ids = np.array([c.id for c in everyone])
    moved = np.array([c.moved for c in everyone], dtype=bool)
    other_x, other_y = np.array([c.x for c in everyone]), np.array([c.y for c in everyone])
    other_dx, other_dy = other_x[None, :] - me.x[:, None], other_y[None, :] - me.y[:, None]
    other_d = np.hypot(other_dx, other_dy)
    stranger = me.ids[:, None] != ids[None, :]  # anyone but itself: family is seen like anyone
    seen_from, heard_from = _noticed(everyone, world)
    seen = stranger & _in_sight(me, world, other_x, other_y, other_dx, other_dy, other_d, seen_from)
    heard = stranger & ~seen & moved[None, :] & (other_d <= me.hearing[:, None] * heard_from[None, :])  # only movers sound
    nearest_seen = np.argmin(np.where(seen, other_d, np.inf), axis=1)
    nearest_heard = np.argmin(np.where(heard, other_d, np.inf), axis=1)
    sheltered = world.terrain.under_tree(me.x, me.y)
    walls, water_ahead = _obstacles_seen(me, world)
    calls = _calls_heard(me, world)

    sensed = []
    for i, creature in enumerate(creatures):
        food = on_food = None
        if foods:
            food = foods[nearest_food[i]] if food_seen[i].any() else None
            on_food = foods[touching[i]] if food_edge[i, touching[i]] <= creature.radius else None
        other = everyone[nearest_seen[i]] if seen[i].any() else None
        sound = everyone[nearest_heard[i]] if heard[i].any() else None
        vector = _vector(creature, food, other, on_food, sound, walls[i], sheltered[i])
        vector[CALL_INPUTS:CALL_INPUTS + N_CALLS + 7] = calls[i]
        vector[IN_WATER] = 1.0 if world.terrain.in_pond(creature.x, creature.y) else 0.0
        vector[WATER_SEEN] = water_ahead[i]
        sensed.append(Senses(vector=vector, food=food, other=other, on_food=on_food, heard=sound))
    return sensed


def voice(caller: Creature, rules: Any) -> np.ndarray:
    """How a caller sounds: its diet, size, water and stealth genes, each scaled 0-1 by its range (alike genes sound
    alike; a gene it does not have yet says 0)."""
    low, high = rules["gene.size.low"], rules["gene.size.high"]
    size = (caller.size - low) / (high - low) if high > low else 0.0
    return np.array([caller.diet, size, getattr(caller, "water", 0.0), getattr(caller, "stealth", 0.0)])


def _calls_heard(me: '_Eyes', world: Any) -> np.ndarray:
    """[listener, 17]: the loudest call each one heard last step (never its own): its kind, how loud it arrived
    (1 at the caller, 0 where it stops carrying), where from (cos, sin) and the caller's voice; zeros when none."""
    out = np.zeros((len(me.x), N_CALLS + 7))
    calls = getattr(world, "calls", [])
    if not calls:
        return out
    cx, cy = np.array([c.x for c, _, _ in calls]), np.array([c.y for c, _, _ in calls])
    carries = np.array([volume for _, _, volume in calls]) * world.rules["call_reach"]
    dx, dy = cx[None, :] - me.x[:, None], cy[None, :] - me.y[:, None]
    distance = np.hypot(dx, dy)
    loud = np.where((distance <= carries[None, :]) & (me.ids[:, None] != np.array([c.id for c, _, _ in calls])[None, :]),
                    1.0 - distance / np.maximum(carries[None, :], 1e-9), -1.0)
    for i in np.nonzero(loud.max(axis=1) >= 0.0)[0]:
        j = int(np.argmax(loud[i]))
        caller, kind, _ = calls[j]
        out[i, kind] = 1.0
        out[i, N_CALLS] = loud[i, j]
        if distance[i, j] > 0:
            out[i, N_CALLS + 1:N_CALLS + 3] = dx[i, j] / distance[i, j], dy[i, j] / distance[i, j]
        out[i, N_CALLS + 3:] = voice(caller, world.rules)
    return out


class _Eyes:
    """The senses of a group of creatures as arrays, for deciding every pair at once."""

    def __init__(self, creatures: Sequence[Creature]) -> None:
        self.x = np.array([c.x for c in creatures])
        self.y = np.array([c.y for c in creatures])
        self.ids = np.array([c.id for c in creatures])
        self.reach = np.array([c.vision_reach for c in creatures])
        self.hearing = np.array([c.hearing for c in creatures])
        width = np.array([c.vision_width for c in creatures])
        self.all_around = width >= 360.0
        self.half = np.cos(np.radians(width / 2.0))
        self.fx = np.array([c.facing[0] for c in creatures])
        self.fy = np.array([c.facing[1] for c in creatures])

    def cone(self, dx: np.ndarray, dy: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """[creature, thing]: whether each thing lies inside each creature's vision cone (at distance 0: inside)."""
        cosine = (dx * self.fx[:, None] + dy * self.fy[:, None]) / np.maximum(distance, 1e-12)
        return self.all_around[:, None] | (cosine >= self.half[:, None]) | (distance == 0)


def sense(creature: Creature, world: Any) -> Senses:
    return sense_all(world, [creature])[0]


def perceived(watchers: Sequence[Creature], actors: Sequence[Creature], world: Any) -> np.ndarray:
    """[actor, watcher]: whether the watcher sees the actor or hears it move (family included)."""
    me = _Eyes(watchers)
    moved = np.array([c.moved for c in actors], dtype=bool)
    ax, ay = np.array([c.x for c in actors]), np.array([c.y for c in actors])
    seen_from, heard_from = _noticed(actors, world)
    dx, dy = ax[None, :] - me.x[:, None], ay[None, :] - me.y[:, None]
    distance = np.hypot(dx, dy)
    hears = (distance <= me.hearing[:, None] * heard_from[None, :]) & moved[None, :]  # sound goes around obstacles
    return (hears | _in_sight(me, world, ax, ay, dx, dy, distance, seen_from)).T


def _noticed(bodies: Sequence[Creature], world: Any) -> tuple:
    """How far each body is noticed, as a share of the looker's reach and the listener's hearing: its size to the seen
    and heard powers, × (1 - its stealth × the stealth strength)."""
    sizes = np.array([c.size for c in bodies])
    hidden = 1.0 - np.array([c.stealth for c in bodies]) * world.rules["stealth_strength"]
    return sizes ** world.rules["seen_power"] * hidden, sizes ** world.rules["heard_power"] * hidden


def _in_sight(me: '_Eyes', world: Any, x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray,
              distance: np.ndarray, seen_from: np.ndarray) -> np.ndarray:
    """[looker, creature]: within its vision reach (× the creature's size to the seen power) and cone, no obstacle between, and not hidden under a tree's crown
    (one under a crown is seen only from within the hide reach)."""
    sees = (distance <= me.reach[:, None] * seen_from[None, :]) & me.cone(dx, dy, distance)
    sees &= ~world.terrain.sight_blocked(me.x, me.y, x, y, only=sees)
    hidden = world.terrain.under_tree(x, y)[None, :] & (distance > world.rules["hide_reach"])
    return sees & ~hidden


def _obstacles_seen(me: '_Eyes', world: Any) -> np.ndarray:
    """Each looker's nearest obstacle in its field of vision as inputs (seen, distance to its edge ÷ vision reach, cos
    and sin of the direction to it), zeros when none is in view; and whether that obstacle is a pond."""
    out, wet = np.zeros((len(me.x), 4)), np.zeros(len(me.x))
    if not world.terrain.obstacles:
        return out, wet
    edge, dx, dy = world.terrain.nearest_edges(me.x, me.y)
    centre = np.hypot(dx, dy)
    visible = (edge <= me.reach[:, None]) & me.cone(dx, dy, centre)
    nearest = np.argmin(np.where(visible, edge, np.inf), axis=1)
    for i in np.nonzero(visible.any(axis=1))[0]:
        j = nearest[i]
        length = max(centre[i, j], 1e-12)
        out[i] = (1.0, edge[i, j] / me.reach[i], dx[i, j] / length, dy[i, j] / length)
        wet[i] = 1.0 if world.terrain.pond_circles[j] else 0.0
    return out, wet


def _sighting(creature: Creature, target: Any, reach: float) -> tuple:
    """A sighting or a sound as inputs: sensed, distance ÷ how far that sense reaches, cos and sin of its direction."""
    dx, dy = target.x - creature.x, target.y - creature.y
    angle = math.atan2(dy, dx)
    return (1.0, math.hypot(dx, dy) / reach, math.cos(angle), math.sin(angle))


def _relative_size(creature: Creature, other: Creature) -> float:
    return min(max(math.log2(other.size / creature.size), -1.0), 1.0)


def _vector(creature: Creature, food: Optional['Food'], other: Optional[Creature], on_food: Optional['Food'],
            heard: Optional[Creature], wall: np.ndarray, sheltered: bool) -> np.ndarray:
    vector = np.zeros(N_SENSES)
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
    vector[20:24] = wall  # the nearest obstacle in its field of vision
    vector[24] = 1.0 if sheltered else 0.0  # under a tree's crown
    vector[25] = 1.0 if food is not None and food.meat else 0.0  # the nearest food it sees is meat
    vector[26] = creature.diet  # what it lives on
    if other is not None:
        vector[KIN_INPUT] = similarity(creature, other, creature.rules)  # how alike the one it sees is
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
    other = senses.other  # any bigger one is run away from, any smaller one chased: no threshold, the brain decides
    if other is not None and other.size > creature.size and _toward(creature, other, dx, dy) < 0:
        return "RUN_AWAY"
    if other is not None and other.size < creature.size and _toward(creature, other, dx, dy) > 0:
        return "CHASE"
    if senses.food is not None and _toward(creature, senses.food, dx, dy) >= math.cos(FOOD_ANGLE):
        return "GO_TO_FOOD"
    return "WANDER"
