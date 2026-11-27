"""What a creature senses, as the brain's 20 inputs, and a display label for what it did."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np

from .brain import N_SENSES
from .creature import SEEN_GENES, Creature, genes_seen, genes_table, kin_genes, similarity  # noqa: F401 (genes_seen: the tests')
from .sphere import arc, bearing, neighbours

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
GENES_AT = 47  # the nearest one it sees: its 18 genes (SEEN_GENES), 0-1 each
HITTER_AT = GENES_AT + len(SEEN_GENES)  # the one that hit it, for one step: its distance ÷ vision reach, then its genes
EYE_PIXELS, EYE_CHANNELS = 16, 6  # the eye: a row of pixels from its left edge of vision to its right, each: a creature's
# closeness, its size against its own, how much of a hunter, how alike; food's closeness; a rock's or water's closeness
EYE_AT = HITTER_AT + 1 + len(SEEN_GENES)
EAR_DIRECTIONS = 8  # the ears: ahead, ahead-left, left, behind-left, behind, behind-right, right, ahead-right; each:
# how loud the movers are that way, how loud the calls are
EARS_AT = EYE_AT + EYE_PIXELS * EYE_CHANNELS
EVERY_PAIR = False  # True: every pair measured, the slow way (the tests' yardstick for the k-d tree)


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
    after a hit: hit, share taken, direction to the hitter. 48-65: the nearest one it sees: its genes. 66-84: the one
    that hit it, for one step: its distance and genes. Every direction is in its own view: (ahead, left); every
    distance is along the ground."""
    creatures = list(world.creatures) if creatures is None else creatures
    if not creatures:
        return []
    me = _Eyes(creatures, world)
    foods, everyone = world.foods, world.creatures
    if foods:
        food_p = np.stack([f.p for f in foods])
        radii = np.array([f.radius for f in foods])
        food_d = me.distance(food_p, np.maximum(me.reach, me.radius + radii.max()))  # what it sees or stands on
        food_edge = food_d - radii[None, :]  # distance to each food's edge
        food_seen = (food_d <= me.reach[:, None]) & me.cone(food_p, food_d)
        food_seen &= ~world.terrain.sight_blocked(me.p, food_p, only=food_seen)  # nothing seen through a rock
        nearest_food = np.argmin(np.where(food_seen, food_d, np.inf), axis=1)
        touching = np.argmin(food_edge, axis=1)
    ids = np.array([c.id for c in everyone])
    moved = np.array([c.moved for c in everyone], dtype=bool)
    other_p = np.stack([c.p for c in everyone])
    seen_from, heard_from = _noticed(everyone, world)
    other_d = me.distance(other_p, me.noticing(seen_from, heard_from))
    stranger = me.ids[:, None] != ids[None, :]  # anyone but itself: family is seen like anyone
    seen = stranger & _in_sight(me, world, other_p, other_d, seen_from)
    heard = stranger & ~seen & moved[None, :] & (other_d <= me.hearing[:, None] * heard_from[None, :])  # only movers sound
    nearest_seen = np.argmin(np.where(seen, other_d, np.inf), axis=1)
    nearest_heard = np.argmin(np.where(heard, other_d, np.inf), axis=1)
    sheltered = world.terrain.under_tree(me.p)
    walls, water_ahead = _obstacles_seen(me, world)
    calls = _calls_heard(me, world)
    eyes = _eye_pixels(me, world, creatures, everyone, seen, other_d,
                       (food_seen, food_d, food_p) if foods else None)
    ears = _ears(me, world, everyone, stranger, moved, other_d, other_p, heard_from)
    shown = np.nonzero(seen.any(axis=1))[0]  # the genes of every one that is someone's nearest seen, worked out once
    seen_ones = sorted(set(nearest_seen[shown].tolist()))
    genes = dict(zip(seen_ones, genes_table([everyone[j] for j in seen_ones], world)))

    sensed = []
    for i, creature in enumerate(creatures):
        food = on_food = None
        if foods:
            food = foods[nearest_food[i]] if food_seen[i].any() else None
            on_food = foods[touching[i]] if food_edge[i, touching[i]] <= creature.radius else None
        other = everyone[nearest_seen[i]] if seen[i].any() else None
        sound = everyone[nearest_heard[i]] if heard[i].any() else None
        vector = _vector(creature, food, other, on_food, sound, walls[i], sheltered[i], me.R)
        vector[CALL_INPUTS:CALL_INPUTS + N_CALLS + 7] = calls[i]
        vector[IN_WATER] = 1.0 if world.terrain.in_pond(creature.p) else 0.0
        vector[WATER_SEEN] = water_ahead[i]
        if other is not None:
            vector[GENES_AT:GENES_AT + len(SEEN_GENES)] = genes[nearest_seen[i]]
        vector[EYE_AT:EARS_AT] = eyes[i].ravel()
        vector[EARS_AT:EARS_AT + EAR_DIRECTIONS * 2] = ears[i].ravel()
        if creature.hurt is not None and len(creature.hurt) > 3:  # who hit it, and how far (a save of before: unknown)
            vector[HITTER_AT] = creature.hurt[3] / max(creature.vision_reach, 1e-9)
            vector[HITTER_AT + 1:HITTER_AT + 1 + len(SEEN_GENES)] = creature.hurt[4:]
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
    (1 at the caller, 0 where it stops carrying), where from (ahead, left) and the caller's voice; zeros when none."""
    out = np.zeros((len(me.p), N_CALLS + 7))
    calls = getattr(world, "calls", [])
    if not calls:
        return out
    where = np.stack([np.asarray(c.p, dtype=float) for c, _, _ in calls])
    carries = np.array([volume for _, _, volume in calls]) * world.rules["call_reach"]
    distance = me.distance(where, np.full(len(me.p), carries.max()))
    loud = np.where((distance <= carries[None, :]) & (me.ids[:, None] != np.array([c.id for c, _, _ in calls])[None, :]),
                    1.0 - distance / np.maximum(carries[None, :], 1e-9), -1.0)
    for i in np.nonzero(loud.max(axis=1) >= 0.0)[0]:
        j = int(np.argmax(loud[i]))
        caller, kind, _ = calls[j]
        out[i, kind] = 1.0
        out[i, N_CALLS] = loud[i, j]
        out[i, N_CALLS + 1:N_CALLS + 3] = bearing(me.p[i], me.f[i], where[j])
        out[i, N_CALLS + 3:] = voice(caller, world.rules)
    return out


def _angles(me: '_Eyes', rows: np.ndarray, points: np.ndarray) -> np.ndarray:
    """How far left of its facing each point is, in its own view (radians, -pi to pi), for lookers `rows`."""
    ahead, left = bearing(me.p[rows], me.f[rows], points)
    return np.arctan2(left, ahead)


def _nearest_per_slice(rows: np.ndarray, slices: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """The positions of the nearest item of each (looker, slice) among these pairs."""
    if not len(rows):
        return np.zeros(0, dtype=int)
    order = np.lexsort((distance, slices, rows))
    first = np.ones(len(order), dtype=bool)
    first[1:] = (rows[order][1:] != rows[order][:-1]) | (slices[order][1:] != slices[order][:-1])
    return order[first]


def _eye_pixels(me: '_Eyes', world: Any, creatures: Sequence[Creature], everyone: Sequence[Creature],
                seen: np.ndarray, other_d: np.ndarray, food: Optional[tuple]) -> np.ndarray:
    """[looker, pixel, channel]: what each looker's eye shows (see EYE_PIXELS). Its field of vision is cut into its
    `eye_pixels` slices; each slice shows the nearest of each kind of thing in it and fills its share of the 16 pixels."""
    n = len(creatures)
    sharp = np.clip(np.array([c.eye_pixels for c in creatures], dtype=int), 1, EYE_PIXELS)
    half = np.where(me.all_around, math.pi, np.arccos(np.clip(me.half, -1.0, 1.0)))
    slices = np.zeros((n, EYE_PIXELS, EYE_CHANNELS))

    def slice_of(rows: np.ndarray, angle: np.ndarray) -> np.ndarray:
        share = np.clip((half[rows] - angle) / (2.0 * half[rows]), 0.0, 1.0 - 1e-9)  # 0 at its left edge, 1 at its right
        return np.floor(share * sharp[rows]).astype(int)

    def closeness(distance: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - distance / np.maximum(me.reach[rows], 1e-9), 0.0, 1.0)

    rows, cols = np.nonzero(seen)  # creatures: the nearest of each slice, with its size, diet and likeness
    if len(rows):
        where = np.stack([everyone[j].p for j in cols])
        cut = slice_of(rows, _angles(me, rows, where))
        pick = _nearest_per_slice(rows, cut, other_d[rows, cols])
        r, s, j = rows[pick], cut[pick], cols[pick]
        sizes = np.array([c.size for c in everyone])
        mine = np.array([c.size for c in creatures])
        slices[r, s, 0] = closeness(other_d[r, j], r)
        slices[r, s, 1] = (np.clip(np.log2(sizes[j] / mine[r]), -1.0, 1.0) + 1.0) / 2.0
        slices[r, s, 2] = np.array([c.diet for c in everyone])[j]
        slices[r, s, 3] = _alike(creatures, everyone, r, j, world.rules)
    if food is not None:  # food: the nearest of each slice
        food_seen, food_d, food_p = food
        rows, cols = np.nonzero(food_seen)
        if len(rows):
            cut = slice_of(rows, _angles(me, rows, food_p[cols]))
            pick = _nearest_per_slice(rows, cut, food_d[rows, cols])
            slices[rows[pick], cut[pick], 4] = closeness(food_d[rows[pick], cols[pick]], rows[pick])
    near = np.full((n, EYE_PIXELS), np.inf)  # rocks and water: the nearest edge in each slice
    if world.terrain.obstacles:
        edge, centres = world.terrain.nearest_edges(me.p, me.reach)
        rows, cols = np.nonzero(edge <= me.reach[:, None])
        if len(rows):
            cut = slice_of(rows, _angles(me, rows, centres[cols]))
            inside = edge[rows, cols] == 0.0
            angle_ok = me.all_around[rows] | inside | (np.abs(_angles(me, rows, centres[cols])) <= half[rows])
            rows, cols, cut = rows[angle_ok], cols[angle_ok], cut[angle_ok]
            np.minimum.at(near, (rows, cut), edge[rows, cols])
    if world.terrain.water is not None:  # one sight line through the middle of each real slice
        k = np.arange(EYE_PIXELS)[None, :]
        middles = half[:, None] - (k + 0.5) / sharp[:, None] * 2.0 * half[:, None]
        angles = np.where(k < sharp[:, None], middles, np.nan)
        near = np.minimum(near, world.terrain.water_rays(me.p, me.f, me.reach, angles))
    slices[:, :, 5] = np.where(np.isfinite(near), np.clip(1.0 - near / np.maximum(me.reach, 1e-9)[:, None], 0.0, 1.0), 0.0)
    pixel = (np.arange(EYE_PIXELS)[None, :] * sharp[:, None]) // EYE_PIXELS  # each pixel's slice
    return np.take_along_axis(slices, pixel[:, :, None], axis=1)


def _alike(creatures: Sequence[Creature], everyone: Sequence[Creature], rows: np.ndarray, cols: np.ndarray,
           rules: Any) -> np.ndarray:
    """How alike each looker (rows) and creature (cols) are (as `similarity`), for these pairs at once."""
    genes = kin_genes(rules)
    if not genes:
        return np.ones(len(rows))
    span = np.array([rules[f"gene.{g}.high"] - rules[f"gene.{g}.low"] for g in genes])
    a = np.array([[getattr(c, g) for g in genes] for c in creatures])[rows]
    b = np.array([[getattr(c, g) for g in genes] for c in everyone])[cols]
    return 1.0 - np.mean(np.abs(a - b) / span, axis=1)


def _ears(me: '_Eyes', world: Any, everyone: Sequence[Creature], stranger: np.ndarray, moved: np.ndarray,
          other_d: np.ndarray, other_p: np.ndarray, heard_from: np.ndarray) -> np.ndarray:
    """[listener, direction, 2]: how loud the movers and the calls are from each of its 8 directions (the loudest:
    1 at it, 0 where it stops being heard)."""
    out = np.zeros((len(me.p), EAR_DIRECTIONS, 2))

    def direction(rows: np.ndarray, points: np.ndarray) -> np.ndarray:
        return np.round(_angles(me, rows, points) / (2.0 * math.pi / EAR_DIRECTIONS)).astype(int) % EAR_DIRECTIONS

    carry = me.hearing[:, None] * heard_from[None, :]
    rows, cols = np.nonzero(stranger & moved[None, :] & (other_d <= carry) & (carry > 0))
    if len(rows):
        loud = 1.0 - other_d[rows, cols] / carry[rows, cols]
        np.maximum.at(out[:, :, 0], (rows, direction(rows, other_p[cols])), loud)
    calls = getattr(world, "calls", [])
    if calls:
        where = np.stack([np.asarray(c.p, dtype=float) for c, _, _ in calls])
        carries = np.array([volume for _, _, volume in calls]) * world.rules["call_reach"]
        distance = me.distance(where, np.full(len(me.p), carries.max()))
        own = me.ids[:, None] == np.array([c.id for c, _, _ in calls])[None, :]
        rows, cols = np.nonzero((distance <= carries[None, :]) & ~own & (carries[None, :] > 0))
        if len(rows):
            loud = 1.0 - distance[rows, cols] / carries[cols]
            np.maximum.at(out[:, :, 1], (rows, direction(rows, where[cols])), loud)
    return out


class _Eyes:
    """The senses of a group of creatures as arrays, for deciding every pair at once."""

    def __init__(self, creatures: Sequence[Creature], world: Any) -> None:
        self.R = world.rules["planet_radius"]
        self.p = np.stack([c.p for c in creatures])
        self.f = np.stack([np.asarray(c.facing, dtype=float) for c in creatures])
        self.ids = np.array([c.id for c in creatures])
        self.reach = np.array([c.vision_reach for c in creatures])
        self.radius = np.array([c.radius for c in creatures])
        self.hearing = np.array([c.hearing for c in creatures])
        width = np.array([c.vision_width for c in creatures])
        self.all_around = width >= 360.0
        self.half = np.cos(np.radians(width / 2.0))
    def distance(self, points: np.ndarray, reach: Optional[np.ndarray] = None) -> np.ndarray:
        """[creature, point]: how far along the ground. With `reach` (each creature's farthest sense), a k-d tree finds
        the points within it and only those are measured; the rest are inf (too far for anything it does)."""
        if reach is None or EVERY_PAIR:
            return arc(self.p[:, None, :], points[None, :, :], self.R)
        i, j, d = neighbours(self.p, points, reach, self.R)
        out = np.full((len(self.p), len(points)), np.inf)
        out[i, j] = d
        return out

    def noticing(self, seen_from: np.ndarray, heard_from: np.ndarray) -> np.ndarray:
        """Each creature's farthest sense of a body: its vision reach or its hearing, × the most noticeable body's."""
        return np.maximum(self.reach * seen_from.max(initial=0.0), self.hearing * heard_from.max(initial=0.0))

    def toward(self, points: np.ndarray) -> tuple:
        """[creature, point]: where each point is in each creature's own view, (ahead, left)."""
        return bearing(self.p[:, None, :], self.f[:, None, :], points[None, :, :])

    def cone(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """[creature, point]: whether each point lies inside each creature's vision cone (at distance 0: inside); only
        the measured points (a finite distance) are looked at, the others are never in sight anyway."""
        out = np.broadcast_to(self.all_around[:, None] | (distance == 0), distance.shape).copy()
        i, j = np.nonzero(np.isfinite(distance) & ~out)
        if len(i):
            ahead, _ = bearing(self.p[i], self.f[i], points[j])
            out[i, j] = ahead >= self.half[i]
        return out


def sense(creature: Creature, world: Any) -> Senses:
    return sense_all(world, [creature])[0]


def perceived(watchers: Sequence[Creature], actors: Sequence[Creature], world: Any) -> np.ndarray:
    """[actor, watcher]: whether the watcher sees the actor or hears it move (family included)."""
    me = _Eyes(watchers, world)
    moved = np.array([c.moved for c in actors], dtype=bool)
    where = np.stack([c.p for c in actors])
    seen_from, heard_from = _noticed(actors, world)
    distance = me.distance(where, me.noticing(seen_from, heard_from))
    hears = (distance <= me.hearing[:, None] * heard_from[None, :]) & moved[None, :]  # sound goes around obstacles
    return (hears | _in_sight(me, world, where, distance, seen_from)).T


def _noticed(bodies: Sequence[Creature], world: Any) -> tuple:
    """How far each body is noticed, as a share of the looker's reach and the listener's hearing: its size to the seen
    and heard powers, × (1 - its stealth × the stealth strength)."""
    sizes = np.array([c.size for c in bodies])
    hidden = 1.0 - np.array([c.stealth for c in bodies]) * world.rules["stealth_strength"]
    return sizes ** world.rules["seen_power"] * hidden, sizes ** world.rules["heard_power"] * hidden


def _in_sight(me: '_Eyes', world: Any, where: np.ndarray, distance: np.ndarray, seen_from: np.ndarray) -> np.ndarray:
    """[looker, creature]: within its vision reach (× the creature's size to the seen power) and cone, no obstacle between, and not hidden under a tree's crown
    (one under a crown is seen only from within the hide reach)."""
    sees = (distance <= me.reach[:, None] * seen_from[None, :]) & me.cone(where, distance)
    sees &= ~world.terrain.sight_blocked(me.p, where, only=sees)
    hidden = world.terrain.under_tree(where)[None, :] & (distance > world.rules["hide_reach"])
    return sees & ~hidden


def _obstacles_seen(me: '_Eyes', world: Any) -> np.ndarray:
    """Each looker's nearest obstacle in its field of vision as inputs (seen, distance to its edge ÷ vision reach, the
    direction to its centre (ahead, left), or for the water the sight line that found it), zeros when none is in view;
    and whether that obstacle is a pond or the water."""
    out, wet = np.zeros((len(me.p), 4)), np.zeros(len(me.p))
    terrain = world.terrain
    if terrain.obstacles:
        _caps_seen(me, terrain, out, wet)
    if terrain.water is not None:
        far, ahead, left = terrain.water_ahead(me.p, me.f, me.reach, me.half)
        closer = (far <= me.reach) & ((out[:, 0] == 0.0) | (far < out[:, 1] * me.reach))
        out[closer] = np.stack([np.ones(len(far)), far / me.reach, ahead, left], axis=1)[closer]
        wet[closer] = 1.0
    return out, wet


def _caps_seen(me: '_Eyes', terrain: Any, out: np.ndarray, wet: np.ndarray) -> None:
    """`_obstacles_seen` of the obstacles (rocks and ponds), written into `out` and `wet`."""
    edge, centres = terrain.nearest_edges(me.p, None if EVERY_PAIR else me.reach)
    near = edge <= me.reach[:, None]
    ahead, left = np.full(edge.shape, -np.inf), np.zeros(edge.shape)
    i, j = np.nonzero(near)
    if len(i):
        ahead[i, j], left[i, j] = bearing(me.p[i], me.f[i], centres[j])
    visible = near & (me.all_around[:, None] | (ahead >= me.half[:, None]) | (edge == 0))
    nearest = np.argmin(np.where(visible, edge, np.inf), axis=1)
    for i in np.nonzero(visible.any(axis=1))[0]:
        j = nearest[i]
        out[i] = (1.0, edge[i, j] / me.reach[i], ahead[i, j], left[i, j])
        wet[i] = 1.0 if terrain.pond_circles[j] else 0.0


def _sighting(creature: Creature, target: Any, reach: float, R: float) -> tuple:
    """A sighting or a sound as inputs: sensed, distance along the ground ÷ how far that sense reaches, and where it is
    in its own view (ahead, left)."""
    ahead, left = bearing(creature.p, creature.facing, target.p)
    return (1.0, float(arc(creature.p, target.p, R)) / reach, float(ahead), float(left))


def _relative_size(creature: Creature, other: Creature) -> float:
    return min(max(math.log2(other.size / creature.size), -1.0), 1.0)


def _vector(creature: Creature, food: Optional['Food'], other: Optional[Creature], on_food: Optional['Food'],
            heard: Optional[Creature], wall: np.ndarray, sheltered: bool, R: float) -> np.ndarray:
    vector = np.zeros(N_SENSES)
    vector[0] = creature.energy / creature.max_energy
    vector[1] = (creature.size - SIZE_CENTER) / SIZE_HALF_RANGE
    vector[2] = 1.0 if on_food is not None else 0.0
    if food is not None:
        vector[3:7] = _sighting(creature, food, creature.vision_reach, R)
    if other is not None:
        vector[7:12] = _sighting(creature, other, creature.vision_reach, R) + (_relative_size(creature, other),)
    if heard is not None:
        vector[12:16] = _sighting(creature, heard, creature.hearing, R)
    if creature.hurt is not None:
        vector[16:20] = (1.0,) + tuple(creature.hurt[:3])
    vector[20:24] = wall  # the nearest obstacle in its field of vision
    vector[24] = 1.0 if sheltered else 0.0  # under a tree's crown
    vector[25] = 1.0 if food is not None and food.meat else 0.0  # the nearest food it sees is meat
    vector[26] = creature.diet  # what it lives on
    if other is not None:
        vector[KIN_INPUT] = similarity(creature, other, creature.rules)  # how alike the one it sees is
    return vector


def _toward(creature: Creature, target: Any, dx: float, dy: float) -> float:
    """Cosine between the move (dx ahead, dy left) and the direction to the target, in the creature's own view."""
    ahead, left = bearing(creature.p, creature.facing, target.p)
    norm = math.hypot(dx, dy) * math.hypot(float(ahead), float(left))
    return (dx * float(ahead) + dy * float(left)) / norm if norm > 0 else 0.0


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
