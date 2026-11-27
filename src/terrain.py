"""The ground a planet stands on: obstacles (rocks, and on a planet of before ponds, of random shape and size), its water
(seas, islands and rivers: `src/water.py`) and trees.

Nothing enters a rock, nothing grows on it, and nobody sees through it; a pond stops only who cannot swim, and hides
nothing. A tree drops fruit near its trunk now and then, and a creature under its crown is seen only from close by. An
obstacle is a blob of a few overlapping caps (a cap: a round patch of the ball, a centre point and an arc radius); a
tree is its trunk's point and its crown's arc radius. Distances are along the surface. The terrain is made once, when
the world is made (`make_terrain`), and kept in the world's saves. The water is water as a pond was: everything said of
a pond (it stops who cannot swim, hides nothing, food grows in it) holds for it.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .rules import Rules
from scipy.spatial import cKDTree

from .sphere import chord, cross, arc, exp_map, local_frame, neighbours, random_point
from .water import Water, make_water

EVERY_ROCK = False  # True: a sight line is tried against every rock, the slow way (the tests' yardstick)
EVERY_CAP = False  # True: a point is tried against every obstacle cap, the slow way (the tests' yardstick)

KINDS = ("rock", "pond")  # a rock stops everyone and hides what is behind it; a pond stops only who cannot swim
CIRCLES = (3, 6)  # caps making one obstacle's blob
PLACE_TRIES = 200  # tries to place one obstacle or tree where it fits
SPOT_TRIES = 1000  # tries for a random spot in the water
RAYS = 5  # sight lines across a looker's field of vision, looking for water
RAY_STEP = 0.5  # a sight line goes on at least this far between two looks (more where the water is farther)
ALL_AROUND = 0.8 * math.pi  # the outer sight lines of one who sees all around: 144 degrees either side (none twice)
Point = Tuple[float, float, float]
Cap = Tuple[Point, float]  # a centre point (unit vector) and an arc radius


@dataclass(frozen=True)
class Obstacle:
    kind: str  # one of KINDS
    caps: Tuple[Cap, ...]  # the caps of its blob


@dataclass(frozen=True)
class Tree:
    p: Point  # its trunk
    canopy: float  # the arc radius of its crown: under it a creature is hidden, near it fruit falls


def _rows(points: Sequence) -> np.ndarray:
    return np.array(points, dtype=float).reshape(-1, 3)


class Terrain:
    def __init__(self, obstacles: Sequence[Obstacle], trees: Sequence[Tree], radius: float = 1.0,
                 forest_centres: Sequence[Point] = (), water: Optional[Water] = None) -> None:
        self.obstacles, self.trees, self.radius = list(obstacles), list(trees), float(radius)
        self.water = water  # a planet's seas, islands and rivers (None on a planet of before: its ponds are its water)
        self.forest_centres = list(forest_centres)  # where a new world's forests grew from (not kept in a save)
        caps = [c for o in self.obstacles for c in o.caps]
        self._c = _rows([c[0] for c in caps])
        self._cr = np.array([c[1] for c in caps], dtype=float)
        self.pond_circles = np.array([o.kind == "pond" for o in self.obstacles for _ in o.caps], dtype=bool)
        self._tree = cKDTree(self._c) if len(self._c) else None  # the caps' centres, to find the ones near a point
        rock = ~self.pond_circles
        self._r, self._rr = self._c[rock], self._cr[rock]  # what hides what is behind it
        self._t = _rows([t.p for t in self.trees])
        self._tr = np.array([t.canopy for t in self.trees], dtype=float)

    @staticmethod
    def empty() -> Terrain:
        return Terrain([], [])

    def _arcs(self, centres: np.ndarray, points: np.ndarray) -> np.ndarray:
        """[point, centre]: the arc from each point to each centre."""
        return arc(np.atleast_2d(points)[:, None, :], centres[None, :, :], self.radius)

    def _near(self, p: np.ndarray, reach: float):
        """The caps whose centre is within `reach` of the point p, found by the k-d tree (all of them for EVERY_CAP)."""
        if EVERY_CAP:
            return slice(None)
        return self._tree.query_ball_point(np.asarray(p, dtype=float), float(chord(reach, self.radius)) * (1.0 + 1e-9) + 1e-12)

    def has_water(self) -> bool:
        """Whether there is any water: ponds, or a planet's water."""
        return bool(self.pond_circles.any()) or (self.water is not None and self.water.any_water())

    def blocked(self, p: np.ndarray, radius: float, swims: bool = False) -> bool:
        """Whether a body of `radius` at the point p would overlap an obstacle (only a rock, for one that `swims`):
        for one that cannot swim, the water is one too."""
        if self.water is not None and not swims:
            d = self.water.distance_at(p)
            if d == 0.0 or d < radius:
                return True
        if not self._cr.size:
            return False
        k = self._near(p, self._cr.max() + radius)
        overlap = self._arcs(self._c[k], p)[0] < self._cr[k] + radius
        return bool(np.any(overlap & ~self.pond_circles[k]) if swims else np.any(overlap))

    def in_pond(self, p: np.ndarray) -> bool:
        """Whether the point p is in a pond or the water."""
        if self.water is not None and self.water.distance_at(p) == 0.0:
            return True
        if not self.pond_circles.any():
            return False
        k = self._near(p, self._cr.max())
        return bool(np.any((self._arcs(self._c[k], p)[0] < self._cr[k]) & self.pond_circles[k]))

    def near_pond(self, p: np.ndarray, width: float) -> bool:
        """Whether p is on the land within `width` of a pond's edge or the water."""
        near = wet = False
        if self.water is not None:
            d = self.water.distance_at(p)
            near, wet = d <= width, d == 0.0
        if self.pond_circles.any():
            k = self._near(p, self._cr.max() + width)
            d, ponds, r = self._arcs(self._c[k], p)[0], self.pond_circles[k], self._cr[k]
            near, wet = near or bool(np.any(ponds & (d <= r + width))), wet or bool(np.any(ponds & (d < r)))
        return near and not wet

    def pond_spot(self, rng: random.Random) -> Optional[np.ndarray]:
        """A random point inside a random pond cap, or in the water on a planet with water (None when there is none)."""
        ponds = np.nonzero(self.pond_circles)[0]
        if not len(ponds):
            if self.water is not None and self.water.any_water():
                for _ in range(SPOT_TRIES):
                    p = random_point(rng)
                    if self.water.distance_at(p) == 0.0:
                        return p
            return None
        i = ponds[rng.randrange(len(ponds))]
        angle, away = rng.uniform(0.0, 2.0 * math.pi), self._cr[i] * math.sqrt(rng.random())
        e1, e2 = local_frame(self._c[i])
        return exp_map(self._c[i], e1, e2, away * math.cos(angle), away * math.sin(angle), self.radius)

    def sight_blocked(self, a: np.ndarray, b: np.ndarray, only: Optional[np.ndarray] = None) -> np.ndarray:
        """[looker, thing]: whether a rock stands on the shortest way along the ground from each looker (points `a`)
        to each thing (points `b`) (a pond hides nothing). With `only` ([looker, thing]), just those pairs are
        checked (the rest come back False)."""
        a, b = np.atleast_2d(a), np.atleast_2d(b)
        out = np.zeros((len(a), len(b)), dtype=bool)
        if not self._rr.size or not len(a) or not len(b):
            return out
        looker, thing = np.nonzero(np.ones(out.shape, dtype=bool) if only is None else only)
        if len(looker):
            out[looker, thing] = (_way_crosses(a[looker], b[thing], self._r, self._rr / self.radius) if EVERY_ROCK
                                  else self._crosses_near(a[looker], b[thing]))
        return out

    def _crosses_near(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """[pair]: whether a rock stands on the way from a to b, trying only the rocks a k-d tree finds near it: every
        point of the way is within half its length of its middle, so a rock that touches it has its centre within half
        the length plus its own radius of that middle (ways between near-opposite points try every rock)."""
        middle = a + b
        size = np.linalg.norm(middle, axis=-1)
        middle = np.where(size[:, None] > 1e-9, middle / np.maximum(size, 1e-9)[:, None], a)
        reach = np.where(size > 1e-9, arc(a, b, self.radius) / 2.0 + self._rr.max(), math.pi * self.radius)
        way, rock, _ = neighbours(middle, self._r, reach, self.radius)
        out = np.zeros(len(a), dtype=bool)
        if len(way):
            out[way[_way_near(a[way], b[way], self._r[rock], self._rr[rock] / self.radius)]] = True
        return out

    def water_ahead(self, points: np.ndarray, facing: np.ndarray, reach: np.ndarray,
                    half: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each looker (its point, facing, vision reach, and the cosine of half its field of vision): how far the
        nearest water in its field of vision is (inf when none within its reach, 0 when it is in it) and the direction
        of the sight line that found it (ahead, left). Only lookers with water within reach look: along `RAYS` sight
        lines across the field, each going on by the distance to the nearest water there (at least `RAY_STEP`)."""
        n = len(points)
        far, ahead, left = np.full(n, np.inf), np.zeros(n), np.zeros(n)
        if self.water is None or not n:
            return far, ahead, left
        start = self.water.distance_to_water(points)
        far[start == 0.0] = 0.0
        look = np.nonzero((start > 0.0) & (start <= reach))[0]
        if not len(look):
            return far, ahead, left
        spread = np.minimum(np.arccos(np.clip(half[look], -1.0, 1.0)), ALL_AROUND)
        angles = spread[:, None] * np.linspace(-1.0, 1.0, RAYS)[None, :]
        hit = self._rays(points[look], facing[look], reach[look], start[look], angles)
        rows = np.arange(len(look))
        best = np.argmin(hit, axis=1)
        nearest = hit[rows, best]
        got = np.isfinite(nearest)
        chosen = angles[rows, best][got]
        far[look[got]], ahead[look[got]], left[look[got]] = nearest[got], np.cos(chosen), np.sin(chosen)
        return far, ahead, left

    def water_rays(self, points: np.ndarray, facing: np.ndarray, reach: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """[looker, line]: how far the water is along each sight line (its angle, left of facing; nan: no line), inf
        when none within its reach, 0 when it stands in it."""
        out = np.full(angles.shape, np.inf)
        if self.water is None or not len(points):
            return out
        start = self.water.distance_to_water(points)
        out[start == 0.0] = 0.0
        look = np.nonzero((start > 0.0) & (start <= reach))[0]
        if len(look):
            out[look] = self._rays(points[look], facing[look], reach[look], start[look], angles[look])
        return out

    def _rays(self, p: np.ndarray, f: np.ndarray, reach: np.ndarray, start: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """[looker, line]: water along sight lines at `angles` (nan: none), each going on by the distance to the nearest
        water there (at least `RAY_STEP`), from `start` (no water nearer) to its reach."""
        way = f[:, None, :] * np.cos(angles)[..., None] + cross(p, f)[:, None, :] * np.sin(angles)[..., None]
        p = np.broadcast_to(p[:, None, :], way.shape)
        t = np.repeat(start[:, None], angles.shape[1], axis=1)
        limit = np.where(np.isnan(angles), -1.0, np.repeat(reach[:, None], angles.shape[1], axis=1))
        hit = np.full(t.shape, np.inf)
        line, ray = np.nonzero(t <= limit)
        while len(line):
            go = t[line, ray] / self.radius
            q = p[line, ray] * np.cos(go)[:, None] + way[line, ray] * np.sin(go)[:, None]
            d = self.water.distance_to_water(q)
            wet = d == 0.0
            hit[line[wet], ray[wet]] = t[line[wet], ray[wet]]
            t[line, ray] += np.maximum(d, RAY_STEP)
            on = ~wet & (t[line, ray] <= limit[line, ray])
            line, ray = line[on], ray[on]
        return hit

    def under_tree(self, points: np.ndarray) -> np.ndarray:
        """Whether each point is under a tree's crown."""
        points = np.atleast_2d(points)
        if not self._tr.size:
            return np.zeros(len(points), dtype=bool)
        return np.any(self._arcs(self._t, points) < self._tr[None, :], axis=1)

    def nearest_edges(self, points: np.ndarray, reach: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """[point, cap]: the arc from each point to each obstacle cap's edge (0 inside it); and the caps' centres. With
        `reach` (each point's), only the caps whose edge may lie within it are measured (a k-d tree finds them); the
        rest are inf."""
        if reach is None or not self._cr.size:
            return np.maximum(self._arcs(self._c, points) - self._cr[None, :], 0.0), self._c
        points = np.atleast_2d(points)
        i, j, d = neighbours(points, self._c, np.asarray(reach) + self._cr.max(), self.radius)
        out = np.full((len(points), len(self._c)), np.inf)
        out[i, j] = np.maximum(d - self._cr[j], 0.0)
        return out, self._c

    def state(self) -> Dict:
        return {"radius": self.radius,
                "obstacles": [{"kind": o.kind, "caps": [list(c[0]) + [c[1]] for c in o.caps]} for o in self.obstacles],
                "trees": [{"p": list(t.p), "canopy": t.canopy} for t in self.trees]}

    @staticmethod
    def from_state(state: Dict, water: Optional[Water] = None) -> Terrain:
        """The terrain of a save (its water is kept beside it: `Water.state`)."""
        return Terrain([Obstacle(o["kind"], tuple((tuple(c[:3]), c[3]) for c in o["caps"])) for o in state["obstacles"]],
                       [Tree(tuple(t["p"]), t["canopy"]) for t in state["trees"]], state["radius"], water=water)


def _angle(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.arctan2(np.linalg.norm(cross(p, q), axis=-1), np.sum(p * q, axis=-1))


def _way_crosses(a: np.ndarray, b: np.ndarray, centres: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """[pair]: whether the shortest way along the ground from a to b (pairs of points) comes within any cap's angle
    of its centre."""
    return np.any(_way_near(a[:, None, :], b[:, None, :], centres[None, :, :], angles[None, :]), axis=1)


def _way_near(a: np.ndarray, b: np.ndarray, c: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Whether the shortest way from a to b comes within `angle` of the centre c (all broadcast): the angle to the
    way's great circle where the centre's nearest point on it lies between a and b, else the angle to the nearer end."""
    pole = cross(a, b)
    length = np.linalg.norm(pole, axis=-1, keepdims=True)
    pole = np.where(length > 1e-12, pole / np.maximum(length, 1e-12), 0.0)  # 0 for a way of no length
    off = np.sum(c * pole, axis=-1)  # the sine of the centre's angle from the great circle
    foot = c - pole * off[..., None]  # its nearest point on that great circle (not normalised)
    between = ((np.sum(cross(a, foot) * pole, axis=-1) >= 0)
               & (np.sum(cross(foot, b) * pole, axis=-1) >= 0) & (length[..., 0] > 1e-12))
    to_way = np.where(between, np.arcsin(np.clip(np.abs(off), 0.0, 1.0)), np.minimum(_angle(a, c), _angle(b, c)))
    return to_way < angle


def make_terrain(rules: Rules, rng: random.Random, radius: float,
                 keep_clear: Sequence[Tuple[np.ndarray, float]] = ()) -> Terrain:
    """A new planet's ground, by its rules: the rocks (each of random size and shape), then its water (seas, islands,
    rivers), then the trees in their forests (anywhere on land when there are none), then `lone_trees` single trees
    anywhere on land; none on another, on a rock, in the water or on the spots in `keep_clear` (point, radius: creatures
    already there)."""
    obstacles: List[Obstacle] = []
    taken: List[Tuple[np.ndarray, float]] = [(np.asarray(p), r) for p, r in keep_clear]
    for _ in range(round(rules["obstacles"])):
        size = rng.uniform(rules["obstacle_size_low"], rules["obstacle_size_high"])
        for _ in range(PLACE_TRIES):
            centre = random_point(rng)
            e1, e2 = local_frame(centre)
            caps = [(centre, size * rng.uniform(0.4, 0.7))]
            for _ in range(rng.randint(*CIRCLES) - 1):  # more caps around the first: a blob of random shape
                angle, away = rng.uniform(0.0, 2.0 * math.pi), size * rng.uniform(0.2, 0.6)
                caps.append((exp_map(centre, e1, e2, away * math.cos(angle), away * math.sin(angle), radius),
                             size * rng.uniform(0.25, 0.5)))
            if all(_clear(c, taken, radius) for c in caps):
                obstacles.append(Obstacle("rock", tuple((tuple(map(float, p)), float(r)) for p, r in caps)))
                taken += caps
                break
    water = make_water(rules, rng, radius)
    rocks = Terrain(obstacles, [], radius, water=water)
    trees: List[Tree] = []
    canopy, spread = rules["tree_canopy"], rules["forest_spread"]
    centres = [_land_point(rocks, rng) for _ in range(round(rules["forests"]))]
    for _ in range(round(rules["trees"])):
        for _ in range(PLACE_TRIES):
            if centres:  # in a forest: within its spread of one of its centres
                c = centres[rng.randrange(len(centres))]
                e1, e2 = local_frame(c)
                angle, away = rng.uniform(0.0, 2.0 * math.pi), spread * math.sqrt(rng.random())
                p = exp_map(c, e1, e2, away * math.cos(angle), away * math.sin(angle), radius)
            else:
                p = random_point(rng)
            if _room_for_a_tree(p, rocks, trees, canopy, radius):
                trees.append(Tree(tuple(map(float, p)), canopy))
                break
    for _ in range(round(rules["lone_trees"])):  # and single trees anywhere on the land
        for _ in range(PLACE_TRIES):
            p = random_point(rng)
            if _room_for_a_tree(p, rocks, trees, canopy, radius):
                trees.append(Tree(tuple(map(float, p)), canopy))
                break
    return Terrain(obstacles, trees, radius, [tuple(map(float, c)) for c in centres], water)


def _room_for_a_tree(p: np.ndarray, ground: Terrain, trees: Sequence[Tree], canopy: float, radius: float) -> bool:
    """Whether a tree fits at p: on land, off every rock, and two crowns from every other tree."""
    return (not ground.blocked(p, canopy * 0.5)
            and all(float(arc(p, np.array(t.p), radius)) > canopy * 2 for t in trees))


def _land_point(ground: Terrain, rng: random.Random) -> np.ndarray:
    """A random point on land (any point, after `PLACE_TRIES` tries in the water)."""
    for _ in range(PLACE_TRIES):
        p = random_point(rng)
        if not ground.in_pond(p):
            return p
    return p


def _clear(cap: Tuple[np.ndarray, float], others: Sequence[Tuple[np.ndarray, float]], radius: float) -> bool:
    p, r = cap
    return all(float(arc(np.asarray(p), np.asarray(q), radius)) > r + other_r for q, other_r in others)
