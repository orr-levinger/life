"""The ground a planet stands on: obstacles (rocks and ponds of random shape and size) and trees.

Nothing enters a rock, nothing grows on it, and nobody sees through it; a pond stops only who cannot swim, and hides
nothing. A tree drops fruit near its trunk now and then, and a creature under its crown is seen only from close by. An
obstacle is a blob of a few overlapping caps (a cap: a round patch of the ball, a centre point and an arc radius); a
tree is its trunk's point and its crown's arc radius. Distances are along the surface. The terrain is made once, when
the world is made (`make_terrain`), and kept in the world's saves.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .rules import Rules
from .sphere import cross, arc, exp_map, local_frame, random_point

KINDS = ("rock", "pond")  # a rock stops everyone and hides what is behind it; a pond stops only who cannot swim
CIRCLES = (3, 6)  # caps making one obstacle's blob
PLACE_TRIES = 200  # tries to place one obstacle or tree where it fits
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
                 forest_centres: Sequence[Point] = ()) -> None:
        self.obstacles, self.trees, self.radius = list(obstacles), list(trees), float(radius)
        self.forest_centres = list(forest_centres)  # where a new world's forests grew from (not kept in a save)
        caps = [c for o in self.obstacles for c in o.caps]
        self._c = _rows([c[0] for c in caps])
        self._cr = np.array([c[1] for c in caps], dtype=float)
        self.pond_circles = np.array([o.kind == "pond" for o in self.obstacles for _ in o.caps], dtype=bool)
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

    def blocked(self, p: np.ndarray, radius: float, swims: bool = False) -> bool:
        """Whether a body of `radius` at the point p would overlap an obstacle (only a rock, for one that `swims`)."""
        if not self._cr.size:
            return False
        overlap = self._arcs(self._c, p)[0] < self._cr + radius
        return bool(np.any(overlap & ~self.pond_circles) if swims else np.any(overlap))

    def in_pond(self, p: np.ndarray) -> bool:
        """Whether the point p is in a pond."""
        if not self.pond_circles.any():
            return False
        return bool(np.any((self._arcs(self._c, p)[0] < self._cr) & self.pond_circles))

    def near_pond(self, p: np.ndarray, width: float) -> bool:
        """Whether p is on the land within `width` of a pond's edge."""
        if not self.pond_circles.any():
            return False
        d, ponds = self._arcs(self._c, p)[0], self.pond_circles
        return bool(np.any(ponds & (d <= self._cr + width)) and not np.any(ponds & (d < self._cr)))

    def pond_spot(self, rng: random.Random) -> Optional[np.ndarray]:
        """A random point inside a random pond cap (None when there is no pond)."""
        ponds = np.nonzero(self.pond_circles)[0]
        if not len(ponds):
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
            out[looker, thing] = _way_crosses(a[looker], b[thing], self._r, self._rr / self.radius)
        return out

    def under_tree(self, points: np.ndarray) -> np.ndarray:
        """Whether each point is under a tree's crown."""
        points = np.atleast_2d(points)
        if not self._tr.size:
            return np.zeros(len(points), dtype=bool)
        return np.any(self._arcs(self._t, points) < self._tr[None, :], axis=1)

    def nearest_edges(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """[point, cap]: the arc from each point to each obstacle cap's edge (0 inside it); and the caps' centres."""
        return np.maximum(self._arcs(self._c, points) - self._cr[None, :], 0.0), self._c

    def state(self) -> Dict:
        return {"radius": self.radius,
                "obstacles": [{"kind": o.kind, "caps": [list(c[0]) + [c[1]] for c in o.caps]} for o in self.obstacles],
                "trees": [{"p": list(t.p), "canopy": t.canopy} for t in self.trees]}

    @staticmethod
    def from_state(state: Dict) -> Terrain:
        return Terrain([Obstacle(o["kind"], tuple((tuple(c[:3]), c[3]) for c in o["caps"])) for o in state["obstacles"]],
                       [Tree(tuple(t["p"]), t["canopy"]) for t in state["trees"]], state["radius"])


def _angle(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.arctan2(np.linalg.norm(cross(p, q), axis=-1), np.sum(p * q, axis=-1))


def _way_crosses(a: np.ndarray, b: np.ndarray, centres: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """[pair]: whether the shortest way along the ground from a to b (pairs of points) comes within any cap's angle
    of its centre: the angle to the way's great circle where the centre's nearest point on it lies between a and b,
    else the angle to the nearer end."""
    pole = cross(a, b)
    length = np.linalg.norm(pole, axis=-1, keepdims=True)
    pole = np.where(length > 1e-12, pole / np.maximum(length, 1e-12), 0.0)  # 0 for a way of no length
    c = centres[None, :, :]
    off = np.sum(c * pole[:, None, :], axis=-1)  # the sine of the centre's angle from the great circle
    foot = c - pole[:, None, :] * off[..., None]  # its nearest point on that great circle (not normalised)
    between = ((np.sum(cross(a[:, None, :], foot) * pole[:, None, :], axis=-1) >= 0)
               & (np.sum(cross(foot, b[:, None, :]) * pole[:, None, :], axis=-1) >= 0) & (length > 1e-12))
    to_way = np.where(between, np.arcsin(np.clip(np.abs(off), 0.0, 1.0)),
                      np.minimum(_angle(a[:, None, :], c), _angle(b[:, None, :], c)))
    return np.any(to_way < angles[None, :], axis=1)


def make_terrain(rules: Rules, rng: random.Random, radius: float,
                 keep_clear: Sequence[Tuple[np.ndarray, float]] = ()) -> Terrain:
    """A new planet's ground, by its rules: the obstacles (each of random size and shape), then the trees in their
    forests (anywhere when there are none), none on another or on the spots in `keep_clear` (point, radius: creatures
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
                obstacles.append(Obstacle(rng.choice(KINDS), tuple((tuple(map(float, p)), float(r)) for p, r in caps)))
                taken += caps
                break
    rocks = Terrain(obstacles, [], radius)
    trees: List[Tree] = []
    canopy, spread = rules["tree_canopy"], rules["forest_spread"]
    centres = [random_point(rng) for _ in range(round(rules["forests"]))]
    for _ in range(round(rules["trees"])):
        for _ in range(PLACE_TRIES):
            if centres:  # in a forest: within its spread of one of its centres
                c = centres[rng.randrange(len(centres))]
                e1, e2 = local_frame(c)
                angle, away = rng.uniform(0.0, 2.0 * math.pi), spread * math.sqrt(rng.random())
                p = exp_map(c, e1, e2, away * math.cos(angle), away * math.sin(angle), radius)
            else:
                p = random_point(rng)
            if (not rocks.blocked(p, canopy * 0.5)
                    and all(float(arc(p, np.array(t.p), radius)) > canopy * 2 for t in trees)):
                trees.append(Tree(tuple(map(float, p)), canopy))
                break
    return Terrain(obstacles, trees, radius, [tuple(map(float, c)) for c in centres])


def _clear(cap: Tuple[np.ndarray, float], others: Sequence[Tuple[np.ndarray, float]], radius: float) -> bool:
    p, r = cap
    return all(float(arc(np.asarray(p), np.asarray(q), radius)) > r + other_r for q, other_r in others)
