"""The ground a world stands on: obstacles (rocks and ponds of random shape and size) and trees.

Nothing enters an obstacle and nothing grows on it, and nobody sees through it. A tree drops fruit near its trunk now
and then, and a creature under its canopy is seen only from close by, so a hunter may wait there. An obstacle is a
blob of a few overlapping circles; a tree is its trunk and canopy. The terrain is made once, when the world is made
(`make_terrain`), and kept in the world's saves.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .rules import Rules

KINDS = ("rock", "pond")  # a rock stops everyone and hides what is behind it; a pond stops only who cannot swim
CIRCLES = (3, 6)  # circles making one obstacle's blob
PLACE_TRIES = 200  # tries to place one obstacle or tree where it fits


@dataclass(frozen=True)
class Obstacle:
    kind: str  # one of KINDS
    circles: Tuple[Tuple[float, float, float], ...]  # (x, y, radius) of each circle of its blob


@dataclass(frozen=True)
class Tree:
    x: float
    y: float
    canopy: float  # the radius of its crown: under it a creature is hidden, near it fruit falls


class Terrain:
    def __init__(self, obstacles: Sequence[Obstacle], trees: Sequence[Tree],
                 forest_centres: Sequence[Tuple[float, float]] = ()) -> None:
        self.obstacles, self.trees = list(obstacles), list(trees)
        self.forest_centres = list(forest_centres)  # where a new world's forests grew from (not kept in a save)
        circles = [c for o in self.obstacles for c in o.circles]
        self._cx = np.array([c[0] for c in circles])
        self._cy = np.array([c[1] for c in circles])
        self._cr = np.array([c[2] for c in circles])
        self.pond_circles = np.array([o.kind == "pond" for o in self.obstacles for _ in o.circles], dtype=bool)
        rock = ~self.pond_circles
        self._rx, self._ry, self._rr = self._cx[rock], self._cy[rock], self._cr[rock]  # what hides what is behind it
        self._tx = np.array([t.x for t in self.trees])
        self._ty = np.array([t.y for t in self.trees])
        self._tr = np.array([t.canopy for t in self.trees])

    @staticmethod
    def empty() -> Terrain:
        return Terrain([], [])

    def blocked(self, p: np.ndarray, radius: float, swims: bool = False) -> bool:
        """Whether a body of `radius` at the point p would overlap an obstacle (only a rock, for one that `swims`)."""
        if not self._cr.size:
            return False
        raise NotImplementedError("rocks and ponds move onto the sphere in Task 3 of the sphere plan")

    def in_pond(self, p: np.ndarray) -> bool:
        """Whether the point p is in a pond."""
        if not self.pond_circles.any():
            return False
        raise NotImplementedError("rocks and ponds move onto the sphere in Task 3 of the sphere plan")

    def near_pond(self, x: float, y: float, width: float) -> bool:
        """Whether (x, y) is on the land within `width` of a pond's edge."""
        if not self.pond_circles.any():
            return False
        d = np.hypot(self._cx - x, self._cy - y)
        ponds = self.pond_circles
        return bool(np.any(ponds & (d <= self._cr + width)) and not np.any(ponds & (d < self._cr)))

    def pond_spot(self, rng) -> Optional[Tuple[float, float]]:
        """A random point inside a random pond circle (None when there is no pond)."""
        ponds = np.nonzero(self.pond_circles)[0]
        if not len(ponds):
            return None
        i = ponds[rng.randrange(len(ponds))]
        angle, away = rng.uniform(0.0, 2.0 * math.pi), self._cr[i] * math.sqrt(rng.random())
        return float(self._cx[i] + away * math.cos(angle)), float(self._cy[i] + away * math.sin(angle))

    def sight_blocked(self, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray,
                      only: Optional[np.ndarray] = None) -> np.ndarray:
        """[looker, thing]: whether a rock stands between each looker (ax, ay) and each thing (bx, by) (a pond hides
        nothing). With
        `only` ([looker, thing]), just those pairs are checked (the rest come back False): one pass over the few pairs
        within reach and cone instead of every pair."""
        out = np.zeros((len(ax), len(bx)), dtype=bool)
        if not self._rr.size or not len(ax) or not len(bx):
            return out
        if only is not None:
            looker, thing = np.nonzero(only)
            px, py = ax[looker], ay[looker]
            dx, dy = bx[thing] - px, by[thing] - py
            length2 = np.maximum(dx * dx + dy * dy, 1e-12)
            t = np.clip(((self._rx[None, :] - px[:, None]) * dx[:, None] + (self._ry[None, :] - py[:, None]) * dy[:, None])
                        / length2[:, None], 0.0, 1.0)  # [pair, rock circle]
            nearest = np.hypot(px[:, None] + t * dx[:, None] - self._rx[None, :],
                               py[:, None] + t * dy[:, None] - self._ry[None, :])
            out[looker, thing] = np.any(nearest < self._rr[None, :], axis=1)
            return out
        dx, dy = bx[None, :] - ax[:, None], by[None, :] - ay[:, None]  # [L, T]
        length2 = np.maximum(dx * dx + dy * dy, 1e-12)
        for cx, cy, cr in zip(self._rx, self._ry, self._rr):  # a handful of rock circles: one array pass each
            t = np.clip(((cx - ax)[:, None] * dx + (cy - ay)[:, None] * dy) / length2, 0.0, 1.0)
            nearest = np.hypot(ax[:, None] + t * dx - cx, ay[:, None] + t * dy - cy)
            out |= nearest < cr
        return out

    def under_tree(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Whether each point is under a tree's canopy."""
        if not self._tr.size:
            return np.zeros(len(x), dtype=bool)
        return np.any(np.hypot(x[:, None] - self._tx[None, :], y[:, None] - self._ty[None, :]) < self._tr[None, :], axis=1)

    def nearest_edges(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """[looker, circle]: the distance from each point to each obstacle circle's edge, and the direction (dx, dy)
        to its centre."""
        dx, dy = self._cx[None, :] - x[:, None], self._cy[None, :] - y[:, None]
        return np.maximum(np.hypot(dx, dy) - self._cr[None, :], 0.0), dx, dy

    def state(self) -> Dict:
        return {"obstacles": [{"kind": o.kind, "circles": [list(c) for c in o.circles]} for o in self.obstacles],
                "trees": [{"x": t.x, "y": t.y, "canopy": t.canopy} for t in self.trees]}

    @staticmethod
    def from_state(state: Dict) -> Terrain:
        return Terrain([Obstacle(o["kind"], tuple(tuple(c) for c in o["circles"])) for o in state["obstacles"]],
                       [Tree(t["x"], t["y"], t["canopy"]) for t in state["trees"]])


def make_terrain(rules: Rules, rng: random.Random, width: float, height: float,
                 keep_clear: Sequence[Tuple[float, float, float]] = ()) -> Terrain:
    """A new world's ground, by its rules: the obstacles (each of random size and shape), then the trees, none of them
    on another or on the spots in `keep_clear` (x, y, radius: creatures already there)."""
    obstacles: List[Obstacle] = []
    for _ in range(round(rules["obstacles"])):
        size = rng.uniform(rules["obstacle_size_low"], rules["obstacle_size_high"])
        for _ in range(PLACE_TRIES):
            x, y = rng.uniform(size, width - size), rng.uniform(size, height - size)
            circles = [(x, y, size * rng.uniform(0.4, 0.7))]
            for _ in range(rng.randint(*CIRCLES) - 1):  # more circles around the first: a blob of random shape
                angle, away = rng.uniform(0.0, 2.0 * math.pi), size * rng.uniform(0.2, 0.6)
                circles.append((x + away * math.cos(angle), y + away * math.sin(angle), size * rng.uniform(0.25, 0.5)))
            if all(_clear(c, keep_clear) and _clear(c, [d for o in obstacles for d in o.circles]) for c in circles):
                obstacles.append(Obstacle(rng.choice(KINDS), tuple(circles)))
                break
    rocks = Terrain(obstacles, [])
    trees: List[Tree] = []
    canopy, spread = rules["tree_canopy"], rules["forest_spread"]
    centres = [(rng.uniform(0.0, width), rng.uniform(0.0, height)) for _ in range(round(rules["forests"]))]
    for _ in range(round(rules["trees"])):
        for _ in range(PLACE_TRIES):
            if centres:  # in a forest: within its spread of one of the centres
                cx, cy = rng.choice(centres)
                angle, away = rng.uniform(0.0, 2.0 * math.pi), spread * math.sqrt(rng.random())
                x, y = cx + away * math.cos(angle), cy + away * math.sin(angle)
            else:
                x, y = rng.uniform(canopy, width - canopy), rng.uniform(canopy, height - canopy)
            inside = canopy <= x <= width - canopy and canopy <= y <= height - canopy
            if (inside and not rocks.blocked(x, y, canopy * 0.5)
                    and all(math.hypot(x - t.x, y - t.y) > canopy * 2 for t in trees)):
                trees.append(Tree(x, y, canopy))
                break
    return Terrain(obstacles, trees, centres)


def _clear(circle: Tuple[float, float, float], others: Sequence[Tuple[float, float, float]]) -> bool:
    x, y, r = circle
    return all(math.hypot(x - ox, y - oy) > r + other_r for ox, oy, other_r in others)
