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
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .rules import Rules

KINDS = ("rock", "pond")  # how an obstacle looks; both are the same to the creatures
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
    def __init__(self, obstacles: Sequence[Obstacle], trees: Sequence[Tree]) -> None:
        self.obstacles, self.trees = list(obstacles), list(trees)
        circles = [c for o in self.obstacles for c in o.circles]
        self._cx = np.array([c[0] for c in circles])
        self._cy = np.array([c[1] for c in circles])
        self._cr = np.array([c[2] for c in circles])
        self._tx = np.array([t.x for t in self.trees])
        self._ty = np.array([t.y for t in self.trees])
        self._tr = np.array([t.canopy for t in self.trees])

    @staticmethod
    def empty() -> Terrain:
        return Terrain([], [])

    def blocked(self, x: float, y: float, radius: float) -> bool:
        """Whether a body of `radius` at (x, y) would overlap an obstacle."""
        return bool(self._cr.size and np.any(np.hypot(self._cx - x, self._cy - y) < self._cr + radius))

    def sight_blocked(self, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
        """[looker, thing]: whether an obstacle stands between each looker (ax, ay) and each thing (bx, by)."""
        out = np.zeros((len(ax), len(bx)), dtype=bool)
        if not self._cr.size or not len(ax) or not len(bx):
            return out
        dx, dy = bx[None, :] - ax[:, None], by[None, :] - ay[:, None]  # [L, T]
        length2 = np.maximum(dx * dx + dy * dy, 1e-12)
        for cx, cy, cr in zip(self._cx, self._cy, self._cr):  # a handful of circles: one array pass each
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
    canopy = rules["tree_canopy"]
    for _ in range(round(rules["trees"])):
        for _ in range(PLACE_TRIES):
            x, y = rng.uniform(canopy, width - canopy), rng.uniform(canopy, height - canopy)
            if not rocks.blocked(x, y, canopy * 0.5) and all(math.hypot(x - t.x, y - t.y) > canopy * 2 for t in trees):
                trees.append(Tree(x, y, canopy))
                break
    return Terrain(obstacles, trees)


def _clear(circle: Tuple[float, float, float], others: Sequence[Tuple[float, float, float]]) -> bool:
    x, y, r = circle
    return all(math.hypot(x - ox, y - oy) > r + other_r for ox, oy, other_r in others)
