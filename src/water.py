"""The water of a planet: an ocean, seas and lakes, islands out in open water, and rivers with a few shallow crossings.

Where water lies comes from a smooth height on the ball (a dozen gentle waves); below the sea level is water. Islands
are round land raised in open water, each far enough from all other land that walkers can never reach it. A river is a
chain of points from high land down the slope to the sea (or to a lake where it ends in a hollow), `river_width` wide,
with crossings where it is land. The rule itself (`wet_exact`) is turned once into a grid of cells on the six faces of a
cube around the ball, so asking "is this water?" is one lookup (`is_water`); and once more into each cell's distance to
the nearest water (`distance_to_water`, 0 in it), so "does this body reach water?" is one lookup too."""
from __future__ import annotations

import math
import random
import struct
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .sphere import chord, local_frame, random_point

CELL = 0.25  # a grid cell's size along the ground
MOST_CELLS = 512  # cells along a cube face's edge at most (a huge planet's cells are coarser)
WAVES = 12  # waves in the smooth height
STRIDE = 0.25  # a river's points are this far apart
MOST_STRIDES = 1600  # a river at most this many strides long (400 units)
LAKE = 2.0  # a river ending in a hollow ends in a lake this wide
TRIES = 200  # tries for each island's or river's place
SAMPLES = 20_000  # points for measuring the share of water
ISLAND_SIZE = (2.0, 6.0)  # an island's radius
SOURCES_APART = 15.0  # rivers start at least this far from each other
CROSSING = 0.6  # a crossing is land within this × the river's width of its point: just across the river


class Water:
    """The water of one planet (see the module). `waves` [k, 5]: unit direction, frequency, phase (each wave's height
    is sin(frequency × direction·p + phase) ÷ frequency); `islands` and `lakes` [m, 4]: centre and radius; `rivers`:
    one [n, 3] chain of points each; `crossings` [c, 3]: points where a river is land."""

    def __init__(self, radius: float, waves: Any, sea_level: float, islands: Any = (), rivers: Sequence[Any] = (),
                 crossings: Any = (), river_width: float = 1.5, lakes: Any = ()) -> None:
        self.radius = float(radius)
        self.waves = np.asarray(waves, dtype=float).reshape(-1, 5)
        self.sea_level = float(sea_level)
        self.islands = np.asarray(islands, dtype=float).reshape(-1, 4)
        self.lakes = np.asarray(lakes, dtype=float).reshape(-1, 4)
        self.rivers = [np.asarray(r, dtype=float).reshape(-1, 3) for r in rivers]
        self.crossings = np.asarray(crossings, dtype=float).reshape(-1, 3)
        self.river_width = float(river_width)
        flowing = [r for r in self.rivers if len(r)]
        self._river_tree = cKDTree(np.concatenate(flowing)) if flowing else None
        self._crossing_tree = cKDTree(self.crossings) if len(self.crossings) else None
        self._n = min(MOST_CELLS, max(8, math.ceil(math.pi * self.radius / 2.0 / CELL)))
        self._grid: Optional[np.ndarray] = None  # made the first time it is asked
        self._far: Optional[np.ndarray] = None  # each cell's distance to the nearest water: made the first time asked
        self._any: Optional[bool] = None

    # ---------- the rule
    def height(self, points: np.ndarray) -> np.ndarray:
        """The smooth height at each point: below the sea level is water (before islands, lakes and rivers)."""
        along = np.atleast_2d(points) @ self.waves[:, :3].T
        return np.sum(np.sin(along * self.waves[:, 3] + self.waves[:, 4]) / self.waves[:, 3], axis=1)

    def _inside(self, points: np.ndarray, discs: np.ndarray) -> np.ndarray:
        """Whether each point lies within any of the discs (centre, radius along the ground)."""
        if not len(discs):
            return np.zeros(len(points), dtype=bool)
        gap = np.linalg.norm(points[:, None, :] - discs[None, :, :3], axis=-1)
        return np.any(gap < chord(discs[:, 3], self.radius)[None, :], axis=1)

    def _near(self, points: np.ndarray, tree: Optional[cKDTree], distance: float) -> np.ndarray:
        if tree is None:
            return np.zeros(len(points), dtype=bool)
        found, _ = tree.query(points, distance_upper_bound=float(chord(distance, self.radius)))
        return np.isfinite(found)

    def open_water(self, points: np.ndarray) -> np.ndarray:
        """The sea and the lakes (not the rivers): below the sea level and off every island, or in a lake."""
        points = np.atleast_2d(points)
        return ((self.height(points) < self.sea_level) & ~self._inside(points, self.islands)) | self._inside(points, self.lakes)

    def wet_exact(self, points: np.ndarray) -> np.ndarray:
        """The rule itself: open water or within half a river's width of its points, but never at a crossing."""
        points = np.atleast_2d(points)
        river = self._near(points, self._river_tree, self.river_width / 2.0)
        crossing = self._near(points, self._crossing_tree, CROSSING * self.river_width)
        return (self.open_water(points) | river) & ~crossing

    # ---------- the grid: six cube faces of n × n cells, each judged at its centre
    def _cells(self, points: np.ndarray):
        p = np.atleast_2d(points)
        axis = np.argmax(np.abs(p), axis=1)
        rows = np.arange(len(p))
        major = np.abs(p[rows, axis])
        face = axis * 2 + (p[rows, axis] < 0)
        a = np.where(axis == 0, p[:, 1], p[:, 0])
        b = np.where(axis == 2, p[:, 1], p[:, 2])
        n = self._n
        i = np.clip(np.floor((np.arctan(a / major) * 4.0 / math.pi + 1.0) / 2.0 * n), 0, n - 1).astype(int)
        j = np.clip(np.floor((np.arctan(b / major) * 4.0 / math.pi + 1.0) / 2.0 * n), 0, n - 1).astype(int)
        return face, i, j

    def _cell(self, p: Sequence[float]) -> Tuple[int, int, int]:
        """The cell of one point, as `_cells` finds it, without arrays (a few microseconds)."""
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        ax, ay, az = abs(x), abs(y), abs(z)
        axis = 0 if ax >= ay and ax >= az else (1 if ay >= az else 2)
        major, value = ((ax, x), (ay, y), (az, z))[axis]
        a = y if axis == 0 else x
        b = y if axis == 2 else z
        n = self._n
        i = min(max(math.floor((math.atan(a / major) * 4.0 / math.pi + 1.0) / 2.0 * n), 0), n - 1)
        j = min(max(math.floor((math.atan(b / major) * 4.0 / math.pi + 1.0) / 2.0 * n), 0), n - 1)
        return axis * 2 + (value < 0), i, j

    def _centres(self, face: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        n = self._n
        a = np.tan(((i + 0.5) / n * 2.0 - 1.0) * math.pi / 4.0)
        b = np.tan(((j + 0.5) / n * 2.0 - 1.0) * math.pi / 4.0)
        axis, sign = face // 2, np.where(face % 2 == 1, -1.0, 1.0)
        p = np.zeros((len(face), 3))
        p[:, 0] = np.where(axis == 0, sign, a)
        p[:, 1] = np.where(axis == 0, a, np.where(axis == 2, b, sign))
        p[:, 2] = np.where(axis == 2, sign, b)
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    def cell_centres(self, points: np.ndarray) -> np.ndarray:
        """The centre of the cell each point falls in (where the grid judged it)."""
        return self._centres(*self._cells(points))

    def _every_centre(self) -> np.ndarray:
        n = self._n
        face, i, j = (x.ravel() for x in np.meshgrid(np.arange(6), np.arange(n), np.arange(n), indexing="ij"))
        return self._centres(face, i, j)

    def _made(self) -> np.ndarray:
        if self._grid is None:
            n = self._n
            centres = self._every_centre()
            wet = np.concatenate([self.wet_exact(chunk) for chunk in np.array_split(centres, max(1, len(centres) // 50_000))])
            self._grid = wet.reshape(6, n, n)
        return self._grid

    def is_water(self, points: np.ndarray) -> np.ndarray:
        """Whether each point is water: one lookup in the grid."""
        return self._made()[self._cells(points)]

    def is_water_at(self, p: np.ndarray) -> bool:
        return bool(self._made()[self._cell(p)])

    def any_water(self) -> bool:
        if self._any is None:
            self._any = bool(self._made().any())
        return self._any

    def _distances(self) -> np.ndarray:
        """[face, i, j]: each cell's distance along the ground to the nearest water: from its centre to the nearest wet
        cell's centre, less half a cell (to that cell's near side); 0 for a wet cell, inf on a planet without water."""
        if self._far is None:
            n, wet = self._n, self._made().ravel()
            far = np.zeros(len(wet))
            if not wet.any():
                far[:] = np.inf
            elif not wet.all():
                centres = self._every_centre()
                gap, _ = cKDTree(centres[wet]).query(centres[~wet], workers=-1)
                along = 2.0 * np.arcsin(np.minimum(gap / 2.0, 1.0)) * self.radius
                far[~wet] = np.maximum(along - math.pi * self.radius / 4.0 / n, 1e-9)
            self._far = far.reshape(6, n, n)
        return self._far

    def distance_to_water(self, points: np.ndarray) -> np.ndarray:
        """How far each point is from the nearest water, along the ground (to a cell's accuracy; 0 in water)."""
        return self._distances()[self._cells(points)]

    def distance_at(self, p: np.ndarray) -> float:
        """`distance_to_water` of one point, the fast way."""
        return float(self._distances()[self._cell(p)])

    # ---------- keeping and showing it
    def state(self) -> Dict[str, Any]:
        return {"radius": self.radius, "waves": self.waves.tolist(), "sea_level": self.sea_level,
                "islands": self.islands.tolist(), "lakes": self.lakes.tolist(),
                "rivers": [r.tolist() for r in self.rivers], "crossings": self.crossings.tolist(),
                "river_width": self.river_width}

    @staticmethod
    def from_state(state: Dict[str, Any]) -> Water:
        return Water(state["radius"], state["waves"], state["sea_level"], state["islands"], state["rivers"],
                     state["crossings"], state["river_width"], state.get("lakes", ()))

    def picture(self, width: int = 1024) -> bytes:
        """A PNG of the whole ball (equirectangular, as three.js wraps a picture on a sphere): blue where water,
        clear elsewhere."""
        height = width // 2
        u = (np.arange(width) + 0.5) / width * 2.0 * math.pi
        v = (np.arange(height) + 0.5) / height * math.pi
        phi, theta = np.meshgrid(u, v)
        points = np.stack([-np.cos(phi) * np.sin(theta), np.cos(theta), np.sin(phi) * np.sin(theta)], axis=-1)
        wet = self.is_water(points.reshape(-1, 3)).reshape(height, width)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[wet] = (38, 102, 178, 235)
        raw = b"".join(b"\x00" + row.tobytes() for row in rgba)

        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# ---------- making a planet's water
def _disc(centre: np.ndarray, distances: np.ndarray, angles: np.ndarray, radius: float) -> np.ndarray:
    """Points at each distance (along the ground) and angle around `centre`."""
    e1, e2 = local_frame(centre)
    d, a = np.meshgrid(distances / radius, angles)
    d, a = d.ravel()[:, None], a.ravel()[:, None]
    return centre * np.cos(d) + (np.cos(a) * e1 + np.sin(a) * e2) * np.sin(d)


def _downhill(water: Water, p: np.ndarray) -> Optional[np.ndarray]:
    """The way down the height's slope at p (unit, along the ground), or None on a flat spot."""
    w = water.waves
    slope = np.sum(np.cos(w[:, :3] @ p * w[:, 3] + w[:, 4])[:, None] * w[:, :3], axis=0)
    along = slope - p * float(slope @ p)
    size = float(np.linalg.norm(along))
    return None if size < 1e-9 else -along / size


def _sea_level(heights: np.ndarray, share: float) -> float:
    if share <= 0.0:
        return float(heights.min()) - 1.0
    if share >= 1.0:
        return float(heights.max()) + 1.0
    return float(np.quantile(heights, share))


def _build(rules: Dict[str, float], seed: int, radius: float, waves: np.ndarray, sea: float, peak: float) -> Water:
    rng = random.Random(seed)
    gap, width = rules["island_gap"], rules["river_width"]
    base = Water(radius, waves, sea, river_width=width)
    islands: List[List[float]] = []
    around = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    for _ in range(round(rules["islands"])):
        for _ in range(TRIES):
            c = random_point(rng)
            r = min(rng.uniform(*ISLAND_SIZE), radius * 0.1)
            if not base.open_water(_disc(c, np.linspace(0.0, r + gap, 8), around, radius)).all():
                continue  # it must stand in open water, with open water all around it
            if any(np.linalg.norm(c - np.array(o[:3])) < chord(r + o[3] + gap, radius) for o in islands):
                continue
            islands.append([*c, r])
            break
    rivers, crossings, lakes = [], [], []
    land = Water(radius, waves, sea, islands, river_width=width)
    for _ in range(round(rules["rivers"])):
        start, best = None, sea
        for _ in range(TRIES):  # its source: the highest of the land points tried, away from the other sources
            p = random_point(rng)
            h = float(land.height(p[None, :])[0])
            if h > best and not land.open_water(p[None, :])[0] and all(
                    np.linalg.norm(p - r[0]) >= chord(SOURCES_APART, radius) for r in rivers):
                start, best = p, h
        if start is None:
            continue
        path, p, reached = [start], start, False
        step = STRIDE / radius
        for _ in range(MOST_STRIDES):
            way = _downhill(land, p)
            if way is None:
                break
            p = p * math.cos(step) + way * math.sin(step)
            p = p / np.linalg.norm(p)
            path.append(p)
            if Water(radius, waves, sea, islands, lakes=lakes).open_water(p[None, :])[0]:
                reached = True
                break
        if not reached:  # it ends in a hollow: a lake there
            lakes.append([*path[-1], LAKE])
        river = np.array(path)
        rivers.append(river)
        for k in range(round(rules["river_crossings"])):
            crossings.append(river[int(len(river) * (k + 1) / (round(rules["river_crossings"]) + 1))])
    return Water(radius, waves, sea, islands, rivers, crossings, width, lakes)


def make_water(rules: Dict[str, float], rng: random.Random, radius: float) -> Water:
    """A new planet's water by its rules: `water_share` of it under water, `islands`, `rivers` with `river_crossings`.
    Sea level is set (a few tries) so that, islands and rivers included, the share of water is the rule's."""
    seed = rng.randrange(2 ** 32)
    own = random.Random(seed)
    waves = np.array([[*random_point(own), own.uniform(1.5, 6.0), own.uniform(0.0, 2.0 * math.pi)]
                      for _ in range(WAVES)])
    probe = Water(radius, waves, 0.0)
    samples = np.array([random_point(own) for _ in range(SAMPLES)])
    heights = probe.height(samples)
    peak, target = float(heights.max()), rules["water_share"]
    share, water = target, None
    for _ in range(4):
        water = _build(rules, seed + 1, radius, waves, _sea_level(heights, share), peak)
        measured = float(water.wet_exact(samples).mean())
        if abs(measured - target) < 0.005 or target <= 0.0:
            break
        share = min(1.0, max(0.0, share - (measured - target)))
    return water
