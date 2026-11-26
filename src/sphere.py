"""The planet's geometry. A place is a unit vector (a point on the ball); the world's radius turns angles into
distances along the surface. A creature faces a ground direction (a unit vector at right angles to its place); what
it senses and how it moves are given in its own view: (ahead, left).

Every function takes single points (shape (3,)) or arrays of them (shape (..., 3)) and broadcasts."""
from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

TINY = 1e-12  # below this a direction has no length: two points are the same or opposite


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=-1))


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a × b over the last axis (np.cross's general machinery costs more than the sum for one point or a few)."""
    return np.stack(np.broadcast_arrays(a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]), axis=-1)


def _cross_length(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a × b| over the last axis, without building the cross product."""
    x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return np.sqrt(x * x + y * y + z * z)


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)[..., None]
    return np.where(n > TINY, v / np.maximum(n, TINY), 0.0)


def random_point(rng: Any) -> np.ndarray:
    """A point drawn evenly over the whole ball (`rng` is a random.Random or a NumPy generator)."""
    z, a = rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2.0 * math.pi)
    s = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([s * math.cos(a), s * math.sin(a), z])


def _one(*points: np.ndarray) -> bool:
    """Whether every argument is a single point: then plain arithmetic is far quicker than array calls."""
    return all(np.ndim(v) == 1 for v in points)


def _cross3(a, b) -> Tuple[float, float, float]:
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def arc(p: np.ndarray, q: np.ndarray, R: float) -> np.ndarray:
    """The distance from p to q along the surface (exact for tiny and for near-opposite distances)."""
    if _one(p, q):
        c = _cross3(p, q)
        return R * math.atan2(math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]), p[0] * q[0] + p[1] * q[1] + p[2] * q[2])
    return R * np.arctan2(_cross_length(p, q), np.sum(p * q, axis=-1))


def tangent_toward(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The ground direction at p toward q (unit), zero when q is p or the point opposite."""
    return _unit(q - p * np.sum(p * q, axis=-1)[..., None])


def bearing(p: np.ndarray, f: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Where q is, seen from p facing f: (ahead, left), the parts of the direction toward q along f and to its left."""
    if _one(p, f, q):
        d = p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
        t = (q[0] - p[0] * d, q[1] - p[1] * d, q[2] - p[2] * d)
        n = math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
        if n <= TINY:
            return 0.0, 0.0
        left = _cross3(p, f)
        return ((t[0] * f[0] + t[1] * f[1] + t[2] * f[2]) / n, (t[0] * left[0] + t[1] * left[1] + t[2] * left[2]) / n)
    t = tangent_toward(p, q)
    return np.sum(t * f, axis=-1), np.sum(t * cross(p, f), axis=-1)


def walk(p: np.ndarray, f: np.ndarray, ahead: float, left: float, d: float, R: float) -> Tuple[np.ndarray, np.ndarray]:
    """From p facing f, go d along the ground in the direction (ahead, left) of its own view; it then faces the way it
    went (that direction carried to where it arrived). No direction or no distance: it stays, facing as before."""
    t = _unit(ahead * f + left * cross(p, f))
    if d <= 0.0 or float(_norm(t)) <= TINY:
        return p, f
    a = d / R
    p2 = p * math.cos(a) + t * math.sin(a)
    t2 = t * math.cos(a) - p * math.sin(a)
    return _unit(p2), _unit(t2)


def local_frame(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two ground directions at p at right angles (e2 to the left of e1): any such pair, the same for the same p."""
    axis = np.array([0.0, 0.0, 1.0]) if abs(float(p[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(cross(axis, p))
    return e1, cross(p, e1)


def exp_map(base: np.ndarray, e1: np.ndarray, e2: np.ndarray, x: float, y: float, R: float) -> np.ndarray:
    """The point reached from `base` by going x along e1 and y along e2 as one straight walk of hypot(x, y): a local
    map around `base` (exact distance from it; nearly flat close to it)."""
    d = math.hypot(x, y)
    if d <= TINY:
        return base.copy()
    t = (x * e1 + y * e2) / d
    return _unit(base * math.cos(d / R) + t * math.sin(d / R))
