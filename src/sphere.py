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
    return np.linalg.norm(v, axis=-1)


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)[..., None]
    return np.where(n > TINY, v / np.maximum(n, TINY), 0.0)


def random_point(rng: Any) -> np.ndarray:
    """A point drawn evenly over the whole ball (`rng` is a random.Random or a NumPy generator)."""
    z, a = rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2.0 * math.pi)
    s = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([s * math.cos(a), s * math.sin(a), z])


def arc(p: np.ndarray, q: np.ndarray, R: float) -> np.ndarray:
    """The distance from p to q along the surface (exact for tiny and for near-opposite distances)."""
    return R * np.arctan2(_norm(np.cross(p, q)), np.sum(p * q, axis=-1))


def tangent_toward(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The ground direction at p toward q (unit), zero when q is p or the point opposite."""
    return _unit(q - p * np.sum(p * q, axis=-1)[..., None])


def bearing(p: np.ndarray, f: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Where q is, seen from p facing f: (ahead, left), the parts of the direction toward q along f and to its left."""
    t = tangent_toward(p, q)
    return np.sum(t * f, axis=-1), np.sum(t * np.cross(p, f), axis=-1)


def walk(p: np.ndarray, f: np.ndarray, ahead: float, left: float, d: float, R: float) -> Tuple[np.ndarray, np.ndarray]:
    """From p facing f, go d along the ground in the direction (ahead, left) of its own view; it then faces the way it
    went (that direction carried to where it arrived). No direction or no distance: it stays, facing as before."""
    t = _unit(ahead * f + left * np.cross(p, f))
    if d <= 0.0 or float(_norm(t)) <= TINY:
        return p, f
    a = d / R
    p2 = p * math.cos(a) + t * math.sin(a)
    t2 = t * math.cos(a) - p * math.sin(a)
    return _unit(p2), _unit(t2)


def local_frame(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two ground directions at p at right angles (e2 to the left of e1): any such pair, the same for the same p."""
    axis = np.array([0.0, 0.0, 1.0]) if abs(float(p[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(axis, p))
    return e1, np.cross(p, e1)


def exp_map(base: np.ndarray, e1: np.ndarray, e2: np.ndarray, x: float, y: float, R: float) -> np.ndarray:
    """The point reached from `base` by going x along e1 and y along e2 as one straight walk of hypot(x, y): a local
    map around `base` (exact distance from it; nearly flat close to it)."""
    d = math.hypot(x, y)
    if d <= TINY:
        return base.copy()
    t = (x * e1 + y * e2) / d
    return _unit(base * math.cos(d / R) + t * math.sin(d / R))
