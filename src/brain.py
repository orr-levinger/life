"""A creature's brain: a small fixed-size neural network that never trains; it changes only by mutation."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def jitter(values: np.ndarray, rng: np.random.Generator, rate: float, scale: float) -> np.ndarray:
    """The one mutation rule for any brain's weights: each, with probability `rate`, gets normal noise of std `scale`."""
    mask = rng.random(values.shape) < rate
    return values + mask * rng.normal(0.0, scale, values.shape)


class Brain:
    N_IN = 21
    N_HIDDEN = 16
    N_OUT = 4  # move x, move y, eat, attack

    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
        shapes = {
            "w1": (w1.shape, (self.N_IN, self.N_HIDDEN)),
            "b1": (b1.shape, (self.N_HIDDEN,)),
            "w2": (w2.shape, (self.N_HIDDEN, self.N_OUT)),
            "b2": (b2.shape, (self.N_OUT,)),
        }
        for name, (got, want) in shapes.items():
            if got != want:
                raise ValueError(f"{name} has shape {got}, expected {want}")
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    @classmethod
    def random(cls, rng: np.random.Generator) -> Brain:
        """Weights drawn from a normal distribution scaled by 1/sqrt(inputs to the layer); biases start at 0."""
        return cls(
            rng.normal(0.0, 1.0 / np.sqrt(cls.N_IN), (cls.N_IN, cls.N_HIDDEN)),
            np.zeros(cls.N_HIDDEN),
            rng.normal(0.0, 1.0 / np.sqrt(cls.N_HIDDEN), (cls.N_HIDDEN, cls.N_OUT)),
            np.zeros(cls.N_OUT),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """The 21 senses in, 4 outputs in -1..1 out."""
        if x.shape != (self.N_IN,):
            raise ValueError(f"the brain takes {self.N_IN} inputs, got shape {x.shape}")
        hidden = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)

    def copy(self) -> Brain:
        return Brain(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

    def mutate(self, rng: np.random.Generator, rate: float, scale: float) -> Brain:
        """A new brain: each weight and bias, with probability `rate`, gets normal noise of std `scale`."""
        return Brain(*(jitter(values, rng, rate, scale) for values in (self.w1, self.b1, self.w2, self.b2)))

    def save(self, path: str | Path) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path: str | Path) -> Brain:
        with np.load(path) as data:
            return cls(data["w1"], data["b1"], data["w2"], data["b2"])
