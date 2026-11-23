"""Evolution in rounds: the fittest of a round are kept, and their mutated children fill the next round."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .creature import Creature

ROUND_SIZE = 30
KEEP = 8  # the fittest kept unchanged into the next round
MUTATION_RATE = 0.1
MUTATION_SCALE = 0.2
SIZE_MUTATION = 0.5  # a child's size is its parent's × (1 ± up to this)
LR_MUTATION = 0.2  # a child's learning rate is its parent's × e^N(0, this)


@dataclass(frozen=True)
class Member:
    genome: Any  # the inborn brain: anything with mutate(np_rng, rate, scale)
    size: float
    lr: Optional[float]  # learning rate of a learning brain; None for a brain that never learns
    id: int
    parent_id: int = 0
    generation: int = 0


def next_generation(scored: Sequence[Tuple[Member, float]], rng: random.Random, np_rng: np.random.Generator,
                    next_id: Callable[[], int], n: int = ROUND_SIZE, keep: int = KEEP) -> List[Member]:
    """The `keep` fittest unchanged (ties keep their order), then mutated children of them in rank order, round-robin."""
    ranked = [member for member, _ in sorted(scored, key=lambda pair: -pair[1])]
    parents = ranked[:keep]
    children = []
    for i in range(n - len(parents)):
        parent = parents[i % len(parents)]
        size = parent.size * (1.0 + rng.uniform(-SIZE_MUTATION, SIZE_MUTATION))
        lr = parent.lr * math.exp(rng.gauss(0.0, LR_MUTATION)) if parent.lr is not None else None
        children.append(Member(
            genome=parent.genome.mutate(np_rng, MUTATION_RATE, MUTATION_SCALE),
            size=min(max(size, Creature.MIN_SIZE), Creature.MAX_SIZE),
            lr=lr,
            id=next_id(),
            parent_id=parent.id,
            generation=parent.generation + 1,
        ))
    return parents + children
