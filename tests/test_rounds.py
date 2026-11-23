import itertools
import math
import random
import unittest

import numpy as np

from src.brain import Brain
from src.rounds import Member, next_generation


def members(n: int, lr=None):
    rng = np.random.default_rng(0)
    ids = itertools.count(1)
    return [Member(Brain.random(rng), size=1.0, lr=lr, id=next(ids)) for _ in range(n)]


def breed(scored, seed=0):
    ids = itertools.count(1000)
    return next_generation(scored, random.Random(seed), np.random.default_rng(seed), lambda: next(ids))


class TestRounds(unittest.TestCase):
    def test_keeps_eight_fittest_unchanged(self):
        crowd = members(30)
        scored = [(m, float(i)) for i, m in enumerate(crowd)]
        nxt = breed(scored)
        self.assertEqual(len(nxt), 30)
        self.assertEqual(nxt[:8], crowd[29:21:-1])

    def test_children_are_mutated_copies_in_rank_order(self):
        crowd = members(30)
        scored = [(m, float(i)) for i, m in enumerate(crowd)]
        nxt = breed(scored)
        parents = nxt[:8]
        for i, child in enumerate(nxt[8:]):
            parent = parents[i % 8]
            self.assertEqual(child.parent_id, parent.id)
            self.assertEqual(child.generation, parent.generation + 1)
            self.assertFalse(np.array_equal(child.genome.w1, parent.genome.w1)
                             and np.array_equal(child.genome.w2, parent.genome.w2))
        self.assertEqual(len({m.id for m in nxt}), 30)

    def test_sizes_stay_in_range(self):
        crowd = [Member(m.genome, size=s, lr=None, id=m.id) for m, s in zip(members(30), [0.5, 2.0] * 15)]
        nxt = breed([(m, 1.0) for m in crowd])
        self.assertTrue(all(0.5 <= m.size <= 2.0 for m in nxt))

    def test_lr_inherited_and_mutated(self):
        nxt = breed([(m, float(i)) for i, m in enumerate(members(30, lr=3e-4))])
        self.assertTrue(all(m.lr == 3e-4 for m in nxt[:8]))
        child_lrs = [m.lr for m in nxt[8:]]
        self.assertTrue(all(lr is not None and lr != 3e-4 for lr in child_lrs))
        self.assertTrue(all(abs(math.log(lr / 3e-4)) < 1.5 for lr in child_lrs))
        self.assertTrue(all(m.lr is None for m in breed([(m, 0.0) for m in members(30)])))

    def test_breeds_when_everyone_ate_nothing(self):
        crowd = members(30)
        nxt = breed([(m, 0.0) for m in crowd])
        self.assertEqual(nxt[:8], crowd[:8])
        self.assertEqual(len(nxt), 30)


if __name__ == "__main__":
    unittest.main()
