import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from src.brain import Brain
from src.exam import exam_worlds, run_exam
from src.learner import NetWeights
from src.stages import TEST_WORLD


def brains(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [Brain.random(rng) for _ in range(n)]


class TestExam(unittest.TestCase):
    def test_copies_start_identical(self):
        with_brains, with_walkers = exam_worlds(brains(3), [0.5, 1.0, 2.0], seed=4)
        self.assertGreater(len(with_brains.foods), 0)
        self.assertEqual([(f.x, f.y, f.energy) for f in with_brains.foods],
                         [(f.x, f.y, f.energy) for f in with_walkers.foods])
        self.assertEqual([(c.x, c.y, c.size) for c in with_brains.creatures],
                         [(c.x, c.y, c.size) for c in with_walkers.creatures])
        self.assertEqual({c.policy for c in with_brains.creatures}, {"brain"})
        self.assertEqual({c.policy for c in with_walkers.creatures}, {"random"})

    def test_exam_uses_the_fixed_test_world(self):
        for world in exam_worlds(brains(2), [1.0, 1.0], seed=1):
            self.assertIs(world.stage, TEST_WORLD)
            self.assertFalse(world.evolving)

    def test_same_seed_same_result(self):
        sizes = [0.7, 1.3, 1.9, 1.0]
        self.assertEqual(run_exam(brains(4), sizes, seed=8, steps=100), run_exam(brains(4), sizes, seed=8, steps=100))

    def test_ratio_none_when_walkers_never_bite(self):
        with mock.patch("src.exam.TEST_WORLD", replace(TEST_WORLD, max_food=0, food_per_step=0)):
            result = run_exam(brains(2), [1.0, 1.0], seed=1, steps=50)
        self.assertEqual(result.walker_bites_per_step, 0.0)
        self.assertEqual(result.brain_bites_per_step, 0.0)
        self.assertIsNone(result.ratio)

    def test_exam_leaves_brains_unchanged(self):
        sitters = brains(3)
        before = [b.copy() for b in sitters]
        run_exam(sitters, [1.0, 1.5, 0.5], seed=2, steps=50)
        for b, original in zip(sitters, before):
            for name in ("w1", "b1", "w2", "b2"):
                np.testing.assert_array_equal(getattr(b, name), getattr(original, name))

    def test_exam_leaves_learners_unchanged(self):
        rng = np.random.default_rng(0)
        nets = [NetWeights.random(rng) for _ in range(3)]
        with mock.patch("src.learner.Learner.update") as update, mock.patch("src.learner.Learner.record") as record:
            result = run_exam(nets, [1.0, 1.5, 0.5], seed=2, steps=50)
        update.assert_not_called()
        record.assert_not_called()
        self.assertIsInstance(result.brain_bites_per_step, float)


if __name__ == "__main__":
    unittest.main()
