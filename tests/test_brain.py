import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.brain import N_SENSES

from src.brain import Brain


def _params(brain: Brain) -> np.ndarray:
    return np.concatenate([brain.w1.ravel(), brain.b1.ravel(), brain.w2.ravel(), brain.b2.ravel()])


class TestBrain(unittest.TestCase):
    def test_forward_shape_and_range(self):
        out = Brain.random(np.random.default_rng(0)).forward(np.ones(N_SENSES))
        self.assertEqual(out.shape, (4,))  # move x, move y, eat, attack
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))

    def test_random_is_seeded(self):
        x = np.linspace(-1.0, 1.0, N_SENSES)
        a = Brain.random(np.random.default_rng(7)).forward(x)
        b = Brain.random(np.random.default_rng(7)).forward(x)
        np.testing.assert_array_equal(a, b)

    def test_copy_is_equal_and_independent(self):
        x = np.linspace(-1.0, 1.0, N_SENSES)
        original = Brain.random(np.random.default_rng(3))
        before = original.forward(x)
        copy = original.copy()
        np.testing.assert_array_equal(copy.forward(x), before)
        copy.w1[0, 0] += 1.0
        np.testing.assert_array_equal(original.forward(x), before)

    def test_mutate_changes_about_rate(self):
        parent = Brain.random(np.random.default_rng(5))
        parent_params = _params(parent).copy()
        child = parent.mutate(np.random.default_rng(1), 0.2, 0.2)
        self.assertEqual(parent_params.size, N_SENSES * 16 + 16 + 16 * 4 + 4)  # its inputs, 16 hidden, 4 out
        changed = np.mean(_params(child) != parent_params)
        self.assertGreaterEqual(changed, 0.15)
        self.assertLessEqual(changed, 0.25)
        np.testing.assert_array_equal(_params(parent), parent_params)

    def test_save_load_round_trip(self):
        x = np.linspace(-1.0, 1.0, N_SENSES)
        brain = Brain.random(np.random.default_rng(9))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "brain.npz"
            brain.save(path)
            loaded = Brain.load(path)
        np.testing.assert_array_equal(loaded.forward(x), brain.forward(x))

    def test_wrong_input_size_raises(self):
        with self.assertRaises(ValueError):
            Brain.random(np.random.default_rng(0)).forward(np.ones(11))


if __name__ == "__main__":
    unittest.main()
