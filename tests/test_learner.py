import math
import unittest

import numpy as np

from src.learner import LEARNING_RATE, Learner, NetWeights, gae


def learner_with(slots: int, seed: int = 0, lr: float = LEARNING_RATE) -> Learner:
    learner = Learner(slots, seed)
    rng = np.random.default_rng(seed)
    for _ in range(slots):
        learner.allocate(NetWeights.random(rng), lr)
    return learner


def same(a: NetWeights, b: NetWeights) -> bool:
    return all(np.allclose(x, y) for x, y in zip(a.arrays(), b.arrays()))


def food_senses(angles: np.ndarray) -> np.ndarray:
    senses = np.zeros((len(angles), 12))
    senses[:, 3] = 1.0
    senses[:, 4] = 0.5
    senses[:, 5] = np.cos(angles)
    senses[:, 6] = np.sin(angles)
    return senses


def toy_rewards(actions, angles: np.ndarray) -> np.ndarray:
    """Reward = cosine between the move and the food direction; eating or attacking earns nothing."""
    rewards = []
    for (choice, (dx, dy)), angle in zip(actions, angles):
        length = math.hypot(dx, dy)
        rewards.append((dx * math.cos(angle) + dy * math.sin(angle)) / length if choice == 0 and length > 0 else 0.0)
    return np.array(rewards)


def live(learner: Learner, slot: int, steps: int) -> None:
    """`steps` steps of life with something to sense (all-zero senses would leave the input weights untouched)."""
    for _ in range(steps):
        learner.act([slot], np.full((1, 12), 0.5), attacks_on=True)
        learner.record(slot, 1.0, False)


class TestLearner(unittest.TestCase):
    def test_act_shapes_and_ranges(self):
        learner = learner_with(4)
        actions = learner.act([0, 1, 2, 3], np.zeros((4, 12)), attacks_on=True)
        self.assertEqual(len(actions), 4)
        for choice, (dx, dy) in actions:
            self.assertIn(choice, (0, 1, 2))  # move, eat or attack
            self.assertTrue(-1.0 <= dx <= 1.0 and -1.0 <= dy <= 1.0)

    def test_attack_masked(self):
        learner = learner_with(10)
        choices = [choice for _ in range(100)
                   for choice, _ in learner.act(list(range(10)), np.zeros((10, 12)), attacks_on=False, learning=False)]
        self.assertEqual(set(choices), {0, 1})  # move or eat: attack is masked
        allowed = [choice for _ in range(100)
                   for choice, _ in learner.act(list(range(10)), np.zeros((10, 12)), attacks_on=True, learning=False)]
        self.assertEqual(set(allowed), {0, 1, 2})

    def test_gae_matches_hand_example(self):
        values = np.array([0.5, 0.2, 0.4])
        adv, ret = gae(np.array([1.0, 0.0, 2.0]), values, np.array([0.0, 0.0, 0.0]), 1.0, gamma=0.5, lam=1.0)
        np.testing.assert_allclose(adv, [1.125, 1.05, 2.1])
        np.testing.assert_allclose(ret, [1.625, 1.25, 2.5])
        adv, _ = gae(np.array([1.0, 0.0, 2.0]), values, np.array([0.0, 1.0, 0.0]), 1.0, gamma=0.5, lam=1.0)
        np.testing.assert_allclose(adv, [0.5, -0.2, 2.1])

    def test_allocate_and_release(self):
        learner = Learner(3, 0)
        rng = np.random.default_rng(0)
        nets = [NetWeights.random(rng) for _ in range(4)]
        self.assertEqual([learner.allocate(n, 1e-3 * (i + 1)) for i, n in enumerate(nets[:3])], [0, 1, 2])
        self.assertAlmostEqual(learner.lr(1), 2e-3)
        self.assertTrue(same(learner.inborn(2), nets[2]))
        with self.assertRaises(RuntimeError):
            learner.allocate(nets[3], 1e-3)
        learner.release(1)
        self.assertEqual(learner.allocate(nets[3], 5e-3), 1)
        self.assertTrue(same(learner.current(1), nets[3]))
        self.assertAlmostEqual(learner.lr(1), 5e-3)

    def test_update_changes_only_full_slots(self):
        learner = learner_with(2)
        for step in range(65):
            learner.act([0, 1], np.zeros((2, 12)), attacks_on=True)
            learner.record(0, 1.0, False)
            if step < 10:
                learner.record(1, 1.0, False)
        self.assertEqual(learner.update(), 1)
        self.assertFalse(same(learner.current(0), learner.inborn(0)))
        self.assertTrue(same(learner.current(1), learner.inborn(1)))

    def test_slot_reuse_starts_clean(self):
        learner = learner_with(1)
        live(learner, 0, 65)
        learner.update()  # the first animal has learned, and one step of its experience is left over
        learner.release(0)
        fresh = NetWeights.random(np.random.default_rng(99))
        self.assertEqual(learner.allocate(fresh, LEARNING_RATE), 0)
        self.assertTrue(same(learner.current(0), fresh))
        self.assertTrue(same(learner.inborn(0), fresh))
        live(learner, 0, 64)
        self.assertEqual(learner.update(), 0)  # the dead animal's leftover step did not count for the newborn
        live(learner, 0, 1)
        self.assertEqual(learner.update(), 1)

    def test_same_seed_same_actions(self):
        runs = []
        for _ in range(2):
            learner = learner_with(3, seed=5)
            runs.append([learner.act([0, 1, 2], np.ones((3, 12)) * 0.3, attacks_on=True) for _ in range(20)])
        self.assertEqual(runs[0], runs[1])

    def test_learns_toy_task(self):
        slots = list(range(8))
        learner = learner_with(8, seed=1)
        rng = np.random.default_rng(2)

        def mean_reward(calls: int) -> float:
            total = []
            for _ in range(calls):
                angles = rng.uniform(-math.pi, math.pi, 8)
                total.extend(toy_rewards(learner.act(slots, food_senses(angles), False, learning=False), angles))
            return float(np.mean(total))

        before = mean_reward(80)
        updates = 0
        while updates < 450 * 8:  # measured at the spec's learning rate: 0.77 after 300 updates and still rising
            angles = rng.uniform(-math.pi, math.pi, 8)
            rewards = toy_rewards(learner.act(slots, food_senses(angles), attacks_on=False), angles)
            for slot, reward in zip(slots, rewards):
                learner.record(slot, float(reward), False)
            updates += learner.update()
        after = mean_reward(80)
        self.assertLess(abs(before), 0.15)
        self.assertGreaterEqual(after, 0.8)


if __name__ == "__main__":
    unittest.main()
