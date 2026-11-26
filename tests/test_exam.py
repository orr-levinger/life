import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from src.brain import Brain
from src.exam import exam_worlds, run_exam
from src.learner import NetWeights
from src.exam import exam_settings
from src.rules import starting_rules

RULES = starting_rules()
TEST_WORLD = exam_settings(RULES)
from tests.helpers import LIVING, NO_FOOD, fixed_brain

EAT = [0, 0, 1, -1]


def brains(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [Brain.random(rng) for _ in range(n)]


class TestExam(unittest.TestCase):
    def test_the_exam_pays_todays_costs(self):  # the living world's cost knobs never reach the exam
        import math
        exam = exam_settings(starting_rules())
        today = {"sprint_power": 2.0, "eat_miss": 0.2, "attack_cost": 1.0, "attack_damage": 5.0, "bump_cost": 0.5,
                 "vision_cost": 0.05 / (math.pi * 9.0), "hearing_cost": 0.05 / (math.pi * 9.0), "neuron_cost": 0.0005,
                 "stealth_cost": 0.3, "armour_cost": 0.3}
        for name, now in today.items():
            self.assertAlmostEqual(getattr(exam, name), now, msg=name)

    def test_the_exam_planet_stays_as_it_was(self):
        from src.exam import _exam_world
        from src.rules import starting_rules
        world = _exam_world("random", [None, None], [1.0, 2.0], 0, starting_rules())
        self.assertEqual(world.radius, 10.9)  # its own small planet (the 50 × 30 exam's ground), whatever a new one's
        self.assertEqual([c.max_energy for c in world.creatures], [100.0, 200.0])  # its bodies as they were
        self.assertAlmostEqual(world.creatures[1].velocity, 0.125)

    def test_copies_start_identical(self):
        with_brains, with_walkers = exam_worlds(brains(3), [0.5, 1.0, 2.0], seed=4, rules=RULES)
        self.assertGreater(len(with_brains.foods), 0)
        self.assertEqual([(tuple(f.p), f.energy) for f in with_brains.foods],
                         [(tuple(f.p), f.energy) for f in with_walkers.foods])
        self.assertEqual([(tuple(c.p), c.size) for c in with_brains.creatures],
                         [(tuple(c.p), c.size) for c in with_walkers.creatures])
        self.assertEqual({c.policy for c in with_brains.creatures}, {"brain"})
        self.assertEqual({c.policy for c in with_walkers.creatures}, {"random"})

    def test_exam_uses_the_fixed_test_world(self):
        for world in exam_worlds(brains(2), [1.0, 1.0], seed=1, rules=RULES):
            self.assertEqual(world.settings, TEST_WORLD)
            self.assertFalse(world.evolving)

    def test_same_seed_same_result(self):
        sizes = [0.7, 1.3, 1.9, 1.0]
        self.assertEqual(run_exam(brains(4), sizes, seed=8, rules={**RULES, "exam_steps": 100}), run_exam(brains(4), sizes, seed=8, rules={**RULES, "exam_steps": 100}))

    def test_intelligence_is_energy_earned_over_spent(self):
        """With nothing to eat, a brain that always eats earns nothing and spends the miss, the step and its senses
        each step."""
        no_food = {**RULES, "exam_max_food": 5.0, "exam_food_per_step": 0.1, "exam_steps": 50}
        with mock.patch("src.world.World.spawn_food", lambda self: None):  # nothing to eat
            result = run_exam([fixed_brain(EAT)] * 2, [1.0, 1.0], seed=1, rules=no_food)
            with_brains, _ = exam_worlds([fixed_brain(EAT)] * 2, [1.0, 1.0], seed=1, rules=no_food)
        self.assertEqual((result.intelligence, result.wanderers), (0.0, 0.0))
        for _ in range(10):
            with_brains.step()
        self.assertEqual(with_brains.energy_earned, 0.0)
        self.assertAlmostEqual(with_brains.energy_spent, 2 * 10 * (0.2 + TEST_WORLD.step_cost + LIVING - NO_FOOD.step_cost))

    def test_wanderers_are_measured_beside_the_brains(self):
        result = run_exam(brains(6), [1.0] * 6, seed=3, rules={**RULES, "exam_steps": 200})
        self.assertGreater(result.wanderers, 0.0)  # random wanderers find some food in the test world
        self.assertGreaterEqual(result.intelligence, 0.0)

    def test_exam_leaves_brains_unchanged(self):
        sitters = brains(3)
        before = [b.copy() for b in sitters]
        run_exam(sitters, [1.0, 1.5, 0.5], seed=2, rules={**RULES, "exam_steps": 50})
        for b, original in zip(sitters, before):
            for name in ("w1", "b1", "w2", "b2"):
                np.testing.assert_array_equal(getattr(b, name), getattr(original, name))

    def test_exam_leaves_learners_unchanged(self):
        rng = np.random.default_rng(0)
        nets = [NetWeights.random(rng) for _ in range(3)]
        with mock.patch("src.learner.Learner.update") as update, mock.patch("src.learner.Learner.record") as record:
            result = run_exam(nets, [1.0, 1.5, 0.5], seed=2, rules={**RULES, "exam_steps": 50})
        update.assert_not_called()
        record.assert_not_called()
        self.assertIsInstance(result.intelligence, float)


if __name__ == "__main__":
    unittest.main()
