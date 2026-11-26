import math
import random
import unittest

import numpy as np

from src.food import Food
from src.policies import brain_action, learner_action, random_action
from src.senses import sense
from tests.helpers import add, fixed_brain, make_world


def food(x: float, y: float) -> Food:
    return Food(x, y, remaining_duration=-1, energy=10.0)


class TestBrainPolicy(unittest.TestCase):
    def test_brain_move_vector(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([0.6, 0.8, -1, -1]))
        kind, (dx, dy) = brain_action(c, sense(c, w), attacks_on=False)
        self.assertEqual(kind, "MOVE")
        np.testing.assert_allclose((dx, dy), (0.6 * c.top_speed, 0.8 * c.top_speed), atol=1e-9)

    def test_brain_move_capped(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([1, 1, -1, -1]))
        kind, (dx, dy) = brain_action(c, sense(c, w), attacks_on=False)
        self.assertEqual(kind, "MOVE")
        self.assertAlmostEqual(math.hypot(dx, dy), c.top_speed)

    def test_brain_eat_in_reach(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([0, 0, 1, -1]))
        f = food(10.5, 10)
        w.foods.append(f)
        self.assertEqual(brain_action(c, sense(c, w), attacks_on=False), ("EAT", f))

    def test_brain_eat_miss_when_nothing_in_reach(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([0, 0, 1, -1]))
        w.foods.append(food(12, 10))
        self.assertEqual(brain_action(c, sense(c, w), attacks_on=False), ("EAT", None))

    def test_brain_does_not_eat_unless_it_chooses_to(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([0, 0, -1, -1]))
        w.foods.append(food(10, 10))
        self.assertEqual(brain_action(c, sense(c, w), attacks_on=False)[0], "MOVE")

    def test_brain_attack_needs_attacks_on(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([0, 0, -1, 1]))
        stranger = add(w, 11, 10)
        self.assertEqual(brain_action(c, sense(c, w), attacks_on=False)[0], "MOVE")
        self.assertEqual(brain_action(c, sense(c, w), attacks_on=True), ("ATTACK", stranger))


class TestLearnerPolicy(unittest.TestCase):
    def test_learner_eat_choice(self):
        w = make_world()
        c = add(w, 10, 10)
        f = food(10.5, 10)
        w.foods.append(f)
        stranger = add(w, 11, 10)
        senses = sense(c, w)
        self.assertEqual(learner_action(c, senses, 1, (0.3, 0.3)), ("EAT", f))
        self.assertEqual(learner_action(c, senses, 2, (0.3, 0.3)), ("ATTACK", stranger))
        kind, (dx, dy) = learner_action(c, senses, 0, (0.3, 0.4))
        self.assertEqual(kind, "MOVE")
        np.testing.assert_allclose((dx, dy), (0.3 * c.top_speed, 0.4 * c.top_speed))


class TestRandomWalker(unittest.TestCase):
    def test_random_walker_eats_in_reach(self):
        w = make_world()
        c = add(w, 10, 10)
        f = food(10.5, 10)
        w.foods.append(f)
        self.assertEqual(random_action(c, sense(c, w), random.Random(0)), ("EAT", f))

    def test_brains_can_choose_any_speed_up_to_the_top(self):
        w = make_world()
        c = add(w, 10, 10, size=2.0)
        senses = sense(c, w)
        self.assertAlmostEqual(learner_action(c, senses, 0, (1.0, 0.0))[1][0], c.top_speed)
        self.assertAlmostEqual(learner_action(c, senses, 0, (0.1, 0.0))[1][0], 0.1 * c.top_speed)

    def test_random_walker_speed_within_velocity(self):
        w = make_world()
        c = add(w, 10, 10, size=0.5)
        rng = random.Random(0)
        moves = [random_action(c, sense(c, w), rng) for _ in range(100)]
        self.assertTrue(all(kind == "MOVE" for kind, _ in moves))
        speeds = [math.hypot(*vec) for _, vec in moves]
        self.assertTrue(all(s <= c.velocity + 1e-12 for s in speeds))
        self.assertGreater(len(set(round(s, 6) for s in speeds)), 50)


if __name__ == "__main__":
    unittest.main()
