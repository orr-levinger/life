import math
import random
import unittest

import numpy as np

from src.food import Food
from src.policies import brain_action, random_action, rules_action
from src.senses import sense
from tests.helpers import add, fixed_brain, make_world


class TestBrainPolicy(unittest.TestCase):
    def test_brain_move_vector(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([0.6, 0.8, -1]))
        kind, (dx, dy) = brain_action(c, sense(c, w), attacks_on=False)
        self.assertEqual(kind, "MOVE")
        np.testing.assert_allclose((dx, dy), (0.6 * c.velocity, 0.8 * c.velocity), atol=1e-9)

    def test_brain_move_capped(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([1, 1, -1]))
        kind, (dx, dy) = brain_action(c, sense(c, w), attacks_on=False)
        self.assertEqual(kind, "MOVE")
        self.assertAlmostEqual(math.hypot(dx, dy), c.velocity)

    def test_brain_attack_needs_attacks_on(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([0, 0, 1]))
        stranger = add(w, 11, 10)
        self.assertEqual(brain_action(c, sense(c, w), attacks_on=False)[0], "MOVE")
        self.assertEqual(brain_action(c, sense(c, w), attacks_on=True), ("ATTACK", stranger))

    def test_brain_never_targets_family(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain([0, 0, 1]))
        add(w, 11, 10, parent_id=c.id)
        self.assertEqual(brain_action(c, sense(c, w), attacks_on=True)[0], "MOVE")

    def test_nobody_decides_to_eat(self):
        w = make_world()
        w.foods.append(Food(10.3, 10, remaining_duration=-1, energy=10.0))
        brainy = add(w, 10, 10, policy="brain", brain=fixed_brain([0.2, 0.2, -1]))
        walker = add(w, 10, 10.2)
        ruled = add(w, 10.2, 10.2, policy="rules")
        kinds = {brain_action(brainy, sense(brainy, w), True)[0], random_action(walker, sense(walker, w), random.Random(0))[0],
                 rules_action(ruled, w, random.Random(0))[0]}
        self.assertLessEqual(kinds, {"MOVE", "ATTACK"})


class TestRandomWalker(unittest.TestCase):
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
