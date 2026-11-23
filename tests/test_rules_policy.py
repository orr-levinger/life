import random
import unittest

from src.food import Food
from src.policies import rules_action
from tests.helpers import add, make_world


class TestRulesPolicy(unittest.TestCase):
    def test_flee_from_bigger(self):
        w = make_world()
        me = add(w, 10, 10, size=1.0, policy="rules")
        add(w, 11, 10, size=2.0, policy="rules")
        kind, (dx, _) = rules_action(me, w, random.Random(0))
        self.assertEqual(kind, "MOVE")
        self.assertLess(dx, 0)

    def test_attack_smaller_in_range(self):
        w = make_world()
        me = add(w, 10, 10, size=2.0, policy="rules")
        small = add(w, 11, 10, size=1.0, policy="rules")
        self.assertEqual(rules_action(me, w, random.Random(0)), ("ATTACK", small))

    def test_parent_does_not_attack_child(self):
        w = make_world()
        parent = add(w, 10, 10, size=2.0, policy="rules")
        add(w, 11, 10, size=1.0, policy="rules", parent_id=parent.id)
        self.assertNotEqual(rules_action(parent, w, random.Random(0))[0], "ATTACK")

    def test_child_does_not_attack_parent(self):
        w = make_world()
        parent = add(w, 11, 10, size=1.0, policy="rules")
        child = add(w, 10, 10, size=2.0, policy="rules", parent_id=parent.id)
        self.assertNotEqual(rules_action(child, w, random.Random(0))[0], "ATTACK")

    def test_goes_toward_visible_food(self):
        w = make_world()
        me = add(w, 10, 10, policy="rules")
        w.foods.append(Food(12, 10, remaining_duration=-1, energy=10.0))
        kind, (dx, dy) = rules_action(me, w, random.Random(0))
        self.assertEqual(kind, "MOVE")
        self.assertGreater(dx, 0)
        self.assertAlmostEqual(dy, 0.0)

    def test_eats_food_in_reach(self):
        w = make_world()
        me = add(w, 10, 10, policy="rules")
        f = Food(10.5, 10, remaining_duration=-1, energy=10.0)
        w.foods.append(f)
        self.assertEqual(rules_action(me, w, random.Random(0)), ("EAT", f))


if __name__ == "__main__":
    unittest.main()
