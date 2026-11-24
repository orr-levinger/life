import math
import unittest
from types import SimpleNamespace

import numpy as np

from src.creature import Creature, is_family
from src.food import Food
from src.senses import display_intent, sense


def creature(x: float, y: float, size: float = 1.25, id_: int = 1, parent_id: int = 0) -> Creature:
    return Creature(id_, x, y, size, size * 50.0, "random", parent_id=parent_id)


def food(x: float, y: float) -> Food:
    return Food(x, y, remaining_duration=-1, energy=10.0)


def world(creatures=(), foods=()):
    return SimpleNamespace(creatures=list(creatures), foods=list(foods))


class TestSenses(unittest.TestCase):
    def test_nothing_seen(self):
        me = creature(10, 10)
        s = sense(me, world([me]))
        self.assertEqual(s.vector.shape, (21,))
        self.assertAlmostEqual(s.vector[0], 0.5)
        self.assertAlmostEqual(s.vector[1], 0.0)
        np.testing.assert_array_equal(s.vector[2:], np.zeros(19))
        self.assertIsNone(s.food)
        self.assertIsNone(s.other)
        self.assertIsNone(s.on_food)

    def test_food_seen_east(self):
        me = creature(10, 10)
        f = food(11.5, 10)
        s = sense(me, world([me], [f]))
        np.testing.assert_allclose(s.vector[3:7], [1.0, 0.5, 1.0, 0.0], atol=1e-12)
        self.assertIs(s.food, f)

    def test_food_out_of_range(self):
        me = creature(10, 10)
        s = sense(me, world([me], [food(13.5, 10)]))
        self.assertEqual(s.vector[3], 0.0)
        self.assertIsNone(s.food)

    def test_family_is_invisible(self):
        parent = creature(10, 10, id_=1)
        child = creature(11, 10, id_=2, parent_id=1)
        stranger = creature(12, 10, id_=3)
        s = sense(parent, world([parent, child, stranger]))
        self.assertIs(s.other, stranger)
        self.assertEqual(s.vector[7], 1.0)
        self.assertAlmostEqual(s.vector[8], 2.0 / 3.0)

    def test_relative_size_clipped(self):
        small = creature(10, 10, size=0.5, id_=1)
        big = creature(11, 10, size=2.0, id_=2)
        self.assertEqual(sense(small, world([small, big])).vector[11], 1.0)
        self.assertEqual(sense(big, world([small, big])).vector[11], -1.0)
        twin = creature(11, 10, size=0.5, id_=3)
        self.assertEqual(sense(small, world([small, twin])).vector[11], 0.0)

    def test_on_food_flag(self):
        me = creature(10, 10)
        f = food(10, 10)
        s = sense(me, world([me], [f]))
        self.assertEqual(s.vector[2], 1.0)
        self.assertIs(s.on_food, f)

    def test_display_intent_toward_food(self):
        me = creature(10, 10)
        s = sense(me, world([me], [food(12, 10)]))
        self.assertEqual(display_intent(me, ("MOVE", (1.0, 0.0)), s), "GO_TO_FOOD")
        self.assertEqual(display_intent(me, ("MOVE", (0.0, 1.0)), s), "WANDER")
        self.assertEqual(display_intent(me, ("MOVE", (0.0, 0.0)), s), "REST")

    def test_display_intent_run_away(self):
        me = creature(10, 10, size=1.0, id_=1)
        hunter = creature(12, 10, size=2.0, id_=2)
        s = sense(me, world([me, hunter]))
        self.assertEqual(display_intent(me, ("MOVE", (-1.0, 0.0)), s), "RUN_AWAY")
        self.assertEqual(display_intent(me, ("ATTACK", hunter), s), "ATTACK")

    def eye(self, x, y, width, reach, hearing, size=1.25):
        me = Creature(1, x, y, size, size * 50.0, "random", vision_width=width, vision_reach=reach, hearing=hearing)
        me.facing = (1.0, 0.0)
        return me

    def test_food_behind_the_cone_is_unseen(self):
        me = self.eye(10, 10, width=90.0, reach=3.0, hearing=0.0)
        self.assertEqual(sense(me, world([me], [food(8, 10)])).vector[3], 0.0)  # behind: unseen
        ahead = sense(me, world([me], [food(12, 10)])).vector
        self.assertEqual(ahead[3], 1.0)
        self.assertAlmostEqual(ahead[4], 2.0 / 3.0)  # distance ÷ its own reach
        self.assertEqual(sense(me, world([me], [food(11, 12)])).vector[3], 0.0)  # 63° off its heading: outside a 90° cone

    def test_a_creature_behind_is_heard(self):
        me = self.eye(10, 10, width=90.0, reach=3.0, hearing=2.0)
        other = creature(9, 10, size=2.5, id_=2)
        v = sense(me, world([me, other])).vector
        self.assertEqual(v.shape, (21,))
        self.assertEqual(v[7], 0.0)  # not seen
        self.assertEqual(v[12], 1.0)  # heard
        self.assertAlmostEqual(v[13], 0.5)  # distance ÷ its hearing radius
        self.assertAlmostEqual(v[14], -1.0)  # behind it: west
        self.assertAlmostEqual(v[15], 0.0, places=6)
        self.assertAlmostEqual(v[16], 1.0)  # twice its size
        far = creature(7.5, 10, id_=3)
        self.assertEqual(sense(me, world([me, far])).vector[12], 0.0)  # 2.5 away: past its hearing

    def test_food_is_not_heard(self):
        me = self.eye(10, 10, width=90.0, reach=3.0, hearing=2.0)
        v = sense(me, world([me], [food(9, 10)])).vector
        self.assertEqual(v[3], 0.0)
        self.assertTrue(np.all(v[12:17] == 0.0))

    def test_is_family(self):
        parent = creature(0, 0, id_=1)
        child = creature(0, 0, id_=2, parent_id=1)
        sibling = creature(0, 0, id_=3, parent_id=1)
        stranger = creature(0, 0, id_=4)
        orphan = creature(0, 0, id_=5)
        self.assertTrue(is_family(parent, child))
        self.assertTrue(is_family(child, parent))
        self.assertTrue(is_family(child, sibling))
        self.assertFalse(is_family(parent, stranger))
        self.assertFalse(is_family(stranger, orphan))
        self.assertTrue(math.isclose(child.radius, 0.25))


if __name__ == "__main__":
    unittest.main()
