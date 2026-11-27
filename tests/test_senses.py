import math
import unittest

from types import SimpleNamespace

import numpy as np

from src.creature import Creature, full_tank, is_family
from src.food import Food
from src.brain import N_SENSES
from src.rules import starting_rules
from src.senses import display_intent, perceived, sense
from src.terrain import Terrain
from tests.helpers import TEST_RADIUS, at, face


def creature(x: float, y: float, size: float = 1.25, id_: int = 1, parent_id: int = 0) -> Creature:
    c = Creature(id_, at(x, y), size, full_tank(size, starting_rules()) / 2, "random", parent_id=parent_id)  # half full
    c.rules = world().rules
    face(c, 1.0, 0.0)  # east on the map
    return c


def food(x: float, y: float) -> Food:
    return Food(at(x, y), remaining_duration=-1, energy=10.0)


def world(creatures=(), foods=()):
    rules = {**starting_rules(), "planet_radius": TEST_RADIUS}  # the tests' nearly flat planet
    return SimpleNamespace(creatures=list(creatures), foods=list(foods), terrain=Terrain.empty(), rules=rules,
                           radius=TEST_RADIUS)


class TestSenses(unittest.TestCase):
    def test_nothing_seen(self):
        me = creature(10, 10)
        s = sense(me, world([me]))
        self.assertEqual(s.vector.shape, (N_SENSES,))
        self.assertAlmostEqual(s.vector[0], 0.5)
        self.assertAlmostEqual(s.vector[1], 0.0)
        np.testing.assert_array_equal(s.vector[2:], np.zeros(N_SENSES - 2))
        self.assertIsNone(s.food)
        self.assertIsNone(s.other)
        self.assertIsNone(s.on_food)

    def test_food_seen_east(self):
        me = creature(10, 10)
        f = food(11.5, 10)
        s = sense(me, world([me], [f]))
        np.testing.assert_allclose(s.vector[3:7], [1.0, 0.5, 1.0, 0.0], atol=1e-4)  # the test map bends a little
        self.assertIs(s.food, f)

    def test_food_out_of_range(self):
        me = creature(10, 10)
        s = sense(me, world([me], [food(13.5, 10)]))
        self.assertEqual(s.vector[3], 0.0)
        self.assertIsNone(s.food)

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

    def test_any_bigger_one_is_run_away_from_and_any_smaller_one_chased(self):
        me = creature(10, 10, size=1.0, id_=1)
        slightly_bigger = creature(12, 10, size=1.1, id_=2)
        self.assertEqual(display_intent(me, ("MOVE", (-1.0, 0.0)), sense(me, world([me, slightly_bigger]))), "RUN_AWAY")
        slightly_smaller = creature(12, 10, size=0.9, id_=3)
        self.assertEqual(display_intent(me, ("MOVE", (1.0, 0.0)), sense(me, world([me, slightly_smaller]))), "CHASE")

    def test_display_intent_chase(self):
        me = creature(10, 10, size=2.0, id_=1)
        prey = creature(12, 10, size=1.0, id_=2)
        s = sense(me, world([me, prey], [food(10, 12)]))
        self.assertEqual(display_intent(me, ("MOVE", (1.0, 0.0)), s), "CHASE")  # toward a smaller one
        self.assertEqual(display_intent(me, ("MOVE", (-0.05, 1.0)), s), "GO_TO_FOOD")  # to the food, not toward the prey

    def test_food_ahead_is_ahead_anywhere(self):
        from src.sphere import local_frame, random_point, walk
        import random
        rng = random.Random(3)
        for _ in range(30):
            p = random_point(rng)
            e1, e2 = local_frame(p)
            turn = rng.uniform(0, 6.28)
            me = Creature(1, p, 1.25, 60.0, "random", vision_width=90.0, vision_reach=5.0, hearing=0.0)
            me.rules = world().rules
            me.facing = math.cos(turn) * e1 + math.sin(turn) * e2
            ahead, _ = walk(p, me.facing, 1.0, 0.0, 2.0, TEST_RADIUS)
            v = sense(me, world([me], [Food(ahead, remaining_duration=-1, energy=10.0)])).vector
            self.assertEqual(v[3], 1.0)
            self.assertAlmostEqual(v[4], 2.0 / 5.0)
            np.testing.assert_allclose(v[5:7], (1.0, 0.0), atol=1e-6)  # ahead of it, wherever it is

    def test_a_call_from_the_left_is_heard_from_the_left(self):
        me = self.eye(10, 10, width=90.0, reach=3.0, hearing=2.0)
        caller = creature(10, 13, id_=2)  # 3 north: to its left (it faces east)
        w = world([me, caller])
        w.calls = [(caller, 0, 1.0)]
        from src.senses import CALL_INPUTS, N_CALLS
        v = sense(me, w).vector
        np.testing.assert_allclose(v[CALL_INPUTS + N_CALLS + 1:CALL_INPUTS + N_CALLS + 3], (0.0, 1.0), atol=1e-4)

    def eye(self, x, y, width, reach, hearing, size=1.25):
        me = Creature(1, at(x, y), size, size * 50.0, "random", vision_width=width, vision_reach=reach, hearing=hearing)
        me.rules = world().rules
        face(me, 1.0, 0.0)
        return me

    def test_food_behind_the_cone_is_unseen(self):
        me = self.eye(10, 10, width=90.0, reach=3.0, hearing=0.0)
        self.assertEqual(sense(me, world([me], [food(8, 10)])).vector[3], 0.0)  # behind: unseen
        ahead = sense(me, world([me], [food(12, 10)])).vector
        self.assertEqual(ahead[3], 1.0)
        self.assertAlmostEqual(ahead[4], 2.0 / 3.0, places=4)  # distance ÷ its own reach
        self.assertEqual(sense(me, world([me], [food(11, 12)])).vector[3], 0.0)  # 63° off its heading: outside a 90° cone

    def test_a_creature_behind_is_heard(self):
        me = self.eye(10, 10, width=90.0, reach=3.0, hearing=2.0)
        other = creature(9, 10, size=2.5, id_=2)
        other.moved = True  # it moved last step, so it makes a sound
        v = sense(me, world([me, other])).vector
        self.assertEqual(v.shape, (N_SENSES,))
        self.assertEqual(v[7], 0.0)  # not seen
        self.assertEqual(v[12], 1.0)  # heard
        self.assertAlmostEqual(v[13], 0.5, places=4)  # distance ÷ its hearing radius
        self.assertAlmostEqual(v[14], -1.0)  # behind it: west
        self.assertAlmostEqual(v[15], 0.0, places=6)
        far = creature(7.0, 10, id_=3)
        far.moved = True
        self.assertEqual(sense(me, world([me, far])).vector[12], 0.0)  # 2.5 away: past its hearing × the mover's size (2 × 1.25)

    def test_hearing_gives_no_size(self):
        me = self.eye(10, 10, width=90.0, reach=3.0, hearing=2.0)
        small, big = creature(9, 10, size=0.6, id_=2), creature(9, 10, size=2.5, id_=3)
        small.moved = big.moved = True
        heard_small = sense(me, world([me, small])).vector
        heard_big = sense(me, world([me, big])).vector
        self.assertEqual(heard_small.shape, (N_SENSES,))
        self.assertEqual(list(heard_small[12:16]), list(heard_big[12:16]))  # where and how far: nothing of its size
        self.assertEqual(heard_small[12], 1.0)

    def test_a_still_creature_is_not_heard(self):
        me = self.eye(10, 10, width=90.0, reach=3.0, hearing=2.0)
        still = creature(9, 10, id_=2)  # behind it, within hearing, but it did not move
        self.assertEqual(sense(me, world([me, still])).vector[12], 0.0)
        self.assertFalse(perceived([me], [still], world([me, still]))[0, 0])  # nor is it learned from by sound
        still.moved = True
        self.assertEqual(sense(me, world([me, still])).vector[12], 1.0)
        self.assertTrue(perceived([me], [still], world([me, still]))[0, 0])

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


class TestTheTree(unittest.TestCase):
    """The k-d tree only finds the near ones faster: what every creature senses is what measuring every pair gives."""

    def planet(self):
        import random
        from src import senses
        from src.settings import START
        from src.terrain import make_terrain
        from src.world import World
        from tests.helpers import SMALL_WORLD, populate
        w = World(SMALL_WORLD["planet_radius"], START, "brain", random.Random(5), np.random.default_rng(5),
                  random.Random(6), rules={**starting_rules(), **SMALL_WORLD})
        w.terrain = make_terrain(w.rules, w.rng, w.radius)
        populate(w, 40)
        for _ in range(5):  # food grows, they turn and move (half the planet is sea: on land, more of them eat)
            w.step()
        return w, senses

    def test_senses_same_with_and_without_the_tree(self):
        from unittest import mock
        w, senses = self.planet()
        self.assertGreater(len(w.creatures), 20)
        self.assertGreater(len(w.foods), 10)
        with_tree = senses.sense_all(w)
        seen_tree = perceived(w.creatures, w.creatures, w)
        with mock.patch.object(senses, "EVERY_PAIR", True):
            every = senses.sense_all(w)
            seen_every = perceived(w.creatures, w.creatures, w)
        for a, b in zip(with_tree, every):
            np.testing.assert_array_equal(a.vector, b.vector)
            self.assertIs(a.food, b.food)
            self.assertIs(a.other, b.other)
            self.assertIs(a.on_food, b.on_food)
            self.assertIs(a.heard, b.heard)
        np.testing.assert_array_equal(seen_tree, seen_every)
        self.assertTrue(any(x.other is not None for x in every))  # some do see each other: the test means something

class TestTheOthersGenes(unittest.TestCase):  # the owner, 2026-11-27: "they should see all the DNA of the other creature"
    def test_it_sees_the_other_ones_genes(self):
        from src.senses import GENES_AT, SEEN_GENES, genes_seen
        from tests.helpers import NO_FOOD, add, make_world
        w = make_world(NO_FOOD)
        me, other = add(w, 0.0, 0.0), add(w, 2.0, 0.0)  # it faces east, the other 2 ahead (it sees 3 far)
        other.diet, other.armour, other.water, other.neurons, other.layers = 1.0, 0.5, 0.25, 32, 2
        genes = sense(me, w).vector[GENES_AT:GENES_AT + len(SEEN_GENES)]
        self.assertEqual(len(SEEN_GENES), 18)
        np.testing.assert_allclose(genes, genes_seen(other, w))
        self.assertEqual(genes[SEEN_GENES.index("diet")], 1.0)
        self.assertEqual(genes[SEEN_GENES.index("armour")], 0.5)
        self.assertTrue(np.all((genes >= 0.0) & (genes <= 1.0)))
        other.p = at(50.0, 0.0)  # out of sight: nothing
        self.assertFalse(sense(me, w).vector[GENES_AT:GENES_AT + 18].any())

    def test_it_knows_who_hit_it_and_how_far(self):
        from src.senses import HITTER_AT, genes_seen
        from tests.helpers import NO_FOOD_ATTACKS, add, make_world
        w = make_world(NO_FOOD_ATTACKS)
        hunter, prey = add(w, 0.0, 0.0), add(w, 1.0, 0.0, energy=40.0)
        hunter.diet = 0.8
        hunter.apply_action(("ATTACK", prey), w)
        felt = sense(prey, w).vector
        self.assertEqual(felt[16], 1.0)  # it was hit
        self.assertAlmostEqual(felt[HITTER_AT], 1.0 / prey.vision_reach)  # 1 away, of its vision reach
        np.testing.assert_allclose(felt[HITTER_AT + 1:HITTER_AT + 19], genes_seen(hunter, w))
        prey.hurt = None  # a step later: gone
        self.assertFalse(sense(prey, w).vector[HITTER_AT:HITTER_AT + 19].any())


if __name__ == "__main__":
    unittest.main()
