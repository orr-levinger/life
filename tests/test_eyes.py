"""Eyes of pixels, ears of directions (the owner, 2026-11-27: "some animals have very high resolution sight, some have
very low resolution")."""
import unittest

import numpy as np

from src.rules import RULES, starting_rules
from src.world import GENES
from tests.helpers import add, make_world


class TestSharpness(unittest.TestCase):
    def test_sharpness_is_a_gene(self):
        by_name = {r.name: r for r in RULES}
        self.assertEqual([by_name[f"gene.eye_pixels.{end}"].start for end in ("low", "high", "new_low", "new_high")],
                         [1.0, 16.0, 2.0, 8.0])
        self.assertIn("eye_pixels", GENES)  # a baby may change it
        w = make_world(seed=4)
        drawn = {w.add_newcomer().eye_pixels for _ in range(60)}
        self.assertTrue(drawn <= set(range(2, 9)) and len(drawn) > 3)  # whole numbers, 2 to 8, of many kinds

    def test_sharpness_costs_energy(self):
        self.assertAlmostEqual(starting_rules()["pixel_cost"], 0.00006)  # a tenth of nature's 0.0006
        paid = {}
        for pixels in (2, 10):
            w = make_world()
            w.rules.update(step_cost=0.0, vision_cost=0.0, hearing_cost=0.0, neuron_cost=0.0, pixel_cost=0.001)
            c = add(w, 0.0, 0.0, energy=50.0)
            c.eye_pixels = pixels
            c.apply_action(("MOVE", (0.0, 0.0)), w)  # standing still: only what it is costs
            paid[pixels] = 50.0 - c.energy
        self.assertAlmostEqual(paid[10] - paid[2], 8 * 0.001)


def eye_of(vector):
    """[pixel, channel]: creature closeness, size, hunter, alike, food closeness, rock or water closeness."""
    from src.senses import EYE_AT, EYE_CHANNELS, EYE_PIXELS
    return vector[EYE_AT:EYE_AT + EYE_PIXELS * EYE_CHANNELS].reshape(EYE_PIXELS, EYE_CHANNELS)


def looker(w, pixels=16):
    """A creature at the map's (0, 0) facing east, seeing 120 degrees out to 6."""
    me = add(w, 0.0, 0.0)
    me.vision_width, me.vision_reach, me.hearing, me.eye_pixels = 120.0, 6.0, 4.0, pixels
    return me


class TestPixels(unittest.TestCase):
    def test_kin_food_and_a_hunter_land_in_their_pixels(self):
        from src.food import Food
        from src.senses import sense
        from tests.helpers import NO_FOOD, at
        w = make_world(NO_FOOD)
        me = looker(w)
        kin = add(w, 3.0, 1.5)  # ahead-left: 26.6 degrees left, pixel 4 of 16 (left edge 0)
        kin.vision_width, kin.vision_reach, kin.hearing = me.vision_width, me.vision_reach, me.hearing  # its genes alike
        hunter = add(w, 3.0, -1.5)  # ahead-right: pixel 11
        hunter.meat_gut, hunter.size = 1.0, 1.6
        w.foods.append(Food(at(2.0, 0.0), remaining_duration=-1, energy=10.0))  # straight ahead: pixel 7 or 8
        eye = eye_of(sense(me, w).vector)
        self.assertGreater(eye[4, 0], 0.3)  # a creature, close
        self.assertAlmostEqual(eye[4, 3], 1.0, places=6)  # alike: kin
        self.assertEqual(eye[11, 2], 1.0)  # a hunter
        self.assertGreater(eye[11, 1], 0.5)  # bigger than it
        self.assertAlmostEqual(max(eye[7, 4], eye[8, 4]), 1.0 - 2.0 / 6.0, places=3)  # food, 2 of its 6 away (ahead:
        # the line between the two middle pixels)
        self.assertEqual(eye[0, 0], 0.0)  # nothing at its left edge

    def test_a_blurry_eye_sees_blocks(self):
        from src.senses import sense
        from tests.helpers import NO_FOOD
        w = make_world(NO_FOOD)
        me = looker(w, pixels=4)
        add(w, 3.0, 1.5)  # pixel 4 of 16: the second block of 4
        eye = eye_of(sense(me, w).vector)
        for block in range(4):
            np.testing.assert_array_equal(eye[block * 4:block * 4 + 4], np.repeat(eye[block * 4:block * 4 + 1], 4, axis=0))
        self.assertTrue(eye[4:8, 0].all())  # the whole block sees it
        self.assertFalse(eye[0:4, 0].any())

    def test_a_rock_hides_from_the_pixels(self):
        from src.senses import sense
        from src.terrain import Obstacle, Terrain
        from tests.helpers import NO_FOOD, cap
        w = make_world(NO_FOOD)
        w.terrain = Terrain([Obstacle("rock", (cap(2.0, 0.0, 0.5),))], [], w.radius)
        me = looker(w)
        add(w, 4.0, 0.0)  # behind the rock, straight ahead
        eye = eye_of(sense(me, w).vector)
        self.assertFalse(eye[:, 0].any())  # unseen
        self.assertAlmostEqual(max(eye[7, 5], eye[8, 5]), 1.0 - 1.5 / 6.0, places=2)  # the rock, its edge 1.5 away

    def test_a_mover_is_heard_from_its_direction(self):
        from src.senses import EARS_AT, sense
        from tests.helpers import NO_FOOD
        w = make_world(NO_FOOD)
        me = looker(w)
        behind_left = add(w, -1.5, 1.5)
        behind_left.moved = True
        ears = sense(me, w).vector[EARS_AT:EARS_AT + 16].reshape(8, 2)  # ahead, ahead-left, left, behind-left, ...
        self.assertGreater(ears[3, 0], 0.0)  # footsteps behind-left
        self.assertEqual(np.count_nonzero(ears[:, 0]), 1)


class TestBrainsOfBefore(unittest.TestCase):  # 84 senses before the eye of pixels
    def test_brains_of_before_act_the_same(self):
        from src.brain import N_SENSES, Brain
        rng = np.random.default_rng(5)
        w1, b1 = rng.normal(size=(84, Brain.N_HIDDEN)), rng.normal(size=Brain.N_HIDDEN)
        w2, b2 = rng.normal(size=(Brain.N_HIDDEN, Brain.N_OUT)), rng.normal(size=Brain.N_OUT)
        x = rng.normal(size=N_SENSES)
        np.testing.assert_allclose(Brain(w1, b1, w2, b2).forward(x), np.tanh(np.tanh(x[:84] @ w1 + b1) @ w2 + b2))


if __name__ == "__main__":
    unittest.main()
