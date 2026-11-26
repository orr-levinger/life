"""Bodies and food on the planet: moving, reaching and dying are measured along its surface."""
import math
import unittest
from dataclasses import replace

import numpy as np

from src.food import Food
from src.sphere import arc
from tests.helpers import NO_FOOD, NO_FOOD_ATTACKS, add, at, make_world


def quiet(world):
    """Nothing costs anything but moving: only what a move takes is left."""
    world.settings = replace(world.settings, step_cost=0.0)
    world.rules.update({"vision_cost": 0.0, "hearing_cost": 0.0, "neuron_cost": 0.0})
    return world


class TestOnSphere(unittest.TestCase):
    def test_a_creature_walks_straight_around_the_planet(self):
        w = quiet(make_world(replace(NO_FOOD, move_cost=0.0)))
        w.rules["planet_radius"] = 37.0
        c = add(w, 0.0, 0.0, energy=50.0)
        start, way = c.p.copy(), 2 * math.pi * 37.0
        step = c.velocity
        for _ in range(int(way / step)):
            c.apply_action(("MOVE", (step, 0.0)), w)
        c.apply_action(("MOVE", (way - int(way / step) * step, 0.0)), w)
        self.assertLess(float(arc(start, c.p, 37.0)), 0.01)

    def test_moving_costs_the_arc_moved(self):
        w = quiet(make_world(NO_FOOD))
        c = add(w, 0.0, 0.0, energy=50.0)
        start = c.p.copy()
        c.apply_action(("MOVE", (0.2, 0.0)), w)
        self.assertAlmostEqual(float(arc(start, c.p, w.radius)), 0.2)
        self.assertAlmostEqual(50.0 - c.energy, 0.2 * NO_FOOD.move_cost * c.size)
        self.assertTrue(c.moved)

    def test_reach_is_measured_along_the_ground(self):
        w = make_world(NO_FOOD)
        c = add(w, 0.0, 0.0)
        for share, reaches in ((0.9, True), (1.1, False)):
            food = Food(at(0.0, 0.0), remaining_duration=-1, energy=10.0)
            reach = (c.radius + food.radius) * w.rules["reach"]
            food.p = at(share * reach, 0.0)
            self.assertEqual(c.can_reach(food), reaches, share)

    def test_a_corpse_lies_where_the_prey_died(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter, prey = add(w, 0.0, 0.0), add(w, 1.0, 0.0, energy=0.5)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertEqual(len(w.foods), 1)
        np.testing.assert_allclose(w.foods[0].p, prey.p)

    def test_its_pain_says_where_the_hitter_stands(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter, prey = add(w, 0.0, 0.0), add(w, 1.0, 0.0, energy=40.0)  # prey faces east: the hunter is behind it
        hunter.apply_action(("ATTACK", prey), w)
        share, ahead, left = prey.hurt
        self.assertGreater(share, 0.0)
        self.assertAlmostEqual(ahead, -1.0, places=3)
        self.assertAlmostEqual(left, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
