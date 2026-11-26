"""Bodies and food on the planet: moving, reaching and dying are measured along its surface."""
import math
import unittest
import unittest.mock
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



class TestWorldOnSphere(unittest.TestCase):
    def test_a_baby_is_born_near_its_parent(self):
        w = make_world(replace(NO_FOOD, lifespan=(1000.0, 1000.0)))
        parent = add(w, 0.0, 0.0, energy=95.0)
        w._births()
        babies = [c for c in w.creatures if c.parent_id == parent.id]
        self.assertEqual(len(babies), 1)
        self.assertLessEqual(float(arc(parent.p, babies[0].p, w.radius)), 1.0 * math.sqrt(2) + 1e-9)

    def test_food_spreads_evenly_over_the_planet(self):
        w = make_world(replace(NO_FOOD, food_per_step=20.0, max_food=10 ** 7))
        w.rules["planet_radius"] = 37.0
        with unittest.mock.patch.object(type(w), "_fits", return_value=True):
            for _ in range(100):
                w.spawn_food()
        pts = np.stack([f.p for f in w.foods])
        self.assertEqual(len(pts), 2000)
        octants = np.bincount((pts > 0).astype(int) @ np.array([4, 2, 1]), minlength=8)
        self.assertTrue(np.all(np.abs(octants - 250) < 0.25 * 250), octants)

    def test_newcomers_land_anywhere_on_the_ball(self):
        w = make_world(seed=3)
        w.rules["planet_radius"] = 37.0
        pts = np.stack([w.add_newcomer().p for _ in range(400)])
        self.assertAlmostEqual(float(np.abs(np.linalg.norm(pts, axis=1) - 1).max()), 0.0)
        self.assertEqual(int((np.bincount((pts > 0).astype(int) @ np.array([4, 2, 1]), minlength=8) > 0).sum()), 8)


if __name__ == "__main__":
    unittest.main()
