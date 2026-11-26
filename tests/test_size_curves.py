"""Size by curves (spec decision 1): a bigger body holds more than its size in energy and costs less than its size to
keep, but is slower, lives longer and is easier to notice."""
import math
import unittest
from dataclasses import replace

import numpy as np

from src.senses import perceived, sense
from src.settings import START
from tests.helpers import at, face, NO_FOOD, add, make_world

SIZES = (0.5, 1.0, 2.0)


def quiet(world):
    """No price for senses, neurons or a missed bite: only the body's own living cost is left."""
    world.settings = replace(world.settings, vision_cost=0.0, hearing_cost=0.0, neuron_cost=0.0, eat_miss=0.0)
    return world


class TestSizeCurves(unittest.TestCase):
    def test_the_tank_grows_faster_than_the_size(self):
        w = make_world()
        for size in SIZES:
            c = add(w, 10, 10, size=size)
            self.assertAlmostEqual(c.max_energy, w.rules["tank"] * size ** w.rules["tank_power"])
        self.assertEqual(w.rules["tank_power"], 1.3)

    def test_living_costs_less_than_the_size(self):
        w = quiet(make_world(NO_FOOD))
        for size in SIZES:
            c = add(w, 10, 10, size=size, energy=20.0)  # within even the smallest tank
            c.apply_action(("EAT", None), w)
            self.assertAlmostEqual(20.0 - c.energy, NO_FOOD.step_cost * size ** w.rules["living_power"])
        self.assertEqual(w.rules["living_power"], 0.75)

    def test_a_bigger_body_cruises_slower(self):
        w = make_world()
        for size in SIZES:
            c = add(w, 10, 10, size=size)
            self.assertAlmostEqual(c.velocity, w.rules["cruise_speed"] / size ** w.rules["speed_power"])
        self.assertEqual(w.rules["speed_power"], 0.5)

    def test_a_bigger_body_lives_longer(self):
        for size in SIZES:
            w = make_world(replace(START, lifespan=(1000.0, 1000.0)), seed=1)
            w.rules["gene.size.new_low"] = w.rules["gene.size.new_high"] = size
            c = w.add_newcomer()
            self.assertEqual(c.lifespan, round(1000.0 * size ** w.rules["lifespan_power"]))

    def test_a_small_mover_is_heard_only_from_close(self):
        w = make_world()
        listener = add(w, 10.0, 10.0)
        listener.hearing, listener.vision_reach = 4.0, 0.1  # it hears, it hardly sees
        mover = add(w, 12.5, 10.0, size=0.5)  # 2.5 away: within 4, beyond 4 × 0.5
        mover.moved = True
        self.assertIsNone(sense(listener, w).heard)
        mover.p = at(11.5, 10.0)  # 1.5 away: within 4 × 0.5 = 2
        self.assertIsNotNone(sense(listener, w).heard)

    def test_a_small_body_is_seen_only_from_closer(self):
        w = make_world()
        looker = add(w, 10.0, 10.0)
        looker.vision_width, looker.vision_reach, looker.hearing = 120.0, 4.0, 0.0
        face(looker, 1.0, 0.0)
        small = add(w, 13.2, 10.0, size=0.5)  # 3.2 away: within 4, beyond 4 × √0.5 ≈ 2.83
        self.assertFalse(perceived([looker], [small], w)[0, 0])
        small.p = at(12.5, 10.0)  # 2.5 away
        self.assertTrue(perceived([looker], [small], w)[0, 0])
        self.assertAlmostEqual(w.rules["seen_power"], 0.5)
        self.assertAlmostEqual(w.rules["heard_power"], 1.0)


if __name__ == "__main__":
    unittest.main()
