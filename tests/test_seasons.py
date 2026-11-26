"""Seasons (spec decision 9): food and fruit per step rise and fall over a year of steps."""
import math
import unittest
from dataclasses import replace
from unittest import mock

from src.settings import START
from tests.helpers import make_world


def grown(world, step, calls=400):
    """How much food grows a step at `step` of the year (every spot fits)."""
    world.step_count = step
    world.foods = []
    with mock.patch.object(type(world), "_fits", return_value=True):
        for _ in range(calls):
            world.spawn_food()
    return len(world.foods) / calls


class TestSeasons(unittest.TestCase):
    def test_food_follows_the_year(self):
        w = make_world(replace(START, food_per_step=4.0, max_food=10 ** 7, fruit_rate=0.0))
        year, depth = w.rules["year_steps"], w.rules["season_depth"]
        self.assertEqual((year, depth), (20000.0, 0.5))
        for share, expected in ((0.0, 4.0), (0.25, 6.0), (0.5, 4.0), (0.75, 2.0)):
            self.assertAlmostEqual(w.season(), 1.0 + depth * math.sin(2 * math.pi * w.step_count / year))
            self.assertAlmostEqual(grown(w, round(share * year)), expected, delta=0.25, msg=share)

    def test_fruit_follows_the_year_too(self):
        from src.terrain import Terrain
        from tests.helpers import tree
        w = make_world(replace(START, food_per_step=0.0, max_food=10 ** 7, fruit_rate=0.4))
        w.terrain = Terrain([], [tree(25.0, 15.0, 1.5)], w.radius)
        self.assertAlmostEqual(grown(w, 5000, 2000), 0.4 * 1.5, delta=0.05)
        self.assertAlmostEqual(grown(w, 15000, 2000), 0.4 * 0.5, delta=0.05)

    def test_an_exam_has_no_seasons(self):
        w = make_world(replace(START, food_per_step=4.0, max_food=10 ** 7), evolving=False)
        w.step_count = 5000
        self.assertEqual(w.season(), 1.0)

    def test_the_page_gets_the_season(self):
        from src.evolve import world_now
        w = make_world()
        w.step_count = 5000
        self.assertAlmostEqual(world_now(w)["season"], 1.5)


if __name__ == "__main__":
    unittest.main()
