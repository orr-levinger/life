"""Water (spec decision 6): a water gene lets a creature enter ponds, swim fast and walk slower; food grows in ponds too;
ponds no longer hide what is behind them (rocks still do)."""
import unittest
from dataclasses import replace

from src.food import Food
from src.senses import IN_WATER, WATER_SEEN, sense
from src.terrain import Obstacle, Terrain
from tests.helpers import at, NO_FOOD, add, make_world


def pond_world():
    """A pond around (20, 10) of radius 3, a rock around (20, 20) of radius 3."""
    w = make_world(NO_FOOD)
    w.terrain = Terrain([Obstacle("pond", ((20.0, 10.0, 3.0),)), Obstacle("rock", ((20.0, 20.0, 3.0),))], [])
    return w


def moved(w, c, dx):
    x = c.x
    c.apply_action(("MOVE", (dx, 0.0)), w)
    return c.x - x


class TestWater(unittest.TestCase):
    def test_without_water_a_pond_is_a_wall(self):
        w = pond_world()
        c = add(w, 16.5, 10, size=0.5)
        c.water = 0.0
        self.assertEqual(moved(w, c, 0.5), 0.0)
        self.assertTrue(c.bumped)

    def test_a_swimmer_swims_fast_and_walks_slowly(self):
        w = pond_world()
        swimmer, walker = add(w, 20, 10), add(w, 5, 5)
        swimmer.water = walker.water = 1.0
        top = swimmer.top_speed
        self.assertAlmostEqual(moved(w, swimmer, 5.0), top * 1.0)  # in water: × its water
        self.assertAlmostEqual(moved(w, walker, 5.0), top * (1 - 1.0 * w.rules["land_penalty"]))  # on land: × 0.3

    def test_half_a_swimmer_enters_a_pond(self):
        w = pond_world()
        c = add(w, 16.5, 10, size=0.5)
        c.water = 0.5
        self.assertGreater(moved(w, c, 0.4), 0.0)

    def test_food_grows_in_ponds_too(self):
        w = pond_world()
        w.settings = replace(w.settings, food_per_step=5.0, max_food=10 ** 6)
        for _ in range(400):
            w.spawn_food()
        in_pond = sum(w.terrain.in_pond(f.x, f.y) for f in w.foods)
        self.assertGreater(in_pond, 0)
        self.assertFalse(any(w.terrain.blocked(f.x, f.y, 0.0, swims=True) for f in w.foods))  # never on a rock

    def test_a_pond_does_not_hide_food_but_a_rock_does(self):
        w = pond_world()
        looker = add(w, 15.0, 10.0)
        looker.vision_width, looker.vision_reach, looker.facing = 120.0, 12.0, (1.0, 0.0)
        w.foods.append(Food(at(25.0, 10.0), remaining_duration=-1, energy=10.0))  # beyond the pond
        self.assertIsNotNone(sense(looker, w).food)
        looker.y, w.foods[0].y = 20.0, 20.0  # beyond the rock
        self.assertIsNone(sense(looker, w).food)

    def test_it_senses_being_in_water_and_water_ahead(self):
        w = pond_world()
        c = add(w, 15.0, 10.0)
        c.vision_width, c.vision_reach, c.facing = 120.0, 6.0, (1.0, 0.0)
        v = sense(c, w).vector
        self.assertEqual((v[IN_WATER], v[WATER_SEEN]), (0.0, 1.0))  # on land, a pond ahead
        c.x = 20.0
        self.assertEqual(sense(c, w).vector[IN_WATER], 1.0)
        c.x, c.y = 15.0, 20.0
        self.assertEqual(sense(c, w).vector[WATER_SEEN], 0.0)  # a rock ahead

    def test_newcomers_draw_their_water(self):
        w = make_world(seed=6)
        waters = [w.add_newcomer().water for _ in range(30)]
        self.assertTrue(all(0.0 <= x <= 1.0 for x in waters))
        self.assertGreater(max(waters) - min(waters), 0.5)


    def test_the_page_gets_each_ones_water(self):
        from src.evolve import world_now, world_view
        w = pond_world()
        add(w, 5, 5).water = 0.6
        self.assertEqual(world_view(w)["creatures"][0]["water"], 0.6)
        self.assertEqual(world_now(w)["genes"]["water"], [0.6, 0.6, 0.6])


if __name__ == "__main__":
    unittest.main()
