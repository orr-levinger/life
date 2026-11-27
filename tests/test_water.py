"""Water (spec decision 6): a water gene lets a creature enter ponds, swim fast and walk slower; food grows in ponds too;
ponds no longer hide what is behind them (rocks still do)."""
import unittest
from dataclasses import replace

from src.food import Food
from src.senses import IN_WATER, WATER_SEEN, sense
from src.terrain import Obstacle, Terrain
import numpy as np

import random

from src.rules import starting_rules
from src.sphere import arc
from src.water import Water
from src.world import World
from tests.helpers import EAST, NO_FOOD, SMALL_WORLD, add, at, cap, face, make_world

SEA_R = 37.0
SEA = Water(SEA_R, [[*EAST, 1.0, 0.0]], 0.0)  # water west of the map's centre: the shore is the map's line x = 0


def pond_world():
    """A pond around (20, 10) of radius 3, a rock around (20, 20) of radius 3."""
    w = make_world(NO_FOOD)
    w.terrain = Terrain([Obstacle("pond", (cap(20.0, 10.0, 3.0),)), Obstacle("rock", (cap(20.0, 20.0, 3.0),))], [], w.radius)
    return w


def sea_world(settings=NO_FOOD, rocks=()):
    """A planet of radius 37 whose west half (map x < 0) is sea."""
    w = World(SEA_R, settings, "random", random.Random(0), np.random.default_rng(0), random.Random(1000),
              rules={**starting_rules(), **SMALL_WORLD, "planet_radius": SEA_R})
    w.terrain = Terrain(list(rocks), [], SEA_R, water=SEA)
    return w


def moved(w, c, dx):
    """How far a move of dx ahead took it, along the ground."""
    start = c.p.copy()
    c.apply_action(("MOVE", (dx, 0.0)), w)
    return float(arc(start, c.p, w.radius))


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
        in_pond = sum(w.terrain.in_pond(f.p) for f in w.foods)
        self.assertGreater(in_pond, 0)
        self.assertFalse(any(w.terrain.blocked(f.p, 0.0, swims=True) for f in w.foods))  # never on a rock

    def test_a_pond_does_not_hide_food_but_a_rock_does(self):
        w = pond_world()
        looker = add(w, 15.0, 10.0)
        looker.vision_width, looker.vision_reach = 120.0, 12.0
        w.foods.append(Food(at(25.0, 10.0), remaining_duration=-1, energy=10.0))  # beyond the pond
        self.assertIsNotNone(sense(looker, w).food)
        looker.p, w.foods[0].p = at(15.0, 20.0), at(25.0, 20.0)  # beyond the rock
        face(looker, 1.0, 0.0)
        self.assertIsNone(sense(looker, w).food)

    def test_it_senses_being_in_water_and_water_ahead(self):
        w = pond_world()
        c = add(w, 15.0, 10.0)
        c.vision_width, c.vision_reach = 120.0, 6.0
        v = sense(c, w).vector
        self.assertEqual((v[IN_WATER], v[WATER_SEEN]), (0.0, 1.0))  # on land, a pond ahead
        c.p = at(20.0, 10.0)
        self.assertEqual(sense(c, w).vector[IN_WATER], 1.0)
        c.p = at(15.0, 20.0)
        face(c, 1.0, 0.0)
        self.assertEqual(sense(c, w).vector[WATER_SEEN], 0.0)  # a rock ahead

    def test_a_walker_never_enters_the_sea(self):
        for water in (0.0, 1.0):
            w = sea_world()
            c = add(w, 1.5, 0.0, energy=1000.0)
            c.water = water
            for _ in range(40):  # it walks west, into the sea
                face(c, -1.0, 0.0)
                c.apply_action(("MOVE", (1.0, 0.0)), w)
            if water:
                self.assertTrue(w.terrain.in_pond(c.p))  # a swimmer goes in
            else:
                self.assertFalse(w.terrain.blocked(c.p, c.radius))  # a walker stays on land, its body too
                self.assertLess(SEA.distance_at(c.p), c.radius + 0.6)  # at the shore
                self.assertTrue(c.bumped)

    def test_in_water_and_water_seen_on_a_planet(self):
        w = sea_world()
        c = add(w, 3.0, 0.0)
        c.vision_width, c.vision_reach = 120.0, 6.0
        face(c, -1.0, 0.0)
        v = sense(c, w).vector
        self.assertEqual((v[IN_WATER], v[WATER_SEEN], v[20]), (0.0, 1.0, 1.0))  # on land, the sea ahead
        self.assertAlmostEqual(v[21], 3.0 / 6.0, delta=0.1)  # 3 away, of its reach 6
        self.assertAlmostEqual(v[22], 1.0, delta=0.05)  # straight ahead
        face(c, 1.0, 0.0)
        v = sense(c, w).vector
        self.assertEqual((v[WATER_SEEN], v[20]), (0.0, 0.0))  # looking away: nothing in view
        c.p = at(-3.0, 0.0, SEA_R)
        face(c, 1.0, 0.0)
        v = sense(c, w).vector
        self.assertEqual((v[IN_WATER], v[WATER_SEEN], v[21]), (1.0, 1.0, 0.0))  # in the sea
        w = sea_world(rocks=[Obstacle("rock", (cap(1.5, 0.0, 0.5, SEA_R),))])
        c = add(w, 3.0, 0.0)
        c.vision_width, c.vision_reach = 120.0, 6.0
        face(c, -1.0, 0.0)
        v = sense(c, w).vector
        self.assertEqual((v[WATER_SEEN], v[20]), (0.0, 1.0))  # a rock before the sea: the rock is what it sees
        self.assertAlmostEqual(v[21], 1.0 / 6.0, delta=0.02)

    def test_food_grows_in_the_sea(self):
        w = sea_world(replace(NO_FOOD, food_per_step=5.0, max_food=10 ** 6, fruit_rate=0.0))
        w.rules.update({"pond_food_share": 0.3, "pond_edge_food": 1.0, "season_depth": 0.0})
        self.assertTrue(w.terrain.has_water())
        for _ in range(300):
            w.spawn_food()
        wet = np.mean([w.terrain.in_pond(f.p) for f in w.foods])
        self.assertGreater(len(w.foods), 1000)
        self.assertAlmostEqual(float(wet), 0.3, delta=0.05)  # the pond share in the sea, the rest on land

    def test_old_ponds_still_water(self):
        t = Terrain([Obstacle("pond", (cap(0.0, 0.0, 3.0),)), Obstacle("rock", (cap(10.0, 0.0, 2.0),))], [], 1.0e6)
        self.assertTrue(t.has_water())
        self.assertFalse(Terrain([Obstacle("rock", (cap(10.0, 0.0, 2.0),))], [], 1.0e6).has_water())
        self.assertEqual((t.in_pond(at(0.0, 0.0)), t.in_pond(at(5.0, 0.0))), (True, False))
        self.assertEqual((t.blocked(at(3.5, 0.0), 1.0), t.blocked(at(3.5, 0.0), 1.0, swims=True)), (True, False))
        self.assertTrue(t.blocked(at(10.0, 0.0), 0.5, swims=True))  # a rock stops a swimmer too
        self.assertEqual([t.near_pond(at(x, 0.0), 2.0) for x in (0.0, 4.0, 6.5)], [False, True, False])
        spot = t.pond_spot(random.Random(1))
        self.assertLess(float(arc(spot, at(0.0, 0.0), 1.0e6)), 3.0)

    def test_a_swimmer_pays_less_in_water_and_more_on_land(self):  # moving and living, by its water gene
        def paid(water, x, y):
            w = pond_world()
            w.rules.update(move_cost=1.0, step_cost=0.1, sprint_power=0.0, vision_cost=0.0,
                           hearing_cost=0.0, neuron_cost=0.0, stealth_cost=0.0, armour_cost=0.0)
            c = add(w, x, y, energy=50.0)
            c.water = water
            c.apply_action(("MOVE", (0.1, 0.0)), w)  # a short move and a step alive, staying where it is (pond or land)
            self.assertAlmostEqual(float(arc(c.p, at(x + 0.1, y), w.radius)), 0.0, places=6)
            return 50.0 - c.energy

        land = paid(0.0, 5.0, 5.0)  # a land creature on land: as before
        self.assertGreater(land, 0.1)
        for water, x, y, share in ((1.0, 20.0, 10.0, 0.5), (1.0, 5.0, 5.0, 1.5),
                                   (0.5, 20.0, 10.0, 0.75), (0.5, 5.0, 5.0, 1.25)):
            self.assertAlmostEqual(paid(water, x, y), land * share, places=9, msg=(water, x, y))

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
