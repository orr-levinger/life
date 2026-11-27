import math
import random
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from src.food import Food
from src.rules import starting_rules
from src.senses import N_SENSES, sense
from src.sphere import arc, exp_map, local_frame, random_point
from src.terrain import Obstacle, Terrain, Tree, make_terrain
from tests.helpers import NO_FOOD, add, at, cap, face, make_world, tree

ROCK = Obstacle("rock", (cap(20.0, 10.0, 2.0),))  # one round rock: x 18-22 at y 10 on the map


def walled(world, obstacles=(ROCK,), trees=()):
    world.terrain = Terrain(list(obstacles), list(trees), world.radius)
    return world


def map_distance(p, x, y, R):
    return float(arc(np.asarray(p), at(x, y, R), R))


class TestMakingTerrain(unittest.TestCase):
    def test_a_world_gets_its_obstacles_and_trees_none_on_another(self):
        rules = {**starting_rules(), "obstacles": 5.0, "trees": 6.0, "forests": 0.0, "lone_trees": 0.0}
        clear = at(10.0, 10.0, 37.0)
        terrain = make_terrain(rules, random.Random(3), 37.0, keep_clear=[(clear, 1.0)])
        self.assertEqual((len(terrain.obstacles), len(terrain.trees)), (5, 6))
        self.assertFalse(terrain.blocked(clear, 1.0, swims=True))  # the spot kept clear of rocks (water may lie there)
        self.assertTrue(all(not terrain.blocked(np.array(t.p), 0.0) for t in terrain.trees))
        sizes = {round(max(c[1] for c in o.caps), 2) for o in terrain.obstacles}
        self.assertGreater(len(sizes), 1)  # of random sizes

    def test_a_new_planet_has_its_forests(self):
        rules = {**starting_rules(), "forests": 4.0, "lone_trees": 0.0}  # its forest trees only
        terrain = make_terrain(rules, random.Random(4), 37.0)
        self.assertEqual(len(terrain.forest_centres), 4)
        self.assertGreater(len(terrain.trees), 0.8 * rules["trees"])  # nearly all of them fit
        for t in terrain.trees:
            self.assertLessEqual(min(float(arc(np.array(t.p), np.array(c), 37.0)) for c in terrain.forest_centres),
                                 rules["forest_spread"] + 1e-6)

    def test_lone_trees_grow_all_over_the_land(self):  # the owner, 2026-11-27: "some trees all around just not in water"
        rules = {**starting_rules(), "forests": 2.0, "trees": 20.0, "lone_trees": 40.0}
        terrain = make_terrain(rules, random.Random(6), 37.0)
        away = [t for t in terrain.trees
                if min(float(arc(np.array(t.p), np.array(c), 37.0)) for c in terrain.forest_centres) > rules["forest_spread"]]
        self.assertGreaterEqual(len(away), 30)  # lone trees, away from every forest
        for t in terrain.trees:
            self.assertFalse(terrain.in_pond(np.array(t.p)))  # never in water
            self.assertFalse(terrain.blocked(np.array(t.p), 0.0, swims=True))  # never on a rock

    def test_pond_edges_grow_more_food_than_open_ground(self):
        w = make_world(replace(NO_FOOD, max_food=10 ** 7, food_per_step=6.0, fruit_rate=0.0))
        w.rules.update({"planet_radius": 12.0, "pond_food_share": 0.0, "pond_edge_food": 2.0, "pond_edge_width": 2.0,
                        "season_depth": 0.0})
        centre = at(0.0, 0.0, 12.0)
        w.terrain = Terrain([Obstacle("pond", ((tuple(centre), 4.0),))], [], 12.0)
        with mock.patch.object(type(w), "_fits", lambda self, food, wet=False: not self.terrain.blocked(food.p, 0.0)):
            for _ in range(1000):
                w.spawn_food()
        d = np.array([float(arc(centre, f.p, 12.0)) for f in w.foods])
        edge = int(np.sum((d >= 4.0) & (d <= 6.0)))
        cap_area = lambda r: 2 * math.pi * 12.0 ** 2 * (1 - math.cos(r / 12.0))  # noqa: E731
        edge_area = cap_area(6.0) - cap_area(4.0)
        open_area = 4 * math.pi * 12.0 ** 2 - cap_area(6.0)
        self.assertGreater(edge / edge_area, 1.6 * (len(w.foods) - edge) / open_area)  # about twice as dense

    def test_the_terrain_round_trips(self):
        terrain = make_terrain(starting_rules(), random.Random(1), 37.0)
        back = Terrain.from_state(terrain.state())
        self.assertEqual((back.obstacles, back.trees, back.radius), (terrain.obstacles, terrain.trees, terrain.radius))


class TestObstacles(unittest.TestCase):
    def test_food_never_grows_on_a_rock(self):
        w = walled(make_world(replace(NO_FOOD, max_food=400, food_per_step=6.0)),
                   [Obstacle("rock", (cap(0.0, 0.0, 10.0),))])  # (food grows in ponds: tests/test_water.py)
        for _ in range(60):
            w.spawn_food()
        self.assertTrue(w.foods)
        self.assertTrue(all(map_distance(f.p, 0.0, 0.0, w.radius) >= 10.0 + f.radius for f in w.foods))

    def test_moving_into_an_obstacle_is_refused_and_costs_energy(self):
        w = walled(make_world())
        c = add(w, 17.5, 10.0, size=1.0)
        start, before = c.p.copy(), c.energy
        c.apply_action(("MOVE", (0.3, 0.0)), w)  # its body would touch the rock
        np.testing.assert_array_equal(c.p, start)
        self.assertFalse(c.moved)
        self.assertTrue(c.bumped)
        free = add(w, 5.0, 5.0, size=1.0)
        free_before = free.energy
        free.apply_action(("MOVE", (0.0, 0.0)), w)
        self.assertAlmostEqual(before - c.energy, (free_before - free.energy) + w.rules["bump_cost"])

    def test_an_obstacle_hides_what_is_behind_it(self):
        w = walled(make_world())
        me = add(w, 15.0, 10.0, size=1.0)
        me.vision_width, me.vision_reach = 120.0, 15.0
        w.foods.append(Food(at(25.0, 10.0), remaining_duration=-1, energy=10.0))  # behind the rock
        self.assertIsNone(sense(me, w).food)
        w.foods.append(Food(at(17.0, 12.5), remaining_duration=-1, energy=10.0))  # beside it, in the open
        self.assertIsNotNone(sense(me, w).food)

    def test_a_rock_anywhere_hides_what_is_behind_it(self):
        rng = random.Random(9)
        for _ in range(20):
            centre = random_point(rng)
            e1, e2 = local_frame(centre)
            t = Terrain([Obstacle("rock", ((tuple(centre), 2.0),))], [], 37.0)
            a = exp_map(centre, e1, e2, -6.0, 0.3, 37.0)
            b = exp_map(centre, e1, e2, 6.0, -0.3, 37.0)
            aside = exp_map(centre, e1, e2, 6.0, 5.0, 37.0)
            self.assertTrue(t.sight_blocked(a, b)[0, 0])
            self.assertFalse(t.sight_blocked(a, aside)[0, 0])
        opposite = -at(0.0, 0.0, 37.0)  # the point opposite the map's centre
        e1, e2 = local_frame(opposite)
        t = Terrain([Obstacle("rock", ((tuple(opposite), 2.0),))], [], 37.0)
        self.assertTrue(t.sight_blocked(exp_map(opposite, e1, e2, 0.0, -5.0, 37.0), exp_map(opposite, e1, e2, 0.0, 5.0, 37.0))[0, 0])

    def test_checking_only_the_pairs_asked_gives_the_same_answer(self):
        rng = random.Random(5)
        terrain = make_terrain(starting_rules(), rng, 37.0)
        a = np.stack([random_point(rng) for _ in range(40)])
        b = np.stack([random_point(rng) for _ in range(70)])
        mask = np.random.default_rng(5).random((40, 70)) < 0.2
        full = terrain.sight_blocked(a, b)
        self.assertTrue(full.any() and (~full).any())  # some ways blocked, some open
        np.testing.assert_array_equal(terrain.sight_blocked(a, b, only=mask), full & mask)

    def test_an_obstacle_in_its_field_of_vision_is_sensed(self):
        w = walled(make_world())
        me = add(w, 15.0, 10.0, size=1.0)
        me.vision_width, me.vision_reach = 120.0, 5.0
        v = sense(me, w).vector
        self.assertEqual(len(v), N_SENSES)
        self.assertEqual(v[20], 1.0)  # seen
        self.assertAlmostEqual(v[21], 3.0 / 5.0, places=4)  # its edge 3 away, ÷ its vision reach
        self.assertAlmostEqual(v[22], 1.0, places=4)  # ahead
        face(me, -1.0, 0.0)  # looking away: not seen
        self.assertTrue(np.all(sense(me, w).vector[20:24] == 0.0))


class TestTrees(unittest.TestCase):
    def test_a_tree_drops_fruit_near_its_trunk_now_and_then(self):
        w = walled(make_world(replace(NO_FOOD, max_food=400, fruit_rate=1.0)), [], [tree(25.0, 15.0, 1.5)])
        for _ in range(40):  # a fruit a step, where one fits (not on another)
            w.spawn_food()
        fruit = [f for f in w.foods if f.fruit]
        self.assertGreater(len(fruit), 3)
        self.assertTrue(all(map_distance(f.p, 25.0, 15.0, w.radius) <= w.rules["fruit_spread"] + 1e-6 for f in fruit))
        none = walled(make_world(replace(NO_FOOD, max_food=400, fruit_rate=0.0)), [], [tree(25.0, 15.0, 1.5)])
        for _ in range(10):
            none.spawn_food()
        self.assertEqual(none.foods, [])

    def test_a_creature_under_a_tree_is_seen_only_from_close_by(self):
        w = walled(make_world(), [], [tree(25.0, 10.0, 1.5)])
        hidden = add(w, 25.0, 10.0, size=1.0)
        far = add(w, 20.0, 10.0, size=1.0)
        far.vision_width, far.vision_reach = 120.0, 10.0
        self.assertIsNone(sense(far, w).other)  # 5 away: it hides under the tree
        near = add(w, 25.0 - w.rules["hide_reach"] * 0.9, 10.0, size=1.0)
        near.vision_width, near.vision_reach = 120.0, 10.0
        self.assertIs(sense(near, w).other, hidden)
        self.assertEqual(sense(hidden, w).vector[24], 1.0)  # it knows it is under a tree
        self.assertEqual(sense(far, w).vector[24], 0.0)


class TestRockTree(unittest.TestCase):
    def test_sight_blocked_same_with_the_rock_tree(self):  # only the rocks near a sight line are tried: the same answer
        from src import terrain as module
        rng = random.Random(9)
        R = 20.0
        rocks = [Obstacle("rock", ((tuple(random_point(rng)), rng.uniform(0.3, 2.0)),)) for _ in range(60)]
        ponds = [Obstacle("pond", ((tuple(random_point(rng)), 1.5),))]
        ground = Terrain(rocks + ponds, [], R)
        lookers = np.array([random_point(rng) for _ in range(60)])
        things = np.array([random_point(rng) for _ in range(90)])
        only = arc(lookers[:, None, :], things[None, :, :], R) <= 18.0
        with_tree = ground.sight_blocked(lookers, things, only=only)
        with mock.patch.object(module, "EVERY_ROCK", True):
            every = ground.sight_blocked(lookers, things, only=only)
        np.testing.assert_array_equal(with_tree, every)
        self.assertTrue(every.any() and (only & ~every).any())  # some lines blocked, some clear

    def test_in_pond_same_with_the_tree(self):  # in a pond, or blocked: only the caps near the point are tried
        from src import terrain as module
        rng = random.Random(11)
        R = 20.0
        caps = [Obstacle(rng.choice(("rock", "pond")), ((tuple(random_point(rng)), rng.uniform(0.3, 2.5)),))
                for _ in range(80)]
        ground = Terrain(caps, [], R)
        points = [random_point(rng) for _ in range(400)]
        def answers():
            return [(ground.in_pond(p), ground.blocked(p, 0.6), ground.blocked(p, 0.6, swims=True)) for p in points]
        with_tree = answers()
        with mock.patch.object(module, "EVERY_CAP", True):
            every = answers()
        self.assertEqual(with_tree, every)
        self.assertTrue(any(a[0] for a in every) and any(a[1] and not a[2] for a in every))  # wet ones; ones a swimmer passes

if __name__ == "__main__":
    unittest.main()
