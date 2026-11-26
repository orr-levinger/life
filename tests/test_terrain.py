import math
import random
import unittest
from dataclasses import replace

import numpy as np

from src.food import Food
from src.rules import starting_rules
from src.senses import N_SENSES, sense
from src.terrain import Obstacle, Terrain, Tree, make_terrain
from tests.helpers import NO_FOOD, add, make_world

ROCK = Obstacle("rock", ((20.0, 10.0, 2.0),))  # one round rock: x 18-22 at y 10


def walled(world, obstacles=(ROCK,), trees=()):
    world.terrain = Terrain(list(obstacles), list(trees))
    return world


class TestMakingTerrain(unittest.TestCase):
    def test_a_world_gets_its_obstacles_and_trees_none_on_another(self):
        rules = {**starting_rules(), "obstacles": 5.0, "trees": 6.0}
        terrain = make_terrain(rules, random.Random(3), 50.0, 30.0, keep_clear=[(10.0, 10.0, 1.0)])
        self.assertEqual((len(terrain.obstacles), len(terrain.trees)), (5, 6))
        circles = [c for o in terrain.obstacles for c in o.circles]
        self.assertTrue(all(0 <= x <= 50 and 0 <= y <= 30 for x, y, _ in circles))
        self.assertFalse(terrain.blocked(10.0, 10.0, 1.0))  # the spot kept clear
        self.assertTrue(all(not terrain.blocked(t.x, t.y, 0.0) for t in terrain.trees))
        sizes = {round(max(c[2] for c in o.circles), 2) for o in terrain.obstacles}
        self.assertGreater(len(sizes), 1)  # of random sizes

    def test_the_terrain_round_trips(self):
        terrain = make_terrain(starting_rules(), random.Random(1), 50.0, 30.0)
        back = Terrain.from_state(terrain.state())
        self.assertEqual((back.obstacles, back.trees), (terrain.obstacles, terrain.trees))


class TestObstacles(unittest.TestCase):
    def test_food_never_grows_on_an_obstacle(self):
        w = walled(make_world(replace(NO_FOOD, max_food=400, food_per_step=6.0)),
                   [Obstacle("pond", ((25.0, 15.0, 10.0),))])
        for _ in range(60):
            w.spawn_food()
        self.assertTrue(w.foods)
        self.assertTrue(all(math.hypot(f.x - 25.0, f.y - 15.0) >= 10.0 + f.radius for f in w.foods))

    def test_moving_into_an_obstacle_is_refused_and_costs_energy(self):
        w = walled(make_world())
        c = add(w, 17.5, 10.0, size=1.0)
        before = c.energy
        c.apply_action(("MOVE", (0.3, 0.0)), w)  # its body would touch the rock
        self.assertEqual((c.x, c.y, c.moved), (17.5, 10.0, False))
        self.assertTrue(c.bumped)
        free = add(w, 5.0, 5.0, size=1.0)
        free_before = free.energy
        free.apply_action(("MOVE", (0.0, 0.0)), w)
        self.assertAlmostEqual(before - c.energy, (free_before - free.energy) + w.rules["bump_cost"])

    def test_an_obstacle_hides_what_is_behind_it(self):
        w = walled(make_world())
        me = add(w, 15.0, 10.0, size=1.0)
        me.vision_width, me.vision_reach, me.facing = 120.0, 15.0, (1.0, 0.0)
        w.foods.append(Food(25.0, 10.0, remaining_duration=-1, energy=10.0))  # behind the rock
        self.assertIsNone(sense(me, w).food)
        w.foods.append(Food(17.0, 12.5, remaining_duration=-1, energy=10.0))  # beside it, in the open
        self.assertIsNotNone(sense(me, w).food)

    def test_checking_only_the_pairs_asked_gives_the_same_answer(self):
        rng = np.random.default_rng(5)
        terrain = make_terrain(starting_rules(), random.Random(5), 120.0, 72.0)
        ax, ay = rng.uniform(0, 120, 40), rng.uniform(0, 72, 40)
        bx, by = rng.uniform(0, 120, 70), rng.uniform(0, 72, 70)
        mask = rng.random((40, 70)) < 0.2
        full = terrain.sight_blocked(ax, ay, bx, by)
        self.assertTrue(full.any() and (~full).any())  # some lines blocked, some open
        np.testing.assert_array_equal(terrain.sight_blocked(ax, ay, bx, by, only=mask), full & mask)

    def test_an_obstacle_in_its_field_of_vision_is_sensed(self):
        w = walled(make_world())
        me = add(w, 15.0, 10.0, size=1.0)
        me.vision_width, me.vision_reach = 120.0, 5.0
        me.facing = (1.0, 0.0)
        v = sense(me, w).vector
        self.assertEqual(len(v), N_SENSES)
        self.assertEqual(v[20], 1.0)  # seen
        self.assertAlmostEqual(v[21], 3.0 / 5.0)  # its edge 3 away, ÷ its vision reach
        self.assertAlmostEqual(v[22], 1.0)  # ahead
        me.facing = (-1.0, 0.0)  # looking away: not seen
        self.assertTrue(np.all(sense(me, w).vector[20:24] == 0.0))


class TestTrees(unittest.TestCase):
    def test_a_tree_drops_fruit_near_its_trunk_now_and_then(self):
        w = walled(make_world(replace(NO_FOOD, max_food=400, fruit_rate=1.0)), [], [Tree(25.0, 15.0, 1.5)])
        for _ in range(40):  # a fruit a step, where one fits (not on another)
            w.spawn_food()
        fruit = [f for f in w.foods if f.fruit]
        self.assertGreater(len(fruit), 3)
        self.assertTrue(all(math.hypot(f.x - 25.0, f.y - 15.0) <= w.rules["fruit_spread"] for f in fruit))
        none = walled(make_world(replace(NO_FOOD, max_food=400, fruit_rate=0.0)), [], [Tree(25.0, 15.0, 1.5)])
        for _ in range(10):
            none.spawn_food()
        self.assertEqual(none.foods, [])

    def test_a_creature_under_a_tree_is_seen_only_from_close_by(self):
        w = walled(make_world(), [], [Tree(25.0, 10.0, 1.5)])
        hidden = add(w, 25.0, 10.0, size=1.0)
        far = add(w, 20.0, 10.0, size=1.0)
        far.vision_width, far.vision_reach, far.facing = 120.0, 10.0, (1.0, 0.0)
        self.assertIsNone(sense(far, w).other)  # 5 away: it hides under the tree
        near = add(w, 25.0 - w.rules["hide_reach"] * 0.9, 10.0, size=1.0)
        near.vision_width, near.vision_reach, near.facing = 120.0, 10.0, (1.0, 0.0)
        self.assertIs(sense(near, w).other, hidden)
        self.assertEqual(sense(hidden, w).vector[24], 1.0)  # it knows it is under a tree
        self.assertEqual(sense(far, w).vector[24], 0.0)
