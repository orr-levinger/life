import unittest
from dataclasses import replace
from unittest import mock

from src.food import Food
from src.levels import EAT_MISS_COST, LEVELS
from tests.helpers import NO_FOOD, NO_FOOD_ATTACKS, add, fixed_brain, make_world, populate

STAY = [0, 0, -1, -1]
EAT = [0, 0, 1, -1]


class TestCosts(unittest.TestCase):
    def test_step_cost(self):
        w = make_world()
        c = add(w, 10, 10, policy="brain", brain=fixed_brain(STAY))
        w.step()
        self.assertAlmostEqual(c.energy, 50.0 - NO_FOOD.step_cost)

    def test_move_cost_and_speed_cap(self):
        w = make_world()
        c = add(w, 10, 10, size=1.0)
        c.apply_action(("MOVE", (3.0, 4.0)), w)
        self.assertAlmostEqual(c.x, 10.6)
        self.assertAlmostEqual(c.y, 10.8)
        self.assertAlmostEqual(c.energy, 50.0 - 1.0 * NO_FOOD.move_cost - NO_FOOD.step_cost)

    def test_move_clamped_in_corner(self):
        w = make_world()
        c = add(w, 0.3, 0.3, size=1.0)
        c.apply_action(("MOVE", (-1.0, -1.0)), w)
        self.assertAlmostEqual(c.x, c.radius)
        self.assertAlmostEqual(c.y, c.radius)


class TestEating(unittest.TestCase):
    def test_bite_moves_one_energy(self):
        w = make_world()
        c = add(w, 10, 10)
        f = Food(10, 10, remaining_duration=-1, energy=10.0)
        w.foods.append(f)
        c.apply_action(("EAT", f), w)
        self.assertAlmostEqual(c.energy, 51.0 - NO_FOOD.step_cost)
        self.assertAlmostEqual(f.energy, 9.0)
        self.assertAlmostEqual(c.food_eaten, 1.0)
        self.assertEqual(w.total_bites, 1)

    def test_second_eater_misses_last_bite(self):
        w = make_world()
        first = add(w, 10, 10)
        second = add(w, 10.2, 10)
        f = Food(10.1, 10, remaining_duration=-1, energy=1.0)
        w.foods.append(f)
        first.apply_action(("EAT", f), w)
        second.apply_action(("EAT", f), w)
        self.assertNotIn(f, w.foods)
        self.assertEqual(second.last_action, "EAT_MISS")
        self.assertAlmostEqual(second.energy, 50.0 - EAT_MISS_COST - NO_FOOD.step_cost)

    def test_energy_capped_at_max(self):
        w = make_world()
        c = add(w, 10, 10, energy=99.5)
        f = Food(10, 10, remaining_duration=-1, energy=10.0)
        w.foods.append(f)
        c.apply_action(("EAT", f), w)
        self.assertEqual(c.energy, c.max_energy)


class TestHunting(unittest.TestCase):
    def test_hunter_gains_eighty_percent(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter = add(w, 10, 10, size=2.0, energy=100.0)
        prey = add(w, 11, 10, size=1.0, energy=50.0)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertAlmostEqual(prey.energy, 40.0)
        self.assertAlmostEqual(hunter.energy, 100.0 + 0.8 * 10.0 - 2.0 - NO_FOOD_ATTACKS.step_cost)

    def test_kill_leaves_corpse_and_counts_attack_death(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter = add(w, 10, 10, size=2.0, energy=100.0)
        prey = add(w, 11, 10, size=1.0, energy=5.0)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertNotIn(prey, w.creatures)
        self.assertEqual(w.deaths["attack"], 1)
        self.assertIn(prey, w.dead_this_step)
        corpse = w.foods[-1]
        self.assertEqual((corpse.x, corpse.y, corpse.energy, corpse.remaining_duration), (11, 10, 8.0, 5))

    def test_dead_target_is_a_miss(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter = add(w, 10, 10, size=2.0, energy=100.0)
        prey = add(w, 11, 10, size=1.0, energy=50.0)
        w.creatures.remove(prey)
        hunter.apply_action(("ATTACK", prey), w)
        self.assertEqual(hunter.last_action, "ATTACK_MISS")
        self.assertEqual(prey.energy, 50.0)

    def test_family_cannot_be_hurt(self):
        w = make_world(NO_FOOD_ATTACKS)
        parent = add(w, 10, 10, size=2.0, energy=100.0)
        child = add(w, 11, 10, size=1.0, energy=50.0, parent_id=parent.id)
        parent.apply_action(("ATTACK", child), w)
        self.assertEqual(child.energy, 50.0)
        self.assertEqual(parent.last_action, "ATTACK_MISS")


class TestLifeCycle(unittest.TestCase):
    def test_full_energy_does_not_split(self):
        w = make_world()
        add(w, 10, 10, size=1.0, energy=100.0, policy="brain", brain=fixed_brain(EAT))
        w.foods.append(Food(10, 10, remaining_duration=-1, energy=10.0))
        w.step()
        self.assertEqual(len(w.creatures), 1)

    def test_no_old_age(self):
        w = make_world(replace(NO_FOOD, step_cost=0.0))
        c = add(w, 10, 10, energy=100.0, policy="brain", brain=fixed_brain(STAY))
        for _ in range(3000):
            w.step()
        self.assertEqual(w.creatures, [c])
        self.assertEqual(sum(w.deaths.values()), 0)

    def test_food_respects_level(self):
        w = make_world(LEVELS[0])
        for _ in range(200):
            w.spawn_food()
        self.assertLessEqual(len(w.foods), LEVELS[0].max_food)
        self.assertGreater(len(w.foods), 0)
        self.assertTrue(all(20 <= f.energy <= 40 for f in w.foods))

    def test_same_seed_same_world(self):
        worlds = [make_world(LEVELS[0], seed=5), make_world(LEVELS[0], seed=5)]
        for w in worlds:
            populate(w)
            for _ in range(200):
                w.step()
        positions = [[(c.id, c.x, c.y) for c in w.creatures] for w in worlds]
        self.assertEqual(positions[0], positions[1])

    def test_exam_world_never_kills_or_splits(self):
        w = make_world(evolving=False)
        c = add(w, 10, 10, energy=1.0, policy="brain", brain=fixed_brain([1, 0, -1, -1]))
        for _ in range(100):
            w.step()
        self.assertEqual(w.creatures, [c])
        self.assertEqual(sum(w.deaths.values()), 0)

    def test_brain_creatures_never_call_rules(self):
        w = make_world(LEVELS[0], seed=2)
        populate(w)
        with mock.patch("src.policies.rules_action", side_effect=AssertionError("rules called for a brain")):
            for _ in range(500):
                w.step()


if __name__ == "__main__":
    unittest.main()
