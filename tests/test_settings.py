import math
import unittest
from dataclasses import FrozenInstanceError

from src.rules import rule
from src.exam import exam_settings
from src.rules import starting_rules
from src.settings import START, Settings

TEST_WORLD = exam_settings(starting_rules())


class TestSettings(unittest.TestCase):
    def test_the_start_is_the_easy_start_of_the_design(self):
        seeing = 0.05 / (math.pi * 9.0)  # every cost a tenth of today's (the owner, 2026-11-26)
        self.assertEqual(START, Settings(max_food=400, food_energy=(20, 40), food_per_step=6.0, move_cost=0.05,
                                         step_cost=0.005, lifespan=(4000, 6000), attacks=False, fruit_rate=0.02,
                                         sprint_power=0.2, eat_miss=0.02, attack_cost=0.1, attack_damage=0.5,
                                         bump_cost=0.05, vision_cost=seeing / 10.0, hearing_cost=seeing / 10.0,
                                         neuron_cost=0.00005, stealth_cost=0.03, armour_cost=0.03))

    def test_the_start_is_easier_than_the_test_world(self):
        self.assertGreater(START.max_food, TEST_WORLD.max_food)
        self.assertGreater(START.food_energy, TEST_WORLD.food_energy)
        self.assertLess(START.move_cost, TEST_WORLD.move_cost)
        self.assertLess(START.step_cost, TEST_WORLD.step_cost)

    def test_a_lifespan_is_a_range(self):
        shortest, longest = START.lifespan
        self.assertLess(shortest, longest)

    def test_the_exam_has_its_own_fixed_world(self):
        self.assertEqual((TEST_WORLD.max_food, TEST_WORLD.food_energy, TEST_WORLD.food_per_step), (80, (10, 25), 3))
        self.assertFalse(TEST_WORLD.attacks)

    def test_settings_are_never_changed_in_place(self):
        with self.assertRaises(FrozenInstanceError):
            START.max_food = 1

    def test_only_the_poorly_fed_die_of_old_age(self):
        self.assertEqual(rule("low_energy").start, 0.3)


if __name__ == "__main__":
    unittest.main()
