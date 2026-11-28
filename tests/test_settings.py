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
        self.assertEqual(START, Settings(max_food=400, food_energy=(20, 40), food_per_step=6.0, lifespan=(4000, 6000),
                                         attacks=False, fruit_rate=0.02, max_fruit=300.0))  # the costs are rules (tests/test_costs.py)

    def test_the_start_is_easier_than_the_test_world(self):
        self.assertGreater(START.max_food, TEST_WORLD.max_food)
        self.assertGreater(START.food_energy, TEST_WORLD.food_energy)

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
