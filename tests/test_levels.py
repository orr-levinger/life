import unittest

from src.levels import LEVELS, level


class TestLevels(unittest.TestCase):
    def test_five_levels_in_order(self):
        self.assertEqual([lv.number for lv in LEVELS], [1, 2, 3, 4, 5])

    def test_level_one_matches_spec(self):
        first = level(1)
        self.assertEqual(first.food_energy, (20, 40))
        self.assertEqual(first.max_food, 200)
        self.assertEqual(first.move_cost, 0.5)
        self.assertEqual(first.step_cost, 0.05)
        self.assertIs(first.attacks, False)

    def test_attacks_from_level_four(self):
        self.assertEqual([lv.attacks for lv in LEVELS], [False, False, False, True, True])

    def test_unknown_level_raises(self):
        for number in (0, 6):
            with self.assertRaises(ValueError):
                level(number)


if __name__ == "__main__":
    unittest.main()
