import unittest

from src.stages import STAGES, TEST_WORLD, stage


class TestStages(unittest.TestCase):
    def test_five_stages_garden_first(self):
        self.assertEqual([(s.number, s.name) for s in STAGES],
                         [(0, "Garden"), (1, "Mortal"), (2, "Scarcity"), (3, "Hunters"), (4, "Harsh")])
        self.assertIs(stage(0), STAGES[0])
        for number in (-1, 5):
            with self.assertRaises(ValueError):
                stage(number)

    def test_food_never_increases(self):
        for easier, harder in zip(STAGES, STAGES[1:]):
            self.assertGreaterEqual(easier.max_food, harder.max_food, harder.name)
            self.assertGreaterEqual(easier.food_per_step, harder.food_per_step, harder.name)
            self.assertGreaterEqual(easier.food_energy[1], harder.food_energy[1], harder.name)
            self.assertLessEqual(easier.move_cost, harder.move_cost, harder.name)
            self.assertLessEqual(easier.step_cost, harder.step_cost, harder.name)

    def test_only_the_garden_cannot_starve(self):
        self.assertEqual([s.hunger for s in STAGES], [False, True, True, True, True])

    def test_hunting_from_stage_three(self):
        self.assertEqual([s.attacks for s in STAGES], [False, False, False, True, True])

    def test_the_exam_has_its_own_fixed_world(self):
        self.assertEqual((TEST_WORLD.max_food, TEST_WORLD.food_energy, TEST_WORLD.food_per_step), (80, (10, 25), 3))
        self.assertNotIn(TEST_WORLD, STAGES)


if __name__ == "__main__":
    unittest.main()
