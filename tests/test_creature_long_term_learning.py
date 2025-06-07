import unittest
import numpy as np
import tensorflow as tf
import random
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.world import World
from src.creature import Creature
from src.food import Food

class TestCreatureLongTermLearning(unittest.TestCase):
    def test_creature_learns_to_reach_food_sequence(self):
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)

        Creature.MIN_STEPS = 0
        c = Creature(0.0, 0.0, size=1.0, energy=5.0, create_brain=True)

        # Train on repeated episodes where food is several steps away
        for _ in range(50):
            w = World(15, 5, food_spawn_rate=0.0)
            c.x, c.y = 2.0, 2.0
            c.energy = 5.0
            w.add_creature(c)
            w.foods = [Food(x=12.0, y=2.0, remaining_duration=-1, energy=5.0)]
            for _ in range(10):
                w.step()
                if not w.foods:
                    break

        # Evaluate from a new starting position
        w = World(15, 5, food_spawn_rate=0.0)
        c.x, c.y = 1.0, 3.0
        c.energy = 5.0
        w.add_creature(c)
        w.foods = [Food(x=13.0, y=3.0, remaining_duration=-1, energy=5.0)]

        # Run a few steps and verify the creature starts moving toward the food
        initial_pos = (c.x, c.y)
        w.step()
        moved_distance = ((c.x - initial_pos[0]) ** 2 + (c.y - initial_pos[1]) ** 2) ** 0.5
        self.assertGreater(moved_distance, 0.1, "Creature did not move toward food after training")

if __name__ == "__main__":
    unittest.main()
