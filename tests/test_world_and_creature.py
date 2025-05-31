import sys, os
import unittest
import math

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.world import World
from src.creature import Creature

class TestWorldAndCreature(unittest.TestCase):
    def test_creature_initialization(self):
        c = Creature(0.0, 0.0, size=1.0, energy=10.0, velocity=1.0)
        self.assertEqual(c.x, 0.0)
        self.assertEqual(c.y, 0.0)
        self.assertEqual(c.size, 1.0)
        self.assertEqual(c.energy, 10.0)
        self.assertEqual(c.velocity, 1.0)
        self.assertEqual(c.radius, c.size * c.RADIUS_FACTOR)
        self.assertEqual(c.intent, "REST")
        self.assertEqual(c.intended_vector, (0.0, 0.0))
        self.assertEqual(c.current_speed, c.velocity)
        self.assertIsNone(c.brain)

    def test_world_add_creature(self):
        world = World(10, 10, food_spawn_rate=0.0)
        c = Creature(5.0, 5.0, size=1.0, energy=10.0, velocity=1.0)
        world.add_creature(c)
        self.assertIn(c, world.creatures)

    def test_world_step_reduces_energy_and_removes_dead(self):
        world = World(10, 10, food_spawn_rate=0.0)
        # Create a creature with energy=0.1 so it will die after one REST step
        c = Creature(5.0, 5.0, size=1.0, energy=0.1, velocity=1.0)
        world.add_creature(c)

        # Override decide to always return REST
        def always_rest(vision, on_food=False):
            return ("REST", None)
        c.decide = always_rest

        world.step()
        # After one step: energy = 0 → removed from "creatures"
        self.assertEqual(len(world.creatures), 0)

    def test_multiple_steps_survival(self):
        world = World(10, 10, food_spawn_rate=0.0)
        c = Creature(5.0, 5.0, size=1.0, energy=5.0, velocity=1.0)
        world.add_creature(c)

        # Override decide to always return REST
        def always_rest(vision, on_food=False):
            return ("REST", None)
        c.decide = always_rest

        # Run 3 steps: each step deducts 0.1 energy for REST → final energy = 5 - 0.3 = 4.7
        for _ in range(3):
            world.step()
        self.assertEqual(len(world.creatures), 1)
        self.assertAlmostEqual(world.creatures[0].energy, 4.7, places=5)

    def test_variable_speed_movement(self):
        world = World(10, 10, food_spawn_rate=0.0)
        c = Creature(5.0, 5.0, size=1.0, energy=10.0, velocity=1.0)
        world.add_creature(c)

        # Test slow movement (0.5 * max_velocity)
        def move_slow(vision, on_food=False):
            # Move at half speed
            c.current_speed = c.velocity * 0.5
            dx = c.current_speed
            dy = 0.0
            c.intended_vector = (dx, dy)
            c.intent = "WANDER"
            return ("MOVE", (dx, dy))
        c.decide = move_slow

        # Initial energy
        initial_energy = c.energy

        world.step()
        # x increased by exactly 0.5 (half velocity), y unchanged
        self.assertAlmostEqual(c.x, 5.5, places=5)
        self.assertAlmostEqual(c.y, 5.0, places=5)
        # Energy decreased by exactly 0.5 (distance moved)
        self.assertAlmostEqual(c.energy, initial_energy - 0.5, places=5)

    def test_zero_speed_movement(self):
        world = World(10, 10, food_spawn_rate=0.0)
        c = Creature(5.0, 5.0, size=1.0, energy=10.0, velocity=1.0)
        world.add_creature(c)

        # Test zero movement
        def move_zero(vision, on_food=False):
            # Move with zero speed
            c.current_speed = 0.0
            dx = 0.0
            dy = 0.0
            c.intended_vector = (dx, dy)
            c.intent = "WANDER"
            return ("MOVE", (dx, dy))
        c.decide = move_zero

        # Initial energy
        initial_energy = c.energy

        world.step()
        # Position unchanged
        self.assertAlmostEqual(c.x, 5.0, places=5)
        self.assertAlmostEqual(c.y, 5.0, places=5)
        # Energy unchanged (no movement cost)
        self.assertAlmostEqual(c.energy, initial_energy, places=5)

if __name__ == "__main__":
    unittest.main()
