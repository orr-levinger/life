# tests/test_continuous_movement.py

import unittest
import matplotlib
matplotlib.use('Agg')
import random
import math
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.world import World
from src.creature import Creature
from src.sensors import VisionSensor

class TestContinuousMovement(unittest.TestCase):
    def test_max_speed_enforced(self):
        """
        Creature with size=1.0 → velocity=1.0. If monkey-patched to move (2,0),
        after world.step() it should only move by (1,0) and energy drop by 1.0.
        """
        w = World(10, 10, food_spawn_rate=0.0)
        c = Creature(5.0, 5.0, size=1.0, energy=10.0)
        w.add_creature(c)

        # Force movement twice max
        def go_too_fast(vision, on_food=False):
            return ("MOVE", (2.0, 0.0))
        c.decide = go_too_fast

        w.step()
        # x increased by exactly 1.0 (max velocity), y unchanged
        self.assertAlmostEqual(c.x, 6.0, places=5)
        self.assertAlmostEqual(c.y, 5.0, places=5)
        self.assertAlmostEqual(c.energy, 9.0, places=5)

    def test_flee_from_adjacent(self):
        """
        Two creatures at (2,2) and (2,3). Prey at (2,2) sees predator north,
        so should move directly south by velocity (which is 1.0).

        Note: In the new predation system, creatures attack by default when they see other creatures.
        For this test, we override the prey's decide method to flee instead of attack.
        """
        w = World(10, 10, food_spawn_rate=0.0)
        predator = Creature(2.0, 3.0, size=1.0, energy=5.0)
        prey = Creature(2.0, 2.0, size=1.0, energy=5.0)
        w.add_creature(predator)
        w.add_creature(prey)

        # Freeze predator so it doesn't move during the step
        def always_rest(vision, on_food=False):
            return ("REST", None)
        predator.decide = always_rest

        # Override prey's decide method to flee instead of attack
        def flee_from_creatures(vision, on_food=False):
            # If creature to the north, flee south
            if vision.get("north") == "creature":
                return ("MOVE", (0.0, -prey.velocity))
            # If creature to the south, flee north
            elif vision.get("south") == "creature":
                return ("MOVE", (0.0, prey.velocity))
            # If creature to the east, flee west
            elif vision.get("east") == "creature":
                return ("MOVE", (-prey.velocity, 0.0))
            # If creature to the west, flee east
            elif vision.get("west") == "creature":
                return ("MOVE", (prey.velocity, 0.0))
            # Otherwise, rest
            return ("REST", None)

        prey.decide = flee_from_creatures

        # First, verify that the sensor reading and decision are correct
        vs = VisionSensor()
        reading = vs.get_reading(prey, w)
        action = prey.decide(reading)
        # Should be ("MOVE", (0.0, -1.0))
        self.assertEqual(action[0], "MOVE")
        dx, dy = action[1]
        self.assertAlmostEqual(dx, 0.0, places=5)
        self.assertAlmostEqual(dy, -1.0, places=5)

        # After world.step(), prey.y should be 1.0 and energy 4.0
        w.step()
        self.assertAlmostEqual(prey.x, 2.0, places=5)
        self.assertAlmostEqual(prey.y, 1.0, places=5)
        self.assertAlmostEqual(prey.energy, 4.0, places=5)

    def test_random_wander_and_rest(self):
        """
        In empty world, creature should sometimes move ~velocity, sometimes rest (energy drops 0.1).
        Run 50 trials and confirm at least one move and one rest occur.
        """
        w = World(10, 10, food_spawn_rate=0.0)
        c = Creature(5.0, 5.0, size=1.0, energy=100.0)
        w.add_creature(c)
        vs = VisionSensor()

        seen_move = False
        seen_rest = False

        # Run multiple trials
        for _ in range(50):
            # Reset position and get reading
            c.x, c.y = 5.0, 5.0
            initial_energy = c.energy

            reading = vs.get_reading(c, w)
            action = c.decide(reading)

            # Apply the action to update energy and position
            c.apply_action(action, w)

            if action[0] == "MOVE":
                seen_move = True
                # Check that energy decreased by approximately the distance moved
                dx, dy = action[1]
                expected_dist = min(math.hypot(dx, dy), c.velocity)
                self.assertAlmostEqual(initial_energy - c.energy, expected_dist, places=5)
                # Reset energy and position
                c.energy = initial_energy
                c.x, c.y = 5.0, 5.0
            elif action[0] == "REST":
                seen_rest = True
                # Check that energy decreased by 0.1
                self.assertAlmostEqual(initial_energy - c.energy, 0.1, places=5)
                # Reset energy
                c.energy = initial_energy

            if seen_move and seen_rest:
                break

        self.assertTrue(seen_move, "No MOVE occurred in 50 trials")
        self.assertTrue(seen_rest, "No REST occurred in 50 trials")

    def test_bounds_clamping(self):
        """
        Creature at (0.5,0.5) tries to move SW by (-1,-1). That has magnitude ~1.414,
        which should be clamped to velocity=1.0 along angle 225°. New position becomes
        (0.5 + (-√2/2), 0.5 + (-√2/2)) clamped to (0,0). Energy drops by 1.0.
        """
        w = World(10, 10, food_spawn_rate=0.0)
        c = Creature(0.5, 0.5, size=1.0, energy=5.0)
        w.add_creature(c)

        def go_sw(vision, on_food=False):
            return ("MOVE", (-1.0, -1.0))
        c.decide = go_sw

        w.step()
        # Compute the normalized SW vector of length 1.0: 
        norm = 1.0 / math.sqrt(2)
        expected_x = 0.5 + (-norm)
        expected_y = 0.5 + (-norm)
        # But clamped to 0.0 minimum
        self.assertAlmostEqual(c.x, max(expected_x, 0.0), places=5)
        self.assertAlmostEqual(c.y, max(expected_y, 0.0), places=5)
        # Energy should have dropped by 1.0
        self.assertAlmostEqual(c.energy, 4.0, places=5)

if __name__ == "__main__":
    unittest.main()
