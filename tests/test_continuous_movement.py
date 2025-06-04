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
from src.sensors import ProximitySensor

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
            # Set intent and intended_vector for visualization
            c.intent = "WANDER"
            c.intended_vector = (2.0, 0.0)
            return ("MOVE", (2.0, 0.0))
        c.decide = go_too_fast

        w.step()
        # x increased by exactly 1.0 (max velocity), y unchanged
        self.assertAlmostEqual(c.x, 6.0, places=5)
        self.assertAlmostEqual(c.y, 5.0, places=5)
        self.assertAlmostEqual(c.energy, 9.0, places=5)
        # Check that current_speed was clamped to max velocity
        self.assertAlmostEqual(c.current_speed, c.velocity, places=5)

    def test_flee_from_nearby(self):
        """
        Two creatures at (2,2) and (2,3). Prey at (2,2) sees predator nearby,
        so should move directly away at max speed.
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
            # Find nearby creatures
            nearby_creatures = [(obj, dist, angle) for type_tag, obj, dist, angle in vision if type_tag == "creature"]

            if nearby_creatures:
                # Sort by distance (closest first)
                nearby_creatures.sort(key=lambda x: x[1])
                closest_creature, distance, angle = nearby_creatures[0]

                # Flee in the opposite direction
                flee_angle = angle + math.pi  # Opposite direction

                # Set speed to maximum for fleeing
                prey.current_speed = prey.velocity
                prey.intent = "RUN_AWAY"

                # Calculate direction vector away from the creature
                dx = math.cos(flee_angle) * prey.current_speed
                dy = math.sin(flee_angle) * prey.current_speed

                # Store the intended vector
                prey.intended_vector = (dx, dy)

                return ("FLEE", (dx, dy))

            # If no creatures nearby, rest
            return ("REST", None)

        prey.decide = flee_from_creatures

        # Initial energy
        initial_energy = prey.energy

        # Call world.step()
        w.step()

        # Verify prey moved away from predator (should move south)
        self.assertAlmostEqual(prey.x, 2.0, places=5)
        self.assertAlmostEqual(prey.y, 1.0, places=5)  # Moved south by velocity=1.0
        # Energy decreased by distance moved (velocity=1.0)
        self.assertAlmostEqual(prey.energy, initial_energy - prey.velocity, places=5)
        # Check intent was set correctly
        self.assertEqual(prey.intent, "RUN_AWAY")

    def test_random_wander_and_rest(self):
        """
        In empty world, creature should sometimes move ~velocity, sometimes rest (energy drops 0.1).
        Run 50 trials and confirm at least one move and one rest occur.
        """
        w = World(10, 10, food_spawn_rate=0.0)
        # Create a creature without a neural network so it uses the old decision logic
        c = Creature(5.0, 5.0, size=1.0, energy=100.0, brain=None)
        w.add_creature(c)

        seen_move = False
        seen_rest = False

        # Run multiple trials
        for _ in range(50):
            # Reset position and energy
            c.x, c.y = 5.0, 5.0
            initial_energy = c.energy

            # Get empty vision (no nearby objects)
            vision = []

            # Decide action based on empty vision
            action = c.decide(vision)

            # Apply the action to update energy and position
            c.apply_action(action, w)

            if action[0] == "MOVE":
                seen_move = True
                # Check that energy decreased by approximately the distance moved
                dx, dy = action[1]
                expected_dist = min(math.hypot(dx, dy), c.velocity)
                self.assertAlmostEqual(initial_energy - c.energy, expected_dist, places=5)
                # Check that intent was set to "WANDER"
                self.assertEqual(c.intent, "WANDER")
                # Reset energy and position
                c.energy = initial_energy
                c.x, c.y = 5.0, 5.0
            elif action[0] == "REST":
                seen_rest = True
                # Check that energy decreased by 0.1
                self.assertAlmostEqual(initial_energy - c.energy, 0.1, places=5)
                # Check that intent was set to "REST"
                self.assertEqual(c.intent, "REST")
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
        # Create creature and override radius to 0.0 for this test
        c = Creature(0.5, 0.5, size=1.0, energy=5.0)
        c.radius = 0.0
        w.add_creature(c)

        def go_sw(vision, on_food=False):
            # Set intent and intended_vector for visualization
            c.intent = "WANDER"
            c.intended_vector = (-1.0, -1.0)
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
        # Check that current_speed was clamped to max velocity
        self.assertAlmostEqual(c.current_speed, c.velocity, places=5)

if __name__ == "__main__":
    unittest.main()
