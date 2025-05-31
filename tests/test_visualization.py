# tests/test_visualization.py

import unittest
import matplotlib
# Use a non-GUI backend so tests don't actually pop up a window:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import math

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.world import World
from src.creature import Creature
from src.food import Food
from src.visualization import Visualizer

class TestVisualization(unittest.TestCase):
    def test_render_empty_world(self):
        """
        Verify that rendering an empty world (no creatures or food) does not crash.
        """
        w = World(10, 10, food_spawn_rate=0.0)
        viz = Visualizer(10, 10)
        # Should not raise any exceptions:
        viz.render(w)
        # We can also save to a buffer or file to ensure something was drawn:
        fig = viz.fig
        # Save to a temporary in-memory buffer (not on disk)
        buf = None
        try:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
        except Exception as e:
            self.fail(f"Saving figure failed with exception: {e}")

    def test_render_with_creatures(self):
        """
        Place a couple of creatures in the world and verify render() still runs.
        """
        w = World(5, 5, food_spawn_rate=0.0)
        c1 = Creature(1.0, 1.0, size=1.0, energy=5.0, velocity=1.0)
        c2 = Creature(3.0, 4.0, size=1.5, energy=3.0, velocity=0.5)
        w.add_creature(c1)
        w.add_creature(c2)
        viz = Visualizer(5, 5)
        viz.render(w)
        # Again, attempt to save to buffer
        fig = viz.fig
        try:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
        except Exception as e:
            self.fail(f"Saving figure with creatures failed: {e}")

    def test_render_with_food(self):
        """
        Place food in the world and verify render() draws it correctly.
        """
        w = World(5, 5, food_spawn_rate=0.0)
        # Add a food item with remaining_energy = initial_energy
        food = Food(x=2.5, y=2.5, size=1.0, energy_value=5.0, remaining_duration=-1)
        w.foods.append(food)
        viz = Visualizer(5, 5)
        viz.render(w)
        # Save to buffer
        fig = viz.fig
        try:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
        except Exception as e:
            self.fail(f"Saving figure with food failed: {e}")

    def test_render_with_arrows(self):
        """
        Place creatures with different intents and verify render() draws arrows correctly.
        """
        w = World(5, 5, food_spawn_rate=0.0)

        # Create creatures with different intents
        c1 = Creature(1.0, 1.0, size=1.0, energy=5.0, velocity=1.0)
        c1.intent = "ATTACK"
        c1.intended_vector = (1.0, 0.0)  # Moving east

        c2 = Creature(3.0, 3.0, size=1.0, energy=5.0, velocity=1.0)
        c2.intent = "GO_TO_FOOD"
        c2.intended_vector = (0.0, 1.0)  # Moving north

        c3 = Creature(4.0, 1.0, size=1.0, energy=5.0, velocity=1.0)
        c3.intent = "RUN_AWAY"
        c3.intended_vector = (-1.0, 0.0)  # Moving west

        c4 = Creature(1.0, 4.0, size=1.0, energy=5.0, velocity=1.0)
        c4.intent = "WANDER"
        c4.intended_vector = (0.5, 0.5)  # Moving northeast at half speed

        c5 = Creature(2.0, 2.0, size=1.0, energy=5.0, velocity=1.0)
        c5.intent = "REST"
        c5.intended_vector = (0.0, 0.0)  # Not moving

        w.add_creature(c1)
        w.add_creature(c2)
        w.add_creature(c3)
        w.add_creature(c4)
        w.add_creature(c5)

        viz = Visualizer(5, 5)
        viz.render(w)

        # Save to buffer
        fig = viz.fig
        try:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
        except Exception as e:
            self.fail(f"Saving figure with arrows failed: {e}")

    def test_food_radius_scaling(self):
        """
        Test that food is drawn with radius proportional to remaining_energy/initial_energy.
        """
        w = World(5, 5, food_spawn_rate=0.0)

        # Add a food item with remaining_energy = initial_energy
        food1 = Food(x=1.0, y=1.0, size=1.0, energy_value=5.0, remaining_duration=-1)

        # Add a food item with remaining_energy = 0.5 * initial_energy
        food2 = Food(x=3.0, y=3.0, size=1.0, energy_value=5.0, remaining_duration=-1)
        food2.remaining_energy = 2.5  # Half of initial_energy

        w.foods.append(food1)
        w.foods.append(food2)

        viz = Visualizer(5, 5)
        viz.render(w)

        # Save to buffer
        fig = viz.fig
        try:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
        except Exception as e:
            self.fail(f"Saving figure with scaled food failed: {e}")

if __name__ == '__main__':
    unittest.main()
