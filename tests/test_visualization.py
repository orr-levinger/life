# tests/test_visualization.py

import unittest
import matplotlib
# Use a non-GUI backend so tests don't actually pop up a window:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.world import World
from src.creature import Creature
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
        c1 = Creature(1, 1, size=1.0, energy=5.0, velocity=1.0)
        c2 = Creature(3, 4, size=1.5, energy=3.0, velocity=0.5)
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

if __name__ == '__main__':
    unittest.main()