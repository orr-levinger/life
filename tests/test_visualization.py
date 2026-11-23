import io
import unittest

import matplotlib

matplotlib.use("Agg")  # no window in tests

from src.food import Food
from src.visualization import Visualizer
from tests.helpers import add, fixed_brain, make_world


class TestVisualization(unittest.TestCase):
    def test_render_empty_world(self):
        viz = Visualizer(50, 30)
        viz.render(make_world())
        viz.fig.savefig(io.BytesIO(), format="png")

    def test_render_with_creatures_and_food(self):
        w = make_world()
        add(w, 10, 10, policy="brain", brain=fixed_brain([1, 0, -1, -1]))
        w.foods.append(Food(20, 20, remaining_duration=-1, energy=10.0))
        w.step()
        viz = Visualizer(50, 30)
        viz.render(w)
        self.assertGreaterEqual(len(viz.ax.patches), 2)

    def test_corner_shows_level(self):
        viz = Visualizer(50, 30)
        world = make_world()
        world.set_settings(world.settings, 3)
        viz.render(world)
        self.assertTrue(any("Level: 3" in t.get_text() for t in viz.ax.texts))


if __name__ == "__main__":
    unittest.main()
