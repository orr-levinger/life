import random
import time
import unittest

import numpy as np

from src.rules import starting_rules
from src.sphere import exp_map, local_frame
from src.water import Water, make_water


def planet(seed: int = 1, radius: float = 37.0, **changes) -> Water:
    return make_water({**starting_rules(), **changes}, random.Random(seed), radius)


def points(n: int, seed: int = 0) -> np.ndarray:
    p = np.random.default_rng(seed).normal(size=(n, 3))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def ring(centre: np.ndarray, distance: float, radius: float, n: int = 24) -> np.ndarray:
    e1, e2 = local_frame(centre)
    return np.array([exp_map(centre, e1, e2, distance * np.cos(a), distance * np.sin(a), radius)
                     for a in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)])


class TestWaterMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = planet()

    def test_about_half_is_water(self):
        self.assertAlmostEqual(float(self.w.is_water(points(100_000)).mean()), 0.5, delta=0.03)
        dry = planet(2, water_share=0.0, islands=0.0, rivers=0.0)
        self.assertEqual(float(dry.is_water(points(20_000)).mean()), 0.0)
        wet = planet(3, water_share=0.8)
        self.assertAlmostEqual(float(wet.is_water(points(100_000)).mean()), 0.8, delta=0.03)

    def test_islands_are_cut_off(self):
        rules = starting_rules()
        self.assertGreaterEqual(len(self.w.islands), 6)
        for island in self.w.islands:
            centre, r = island[:3], float(island[3])
            self.assertFalse(self.w.wet_exact(centre[None, :])[0])  # the island is land
            for d in np.linspace(r + 0.5, r + rules["island_gap"] - 0.5, 5):  # all around it, water
                self.assertTrue(self.w.wet_exact(ring(centre, d, 37.0)).all(), (island, d))

    def test_rivers_reach_water_and_have_crossings(self):
        rules = starting_rules()
        self.assertEqual(len(self.w.rivers), 3)
        for river in self.w.rivers:
            self.assertTrue(self.w.open_water(river[-1:])[0])  # it ends in the sea or a lake
            self.assertFalse(self.w.open_water(river[:1])[0])  # it starts on land
            near = [c for c in self.w.crossings
                    if np.min(np.linalg.norm(river - c, axis=1)) * 37.0 <= rules["river_width"]]
            self.assertEqual(len(near), int(rules["river_crossings"]))
            self.assertFalse(self.w.wet_exact(np.array(near)).any())  # walkers cross there
            middle = river[len(river) // 2]
            if not any(np.linalg.norm(middle - c) * 37.0 < 2.0 * rules["river_width"] for c in near):
                self.assertTrue(self.w.wet_exact(middle[None, :])[0])  # elsewhere it is water

    def test_the_grid_is_the_rule(self):
        p = points(10_000, seed=5)
        np.testing.assert_array_equal(self.w.is_water(p), self.w.wet_exact(self.w.cell_centres(p)))
        corners = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1], [1, 1, 1], [-1, 1, 0], [1, -1, -1]], dtype=float)
        corners /= np.linalg.norm(corners, axis=1, keepdims=True)
        self.assertEqual(self.w.is_water(corners).shape, (6,))  # edges and corners of the cube answer
        self.assertEqual(self.w.is_water_at(corners[0]), bool(self.w.is_water(corners[:1])[0]))

    def test_water_comes_back_from_a_save(self):
        again = Water.from_state(self.w.state())
        p = points(10_000, seed=7)
        np.testing.assert_array_equal(again.is_water(p), self.w.is_water(p))

    def test_any_planet_size_makes_water(self):
        for radius in (5.0, 2000.0):
            for share in (0.0, 0.8):
                start = time.perf_counter()
                w = planet(4, radius, water_share=share)
                self.assertLess(time.perf_counter() - start, 5.0, (radius, share))
                self.assertEqual(w.is_water(points(100)).shape, (100,))

    def test_the_picture_is_a_png(self):
        png = self.w.picture()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual((int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")), (1024, 512))


if __name__ == "__main__":
    unittest.main()
