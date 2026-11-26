"""The planet's geometry (src/sphere.py): points on a ball, distances along its surface, a creature's own view."""
import math
import unittest

import numpy as np

from src.sphere import arc, bearing, exp_map, local_frame, random_point, tangent_toward, walk

R = 37.0


class TestSphere(unittest.TestCase):
    def test_a_quarter_turn_is_a_quarter_of_the_way(self):
        self.assertAlmostEqual(float(arc(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), R)), math.pi * R / 2)

    def test_opposite_and_same_points(self):
        p = np.array([0.0, 0.6, 0.8])
        self.assertAlmostEqual(float(arc(p, -p, R)), math.pi * R)
        self.assertEqual(float(arc(p, p, R)), 0.0)
        for q in (p, -p):
            self.assertTrue(np.all(np.isfinite(tangent_toward(p, q))))
            np.testing.assert_allclose(tangent_toward(p, q), 0.0)

    def test_walking_straight_comes_back(self):
        rng = np.random.default_rng(1)
        steps = 2000
        for _ in range(5):
            start = random_point(rng)
            facing = local_frame(start)[0]
            p, f = start.copy(), facing.copy()
            for _ in range(steps):
                p, f = walk(p, f, 1.0, 0.0, 2 * math.pi * R / steps, R)
            self.assertLess(float(np.linalg.norm(p - start)) * R, 0.001)
            self.assertLess(float(np.linalg.norm(f - facing)), 0.001)
            self.assertAlmostEqual(float(np.linalg.norm(p)), 1.0)

    def test_a_step_to_the_left_turns_it_left(self):
        p, f = np.array([0.0, 0, 1.0]), np.array([1.0, 0, 0])
        p2, f2 = walk(p, f, 0.0, 1.0, 0.5, R)
        np.testing.assert_allclose(f2, np.cross(p2, f) / np.linalg.norm(np.cross(p2, f)), atol=1e-9)  # it faces its left
        self.assertAlmostEqual(float(arc(p, p2, R)), 0.5)

    def test_bearing_is_the_creatures_own_view(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            p = random_point(rng)
            e1, e2 = local_frame(p)
            angle = rng.uniform(0, 2 * math.pi)
            f = math.cos(angle) * e1 + math.sin(angle) * e2
            ahead, _ = walk(p, f, 1.0, 0.0, 2.0, R)
            left, _ = walk(p, f, 0.0, 1.0, 2.0, R)
            np.testing.assert_allclose(bearing(p, f, ahead), (1.0, 0.0), atol=1e-9)
            np.testing.assert_allclose(bearing(p, f, left), (0.0, 1.0), atol=1e-9)

    def test_many_at_once(self):
        rng = np.random.default_rng(3)
        ps = np.stack([random_point(rng) for _ in range(4)])
        qs = np.stack([random_point(rng) for _ in range(6)])
        d = arc(ps[:, None, :], qs[None, :, :], R)
        self.assertEqual(d.shape, (4, 6))
        self.assertAlmostEqual(float(d[2, 3]), float(arc(ps[2], qs[3], R)))

    def test_exp_map_keeps_distance_from_its_base(self):
        base = np.array([0.0, 0, 1.0])
        e1, e2 = local_frame(base)
        for x, y in ((3.0, 4.0), (-10.0, 2.0), (0.0, 0.0), (20.0, -15.0)):
            q = exp_map(base, e1, e2, x, y, R)
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0)
            self.assertAlmostEqual(float(arc(base, q, R)), math.hypot(x, y))

    def test_random_points_cover_the_ball_evenly(self):
        rng = np.random.default_rng(4)
        pts = np.stack([random_point(rng) for _ in range(8000)])
        octants = np.unique((pts > 0).astype(int) @ np.array([4, 2, 1]), return_counts=True)[1]
        self.assertEqual(len(octants), 8)
        self.assertTrue(np.all(np.abs(octants - 1000) < 250))


if __name__ == "__main__":
    unittest.main()
