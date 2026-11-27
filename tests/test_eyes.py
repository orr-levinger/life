"""Eyes of pixels, ears of directions (the owner, 2026-11-27: "some animals have very high resolution sight, some have
very low resolution")."""
import unittest

from src.rules import RULES, starting_rules
from src.world import GENES
from tests.helpers import add, make_world


class TestSharpness(unittest.TestCase):
    def test_sharpness_is_a_gene(self):
        by_name = {r.name: r for r in RULES}
        self.assertEqual([by_name[f"gene.eye_pixels.{end}"].start for end in ("low", "high", "new_low", "new_high")],
                         [1.0, 16.0, 2.0, 8.0])
        self.assertIn("eye_pixels", GENES)  # a baby may change it
        w = make_world(seed=4)
        drawn = {w.add_newcomer().eye_pixels for _ in range(60)}
        self.assertTrue(drawn <= set(range(2, 9)) and len(drawn) > 3)  # whole numbers, 2 to 8, of many kinds

    def test_sharpness_costs_energy(self):
        self.assertAlmostEqual(starting_rules()["pixel_cost"], 0.00006)  # a tenth of nature's 0.0006
        paid = {}
        for pixels in (2, 10):
            w = make_world()
            w.rules.update(step_cost=0.0, vision_cost=0.0, hearing_cost=0.0, neuron_cost=0.0, pixel_cost=0.001)
            c = add(w, 0.0, 0.0, energy=50.0)
            c.eye_pixels = pixels
            c.apply_action(("MOVE", (0.0, 0.0)), w)  # standing still: only what it is costs
            paid[pixels] = 50.0 - c.energy
        self.assertAlmostEqual(paid[10] - paid[2], 8 * 0.001)


if __name__ == "__main__":
    unittest.main()
