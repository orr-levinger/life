"""Stealth (spec decision 7): a stealthy body is seen and heard only from closer, and costs more to keep."""
import unittest
from dataclasses import replace

from src.senses import perceived, sense
from tests.helpers import NO_FOOD, add, make_world


class TestStealth(unittest.TestCase):
    def test_a_stealthy_body_is_seen_only_from_closer(self):
        w = make_world(NO_FOOD)
        looker = add(w, 10.0, 10.0)
        looker.vision_width, looker.vision_reach, looker.facing, looker.hearing = 120.0, 4.0, (1.0, 0.0), 0.0
        hidden = add(w, 12.0, 10.0)  # 2 away: within 4, beyond 4 × (1 - 0.7)
        hidden.stealth = 1.0
        self.assertFalse(perceived([looker], [hidden], w)[0, 0])
        hidden.x = 11.0  # 1 away: within 1.2
        self.assertTrue(perceived([looker], [hidden], w)[0, 0])

    def test_a_stealthy_mover_is_heard_only_from_closer(self):
        w = make_world(NO_FOOD)
        listener = add(w, 10.0, 10.0)
        listener.hearing, listener.vision_reach = 4.0, 0.1
        mover = add(w, 12.0, 10.0)
        mover.moved, mover.stealth = True, 0.5  # heard within 4 × (1 - 0.35) = 2.6
        self.assertIsNotNone(sense(listener, w).heard)
        mover.stealth = 1.0  # within 1.2
        self.assertIsNone(sense(listener, w).heard)

    def test_stealth_costs_more_to_keep(self):
        w = make_world(NO_FOOD)
        w.settings = replace(w.settings, step_cost=0.1)
        w.rules.update({"vision_cost": 0.0, "hearing_cost": 0.0, "neuron_cost": 0.0, "eat_miss": 0.0})
        plain, stealthy = add(w, 10, 10, energy=20.0), add(w, 20, 10, energy=20.0)
        stealthy.stealth = 1.0
        for c in (plain, stealthy):
            c.apply_action(("EAT", None), w)
        self.assertAlmostEqual(20.0 - stealthy.energy, (20.0 - plain.energy) * (1 + w.rules["stealth_cost"]))
        self.assertEqual((w.rules["stealth_strength"], w.rules["stealth_cost"]), (0.7, 0.3))

    def test_newcomers_draw_their_stealth(self):
        w = make_world(seed=7)
        values = [w.add_newcomer().stealth for _ in range(30)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        self.assertGreater(max(values) - min(values), 0.5)

    def test_the_page_gets_each_ones_stealth(self):
        from src.evolve import world_now, world_view
        w = make_world(NO_FOOD)
        add(w, 10, 10).stealth = 0.3
        self.assertEqual(world_view(w)["creatures"][0]["stealth"], 0.3)
        self.assertEqual(world_now(w)["genes"]["stealth"], [0.3, 0.3, 0.3])


if __name__ == "__main__":
    unittest.main()
