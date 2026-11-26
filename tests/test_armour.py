"""Armour (spec decision 8): a hit takes less from an armoured body, which is slower and costs more to keep."""
import unittest
from dataclasses import replace

from tests.helpers import NO_FOOD, NO_FOOD_ATTACKS, add, make_world


class TestArmour(unittest.TestCase):
    def test_a_hit_takes_less_from_armour(self):
        w = make_world(NO_FOOD_ATTACKS)
        hunter, plain, armoured = add(w, 10, 10), add(w, 11, 10, energy=50.0), add(w, 10, 11, energy=50.0)
        armoured.armour = 1.0
        hunter.apply_action(("ATTACK", plain), w)
        hunter.apply_action(("ATTACK", armoured), w)
        self.assertAlmostEqual(50.0 - armoured.energy, (50.0 - plain.energy) * (1 - w.rules["armour_strength"]))

    def test_armour_is_slower(self):
        w = make_world()
        plain, armoured = add(w, 10, 10), add(w, 20, 10)
        armoured.armour = 1.0
        slow = 1 - w.rules["armour_slow"]
        self.assertAlmostEqual(armoured.velocity, plain.velocity * slow)
        self.assertAlmostEqual(armoured.top_speed, plain.top_speed * slow)

    def test_armour_costs_more_to_keep(self):
        w = make_world(NO_FOOD)
        w.settings = replace(w.settings, step_cost=0.1)
        w.rules.update({"vision_cost": 0.0, "hearing_cost": 0.0, "neuron_cost": 0.0, "eat_miss": 0.0})
        plain, armoured = add(w, 10, 10, energy=20.0), add(w, 20, 10, energy=20.0)
        armoured.armour = 1.0
        for c in (plain, armoured):
            c.apply_action(("EAT", None), w)
        self.assertAlmostEqual(20.0 - armoured.energy, (20.0 - plain.energy) * (1 + w.rules["armour_cost"]))
        self.assertEqual((w.rules["armour_strength"], w.rules["armour_slow"], w.rules["armour_cost"]), (0.7, 0.3, 0.3))

    def test_newcomers_draw_their_armour(self):
        w = make_world(seed=8)
        values = [w.add_newcomer().armour for _ in range(30)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        self.assertGreater(max(values) - min(values), 0.5)

    def test_the_page_gets_each_ones_armour(self):
        from src.evolve import world_now, world_view
        w = make_world(NO_FOOD)
        add(w, 10, 10).armour = 0.2
        self.assertEqual(world_view(w)["creatures"][0]["armour"], 0.2)
        self.assertEqual(world_now(w)["genes"]["armour"], [0.2, 0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
