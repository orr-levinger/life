import unittest
from dataclasses import replace

from tests.helpers import add, make_world

FREE = dict(step_cost=0.0, vision_cost=0.0, hearing_cost=0.0, neuron_cost=0.0, eat_miss=0.0, bump_cost=0.0)


class TestCostsAreKnobs(unittest.TestCase):
    def test_a_sprint_costs_what_the_knob_says(self):  # the sprint's price is the world's knob, not a fixed rule
        paid = {}
        for power in (0.0, 1.0, 2.0):
            w = make_world()
            w.rules.update(move_cost=1.0, sprint_power=power, **FREE)
            c = add(w, 0.0, 0.0, size=1.0, energy=50.0)
            c.apply_action(("MOVE", (1.0, 0.0)), w)  # flat out
            top, cruise = c.top_speed, c.velocity
            paid[power] = 50.0 - c.energy
            self.assertAlmostEqual(paid[power], top * 1.0 * (top / cruise) ** power, places=9)
        self.assertLess(paid[0.0], paid[1.0])
        self.assertLess(paid[1.0], paid[2.0])

    def test_living_costs_are_the_worlds_knobs(self):
        w = make_world()
        w.rules.update(**{**FREE, "step_cost": 0.3, "neuron_cost": 0.0})
        c = add(w, 0.0, 0.0, size=1.0, energy=50.0)
        c.apply_action(("MOVE", (0.0, 0.0)), w)  # standing still: only living costs
        self.assertAlmostEqual(50.0 - c.energy, 0.3 * (1.0 + c.stealth * w.rules["stealth_cost"])
                               * (1.0 + c.armour * w.rules["armour_cost"]))


if __name__ == "__main__":
    unittest.main()
