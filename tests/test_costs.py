"""Costs are fixed by nature and the owner's to turn, never Auto's (the owner, 2026-11-27)."""
import math
import unittest
from dataclasses import fields

from src.dial import KNOBS
from src.rules import RULES, means
from src.settings import Settings
from tests.helpers import add, make_world

NATURE = {"step_cost": 0.1, "neuron_cost": 0.0006, "vision_cost": 0.025 / (math.pi * 9.0),
          "hearing_cost": 0.025 / (math.pi * 9.0), "move_cost": 0.4, "sprint_power": 2.0, "eat_miss": 0.1,
          "bump_cost": 0.2, "attack_cost": 1.0, "attack_damage": 10.0, "stealth_cost": 0.3, "armour_cost": 0.3}
FREE = {name: 0.0 for name in NATURE}


class TestCostsAreRules(unittest.TestCase):
    def test_the_costs_are_rules_with_these_values(self):
        by_name = {r.name: r for r in RULES}
        for name, value in NATURE.items():
            r = by_name[name]
            top = 4.0 if name == "sprint_power" else 3.0 * value  # the max: what the smartest can bear
            self.assertEqual((r.group, r.low), ("costs", 0.0), name)
            self.assertAlmostEqual(r.start, value, msg=name)
            self.assertAlmostEqual(r.high, top, msg=name)
            self.assertGreater(len(means(name)), 15, name)
            self.assertNotIn(f"knob.{name}.easy", by_name)  # a cost's range is its own rule's

    def test_auto_turns_only_food_and_lifespan(self):
        self.assertEqual({k.name for k in KNOBS}, {"max_food", "food_energy", "food_per_step", "fruit_rate", "lifespan"})
        self.assertFalse(set(NATURE) & {f.name for f in fields(Settings)})  # no cost is a setting any more

    def test_a_cost_change_reaches_the_running_world(self):
        w = make_world()
        w.change_rules({**FREE, "move_cost": 1.0})
        c = add(w, 0.0, 0.0, energy=50.0)
        c.apply_action(("MOVE", (0.2, 0.0)), w)
        self.assertAlmostEqual(50.0 - c.energy, 0.2 * 1.0 * c.size)


class TestGrowingUp(unittest.TestCase):
    def test_a_new_world_grows_into_its_costs(self):  # a tenth at first, all of it by generation 10, never back
        w = make_world()
        w.grown = 0.1
        c = add(w, 0.0, 0.0)
        for generation, share in ((0, 0.1), (5, 0.55), (10, 1.0), (14, 1.0), (0, 1.0)):
            c.generation = generation
            w.grow_up()
            self.assertAlmostEqual(w.grown, share, msg=generation)
        self.assertAlmostEqual(w.cost("step_cost"), w.rules["step_cost"])

    def test_a_new_planet_starts_at_a_tenth(self):
        from src.evolve import _new_world, parse_args
        from tests.helpers import small_world
        with small_world():
            world, _ = _new_world(parse_args(["--seed", "1", "--species", "instinct", "--steps", "10"]), {})
        self.assertEqual(world.grown, 0.1)
        self.assertAlmostEqual(world.cost("step_cost"), 0.1 * world.rules["step_cost"])


if __name__ == "__main__":
    unittest.main()
