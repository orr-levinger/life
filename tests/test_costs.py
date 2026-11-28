"""What everything costs: nature sets each cost's top, Auto turns it between 0 and there, the owner may set it any time
(the owner, 2026-11-27)."""
import math
import unittest
from dataclasses import fields, replace

from src.dial import KNOBS, Dial, difficulty
from src.rules import RULES, means, starting_rules
from src.settings import START, Settings
from tests.helpers import add, make_world

NATURE = {"step_cost": 0.1, "neuron_cost": 0.0006, "vision_cost": 0.025 / (math.pi * 9.0),
          "hearing_cost": 0.025 / (math.pi * 9.0), "move_cost": 0.4, "sprint_power": 1.0, "eat_miss": 0.1,
          "bump_cost": 0.2, "attack_cost": 1.0, "attack_damage": 10.0, "stealth_cost": 0.3, "armour_cost": 0.3,
          "pixel_cost": 0.0006,  # each pixel of its eye (2026-11-27)
          "gut_cost": 0.15, "breath_cost": 0.15}  # each gut, gill and lung (2026-11-27)
FREE = {name: 0.0 for name in NATURE}
RULES_NOW = starting_rules()
HARDER = (round(RULES_NOW["auto_target"]) + 40, 0.0, 0.0, 0)  # many above the target, steady: harder


def dial_turning(name):
    """A dial whose next nudge goes to the cost `name` (hunters on: no nudge goes to them)."""
    dial = Dial(replace(START, attacks=True))
    dial._next_harder = dial._next_easier = [k.name for k in KNOBS].index(name)
    return dial


class TestCostsAreRules(unittest.TestCase):
    def test_a_cost_goes_from_0_to_nature_starting_at_a_tenth(self):
        by_name = {r.name: r for r in RULES}
        for name, value in NATURE.items():
            r = by_name[name]
            self.assertEqual((r.group, r.low), ("costs", 0.0), name)
            self.assertAlmostEqual(r.high, value, msg=name)  # never above nature
            self.assertAlmostEqual(r.start, value / 10.0, msg=name)  # a new world: a tenth
            self.assertGreater(len(means(name)), 15, name)
        self.assertFalse(set(NATURE) & {f.name for f in fields(Settings)})  # a cost is a rule, not a setting

    def test_auto_turns_the_costs_with_food_and_lifespan(self):
        self.assertEqual({k.name for k in KNOBS},
                         {"max_food", "food_energy", "food_per_step", "fruit_rate", "lifespan", *NATURE})

    def test_auto_raises_a_cost_up_to_nature_and_no_further(self):
        dial = dial_turning("move_cost")
        seen = []
        for _ in range(4000):
            dial._harder()
            seen.append(dial.rules["move_cost"])
        self.assertGreater(seen[0], 0.04)  # up from a tenth
        self.assertAlmostEqual(max(seen), 0.4)  # nature's, and never above

    def test_auto_lowers_a_cost_to_zero(self):
        dial = dial_turning("eat_miss")
        for _ in range(4000):
            dial._easier()
        self.assertEqual(dial.rules["eat_miss"], 0.0)

    def test_auto_goes_on_from_the_owners_cost(self):
        dial = dial_turning("step_cost")
        dial.rules["step_cost"] = 0.07  # the owner set it (the Rules page or its slider)
        dial.review(*HARDER)
        self.assertGreater(dial.rules["step_cost"], 0.07)
        self.assertLess(dial.rules["step_cost"], 0.08)

    def test_auto_takes_back_its_own_cost_nudge(self):
        dial = dial_turning("bump_cost")
        before = dial.rules["bump_cost"]
        dial._harder()
        dial._easier()
        self.assertEqual(dial.rules["bump_cost"], before)

    def test_difficulty_counts_cost_nudges(self):
        dial = dial_turning("attack_cost")
        start = dial.level
        dial._harder()
        self.assertAlmostEqual(dial.level - start, 1.0)
        self.assertAlmostEqual(difficulty(START, 0.05, 0.01, starting_rules()), 0.0)

    def test_a_cost_change_reaches_the_running_world(self):
        w = make_world()
        w.change_rules({**FREE, "move_cost": 0.3})
        c = add(w, 0.0, 0.0, energy=50.0)
        c.apply_action(("MOVE", (0.2, 0.0)), w)
        self.assertAlmostEqual(50.0 - c.energy, 0.2 * 0.3 * c.size)

    def test_a_cost_above_nature_comes_down_at_autos_next_look(self):  # an old save's (today's moving was 0.5)
        dial = Dial(replace(START, attacks=True))
        dial.rules.update(move_cost=0.5, bump_cost=0.5, step_cost=0.05)
        dial.review(round(RULES_NOW["auto_target"]), 0.0, 0.0, 0)  # at the target: no nudge, but within the ends
        self.assertEqual((dial.rules["move_cost"], dial.rules["bump_cost"], dial.rules["step_cost"]), (0.4, 0.2, 0.05))

    def test_an_earlier_cost_change_is_not_applied_again(self):  # Auto turned the cost since: it stays as Auto left it
        from pathlib import Path

        from src.control import Control
        from src.evolve import _Knobs
        from src.report import Reporter
        w = make_world()
        knobs = _Knobs(Control("auto", None, False), Reporter(Path("unused")))  # writes only when a creature is placed
        knobs.take(Control("auto", None, False, rules={"move_cost": 0.3}), Dial(), w)
        w.rules["move_cost"] = 0.32  # Auto's nudge
        knobs.take(Control("auto", None, False, rules={"move_cost": 0.3, "tank": 120.0}), Dial(), w)
        self.assertEqual((w.rules["move_cost"], w.rules["tank"]), (0.32, 120.0))


if __name__ == "__main__":
    unittest.main()
