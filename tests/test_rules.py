import unittest

from src.rules import RULES, check, starting_rules
from tests.helpers import LIVING, add, make_world


class TestRules(unittest.TestCase):
    def test_every_rule_starts_inside_its_range(self):
        names = [rule.name for rule in RULES]
        self.assertEqual(len(names), len(set(names)))  # each rule once
        for rule in RULES:
            self.assertIn(rule.table, ("costs", "values", "genes"))
            self.assertTrue(rule.low <= rule.start <= rule.high, rule.name)
        self.assertEqual(starting_rules(), {rule.name: rule.start for rule in RULES})

    def test_a_value_outside_its_range_is_refused(self):
        rules = starting_rules()
        self.assertIsNone(check(rules, "eat_miss", 0.3))
        self.assertIn("at most 5", check(rules, "eat_miss", 6.0))
        self.assertIn("at least 0", check(rules, "eat_miss", -1.0))
        self.assertIn("lowest", check(rules, "gene.size.low", 3.0))  # above the gene's highest (2)
        self.assertIsNone(check(rules, "gene.size.low", 1.0))
        self.assertIn("unknown", check(rules, "no_such_rule", 1.0))

    def test_a_cone_is_at_most_120(self):
        from src.rules import rule
        self.assertEqual((rule("gene.vision_width.high").start, rule("gene.vision_width.high").high), (120.0, 120.0))
        self.assertEqual(rule("gene.vision_width.new_high").start, 120.0)
        self.assertIn("at most 120", check(starting_rules(), "gene.vision_width.high", 200.0))
        self.assertEqual(rule("gene.vision_reach.high").start, 10.0)  # it sees farther than it hears (4)
        w = make_world(seed=2)
        self.assertTrue(all(w.add_newcomer().vision_width <= 120.0 for _ in range(40)))

    def test_a_baby_gene_changes_by_at_most_5_percent(self):
        changes = {r.name: r.start for r in RULES if r.name.startswith("gene.") and r.name.endswith(".change")}
        self.assertEqual(set(changes.values()), {0.05})
        self.assertGreaterEqual(len(changes), 8)

    def test_the_world_reads_its_rules(self):
        w = make_world()
        c = add(w, 10, 10)
        w.rules["eat_miss"] = 1.0
        before = c.energy
        c.apply_action(("EAT", None), w)  # nothing within reach
        self.assertAlmostEqual(before - c.energy, 1.0 + LIVING)
        w.rules["cruise_speed"] = 0.5
        self.assertAlmostEqual(c.velocity, 0.5)  # a speed reaches everyone at once


if __name__ == "__main__":
    unittest.main()
