import unittest

from src.rules import RULES, check, starting_rules
from tests.helpers import LIVING, add, make_world


class TestRules(unittest.TestCase):
    def test_every_rule_starts_inside_its_range(self):
        names = [rule.name for rule in RULES]
        self.assertEqual(len(names), len(set(names)))  # each rule once
        for rule in RULES:
            self.assertIn(rule.group, [name for name, _ in __import__("src.rules", fromlist=["GROUPS"]).GROUPS])
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

    def test_how_a_baby_changes_is_in_its_genes(self):
        names = {r.name: r for r in RULES}
        self.assertFalse([n for n in names if n.startswith("gene.") and n.endswith(".change")])  # no fixed share left
        self.assertIn("gene.mutations.high", names)
        self.assertIn("gene.mutation_size.high", names)

    def test_every_rule_and_gene_says_what_it_means(self):
        from src.rules import GENE_MEANS, means
        for r in RULES:
            self.assertGreater(len(means(r.name)), 15, r.name)
        genes = {r.name.split(".")[1] for r in RULES if r.name.startswith("gene.") and not r.name.startswith("gene.weights")}
        self.assertEqual(genes - set(GENE_MEANS), set())

    def test_a_new_planet_has_its_numbers(self):
        from src.rules import ROOM_SLOTS, rule
        wanted = {"planet_radius": (37.0, 2000.0), "room": (600.0, 600.0), "auto_target": (480.0, 600.0),
                  "first_creatures": (200.0, 600.0), "soup": (80.0, 600.0), "obstacles": (70.0, 300.0),
                  "trees": (92.0, 400.0), "forests": (4.0, 20.0)}
        for name, (start, high) in wanted.items():
            self.assertEqual((rule(name).start, rule(name).high), (start, high), name)
        self.assertEqual(ROOM_SLOTS, 600)
        names = {r.name for r in RULES}
        self.assertFalse({"world_width", "world_height"} & names)  # a planet has no width or height

    def test_no_rule_decides_who_is_a_threat(self):
        self.assertNotIn("threat_ratio", {r.name for r in RULES})  # the creatures learn whom to flee or chase

    def test_one_reach_for_eating_and_attacking(self):
        names = {r.name for r in RULES}
        self.assertIn("reach", names)
        self.assertFalse({"eat_range", "attack_range"} & names)

    def test_every_gene_of_a_newcomer_is_drawn_from_a_range(self):
        names = {r.name for r in RULES}
        genes = {n.split(".")[1] for n in names if n.startswith("gene.") and not n.startswith("gene.weights")}
        self.assertFalse([n for n in names if n.startswith("gene.") and n.endswith(".start")])  # no fixed start left
        for gene in genes:
            self.assertIn(f"gene.{gene}.new_low", names)
            self.assertIn(f"gene.{gene}.new_high", names)

    def test_every_rule_has_a_group(self):
        from src.rules import GROUPS
        self.assertEqual([g for g, _ in GROUPS], ["costs", "world", "body", "senses", "birth", "genes", "learning", "auto", "exam", "runner"])
        for r in RULES:
            self.assertIn(r.group, dict(GROUPS))
        for name in ("learn_rollout", "learn_passes", "learn_clip", "learn_discount", "learn_credit", "learn_explore",
                     "learn_guess", "learn_spread", "auto_per_creatures", "auto_per_alive_rate", "auto_per_energy_rate",
                     "auto_max_nudges", "auto_measure", "gene.weights.share", "gene.weights.size", "gene.layers.low",
                     "gene.neurons.low"):
            self.assertIn(name, starting_rules())
        self.assertEqual((starting_rules()["gene.weights.share"], starting_rules()["gene.weights.size"]), (0.02, 0.02))

    def test_auto_reads_its_rules(self):
        from src.dial import step_for
        rules = {**starting_rules(), "auto_target": 64.0}
        self.assertEqual(step_for(72, 0.0, 0.0, 0, rules), 1)  # 8 above the target of 64: one nudge harder
        self.assertEqual(step_for(72, 0.0, 0.0, 0, {**rules, "auto_target": 72.0}), 0)
        self.assertEqual(step_for(72, 0.0, 0.0, 0, {**rules, "auto_per_creatures": 4.0}), 2)
        self.assertEqual(step_for(80, 1.0, 0.0, 0, {**rules, "auto_max_nudges": 2.0}), 2)

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
