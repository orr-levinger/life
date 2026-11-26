"""Care for kin (spec decision 5): a share of what happens to the creatures it perceives counts in its own reward, by how
alike their genes are; family is seen and can be attacked like anyone."""
import unittest
from unittest import mock

import numpy as np

from src.brain import N_SENSES
from src.learner import Learner
from src.rules import ROOM_SLOTS
from src.senses import KIN_INPUT, sense
from src.world import similarity
from tests.helpers import NO_FOOD, NO_FOOD_ATTACKS, add, make_world


def opposite(a, b, rules):
    """`b` as unlike `a` as the genes allow."""
    for gene, attr in (("size", "size"), ("diet", "diet"), ("water", "water"), ("stealth", "stealth"), ("armour", "armour"), ("vision_width", "vision_width"),
                       ("vision_reach", "vision_reach"), ("hearing", "hearing")):
        low, high = rules[f"gene.{gene}.low"], rules[f"gene.{gene}.high"]
        setattr(a, attr, low)
        setattr(b, attr, high)


class TestKin(unittest.TestCase):
    def test_similarity_is_one_for_twins_and_zero_for_opposites(self):
        w = make_world()
        a, b = add(w, 10, 10), add(w, 12, 10)
        self.assertEqual(similarity(a, b, w.rules), 1.0)
        opposite(a, b, w.rules)
        self.assertEqual(similarity(a, b, w.rules), 0.0)

    def reward(self, kin, alike):
        """The watcher's reward when a creature it sees loses 10 energy (its own energy unchanged)."""
        learner = Learner(ROOM_SLOTS, 0)
        w = make_world(NO_FOOD, learner=learner, species="learner")
        watcher, other = w.add_newcomer(), w.add_newcomer()
        watcher.x, watcher.y, other.x, other.y = 10.0, 10.0, 12.0, 10.0
        watcher.facing, watcher.vision_width, watcher.vision_reach = (1.0, 0.0), 120.0, 5.0
        for gene in ("size", "diet", "water", "vision_width", "vision_reach", "hearing"):
            setattr(other, gene, getattr(watcher, gene))
        if not alike:
            opposite(watcher, other, w.rules)
        watcher.vision_width, watcher.vision_reach = 120.0, 5.0  # it sees the other either way
        if not alike:
            other.vision_width, other.vision_reach = 30.0, 1.0  # and the other's eyes stay unlike its own
        watcher.kin = kin
        learner.act([watcher.slot, other.slot], np.zeros((2, N_SENSES)), True)
        with mock.patch.object(learner, "update"):
            w._teach([watcher, other], [watcher.energy, other.energy + 10.0])
        return learner._buffers[watcher.slot][-1]["reward"], similarity(watcher, other, w.rules)

    def test_a_gene_with_no_range_is_left_out(self):
        w = make_world()
        a, b = add(w, 10, 10), add(w, 12, 10, size=1.5)
        without = similarity(a, b, {k: v for k, v in w.rules.items() if not k.startswith("gene.armour")})
        w.rules["gene.armour.low"] = w.rules["gene.armour.high"] = 0.0  # the owner turns armour off
        self.assertEqual(similarity(a, b, w.rules), without)  # no division by zero: as if there were no armour
        a.policy = "learner"
        a.kin = 1.0
        with mock.patch("src.world.perceived", return_value=np.ones((2, 1), dtype=bool)):
            shares = w._kin_shares([a, b], [a.energy, b.energy + 5.0])
        self.assertTrue(np.isfinite(shares[id(a)]))

    def test_it_shares_what_happens_to_one_like_it(self):
        reward, alike = self.reward(kin=1.0, alike=True)
        self.assertGreater(alike, 0.9)
        self.assertAlmostEqual(reward, 1.0 * alike * -10.0)

    def test_nothing_from_one_unlike_it(self):
        reward, alike = self.reward(kin=1.0, alike=False)
        self.assertLess(alike, 0.25)
        self.assertAlmostEqual(reward, alike * -10.0)

    def test_no_care_no_share(self):
        self.assertEqual(self.reward(kin=0.0, alike=True)[0], 0.0)

    def test_family_can_be_attacked(self):
        w = make_world(NO_FOOD_ATTACKS)
        parent = add(w, 10, 10)
        child = add(w, 11, 10, parent_id=parent.id, energy=40.0)
        parent.apply_action(("ATTACK", child), w)
        self.assertEqual(parent.last_action, "ATTACK")
        self.assertLess(child.energy, 40.0)

    def test_family_is_seen_with_how_alike_it_is(self):
        w = make_world(NO_FOOD)
        parent = add(w, 10, 10)
        parent.vision_width, parent.vision_reach, parent.facing = 120.0, 5.0, (1.0, 0.0)
        child = add(w, 12, 10, parent_id=parent.id)
        seen = sense(parent, w)
        self.assertIs(seen.other, child)
        self.assertAlmostEqual(seen.vector[KIN_INPUT], similarity(parent, child, w.rules))
        self.assertLess(KIN_INPUT, N_SENSES)

    def test_newcomers_draw_how_much_they_care(self):
        w = make_world(seed=5)
        kins = [w.add_newcomer().kin for _ in range(30)]
        self.assertTrue(all(0.0 <= k <= 1.0 for k in kins))
        self.assertGreater(max(kins) - min(kins), 0.5)


    def test_the_page_gets_how_much_each_cares(self):
        from src.evolve import world_now, world_view
        w = make_world(NO_FOOD)
        add(w, 10, 10).kin = 0.4
        self.assertEqual(world_view(w)["creatures"][0]["kin"], 0.4)
        self.assertEqual(world_now(w)["genes"]["kin"], [0.4, 0.4, 0.4])


if __name__ == "__main__":
    unittest.main()
