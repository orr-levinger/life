"""Calls (spec decision 4, the owner 2026-11-26): up to 10 kinds of call, as many as its `calls` gene; each with a volume
that sets how far it carries; heard with its kind, loudness, direction and the caller's voice; no energy cost."""
import math
import unittest
from dataclasses import replace

import numpy as np

from src.learner import CALL_FIRST, N_CALLS, Learner, NetWeights
from src.rules import ROOM_SLOTS
from src.senses import CALL_INPUTS, sense
from tests.helpers import NO_FOOD, add, make_world

KIND, LOUD, DIRECTION, VOICE = CALL_INPUTS, CALL_INPUTS + N_CALLS, CALL_INPUTS + N_CALLS + 1, CALL_INPUTS + N_CALLS + 3


def heard(world, listener):
    return sense(listener, world).vector


class TestCalls(unittest.TestCase):
    def world(self):
        w = make_world(NO_FOOD)
        w.rules.update(step_cost=0.0)
        w.rules.update(vision_cost=0.0, hearing_cost=0.0, neuron_cost=0.0)
        w.rules.update({"call_reach": 10.0})
        return w

    def test_a_creature_without_calls_cannot_call(self):
        w = self.world()
        c = add(w, 10, 10)
        self.assertEqual(c.calls, 0)
        c.apply_action(("CALL", (0, 1.0)), w)
        self.assertEqual(w.calls, [])

    def test_it_makes_only_the_calls_of_its_gene(self):
        w = self.world()
        c = add(w, 10, 10)
        c.calls = 3
        c.apply_action(("CALL", (2, 1.0)), w)
        c.apply_action(("CALL", (3, 1.0)), w)  # a fourth kind: not in its genes
        self.assertEqual([(caller.id, kind) for caller, kind, _ in w.calls], [(c.id, 2)])
        self.assertEqual(c.last_action, "CALL_MISS")

    def test_calling_costs_no_energy(self):
        w = self.world()
        c = add(w, 10, 10, energy=30.0)
        c.calls = 1
        c.apply_action(("CALL", (0, 1.0)), w)
        self.assertEqual(c.energy, 30.0)

    def test_the_volume_sets_how_far_it_carries(self):
        w = self.world()
        caller, near, far = add(w, 10, 10), add(w, 14, 10), add(w, 16, 10)
        caller.calls = 2
        caller.apply_action(("CALL", (1, 0.5)), w)  # carries 0.5 × 10 = 5
        v = heard(w, near)
        self.assertEqual(list(v[KIND:KIND + N_CALLS]), [0, 1] + [0] * (N_CALLS - 2))  # the second kind
        self.assertAlmostEqual(v[LOUD], 1 - 4 / 5, places=4)  # how loud it arrived
        np.testing.assert_allclose(v[DIRECTION:DIRECTION + 2], [-1.0, 0.0], atol=1e-9)  # west of it
        self.assertFalse(heard(w, far)[KIND:LOUD].any())  # 6 away: beyond it
        self.assertFalse(heard(w, caller)[KIND:LOUD].any())  # never its own

    def test_the_voice_is_the_callers_genes(self):
        w = self.world()
        a, b, c, listener = add(w, 10, 10, size=1.0), add(w, 10, 12, size=1.02), add(w, 10, 14, size=1.9), add(w, 12, 10)
        a.diet, b.diet, c.diet = 0.3, 0.31, 0.9
        voices = []
        for caller in (a, b, c):
            w.calls = []
            caller.calls = 1
            caller.apply_action(("CALL", (0, 1.0)), w)
            voices.append(heard(w, listener)[VOICE:VOICE + 4])
        self.assertLess(np.abs(voices[0] - voices[1]).max(), 0.05)  # alike sound alike
        self.assertGreater(np.abs(voices[0] - voices[2]).max(), 0.3)
        self.assertAlmostEqual(voices[0][0], 0.3)  # its diet

    def test_no_call_means_silence(self):
        w = self.world()
        listener = add(w, 12, 10)
        self.assertFalse(heard(w, listener)[CALL_INPUTS:CALL_INPUTS + N_CALLS + 7].any())

    def test_a_newcomer_cannot_call_yet(self):
        w = make_world(seed=3)
        self.assertEqual({w.add_newcomer().calls for _ in range(20)}, {0})

    def test_a_brain_with_every_call_still_rarely_calls(self):
        learner = Learner(ROOM_SLOTS, 0)
        rng = np.random.default_rng(1)
        slots = [learner.allocate(NetWeights.random(rng), 1e-3) for _ in range(50)]
        senses = rng.normal(0.0, 1.0, (50, learner_inputs()))
        choices = []
        for _ in range(20):
            choices += [c for c, _ in learner.act(slots, senses, True, calls=[N_CALLS] * 50, learning=False)]
        self.assertLess(np.mean(np.array(choices) >= CALL_FIRST), 0.02)  # under 2 % of 1,000 steps

    def test_a_brain_without_calls_never_calls(self):
        learner = Learner(ROOM_SLOTS, 0)
        rng = np.random.default_rng(2)
        slots = [learner.allocate(NetWeights.random(rng), 1e-3) for _ in range(50)]
        for s in slots:
            learner._params["bc"][s].assign(np.r_[np.zeros(CALL_FIRST), np.full(N_CALLS, 50.0)].astype(np.float32))
        senses = rng.normal(0.0, 1.0, (50, learner_inputs()))
        choices = [c for c, _ in learner.act(slots, senses, True, calls=[0] * 50, learning=False)]
        self.assertTrue(all(c < CALL_FIRST for c in choices))  # closed, however much it would like to

    def test_the_page_gets_each_call(self):
        from src.evolve import world_now, world_view
        w = self.world()
        c = add(w, 10, 10)
        c.calls = 4
        c.apply_action(("CALL", (3, 0.5)), w)
        view = world_view(w)["creatures"][0]
        self.assertEqual((view["calls"], view["call"]), (4, [3, 5.0]))  # its fourth kind, carrying 0.5 × 10
        self.assertEqual(world_now(w)["genes"]["calls"], [4, 4, 4])


def learner_inputs():
    from src.brain import N_SENSES
    return N_SENSES


if __name__ == "__main__":
    unittest.main()
