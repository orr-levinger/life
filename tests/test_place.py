"""The owner makes a creature and places it (the owner, 2026-11-27: "play with genes and see how the creature will look
like and spawn it if I wish in map where ever I click"): where it was placed, with exactly the genes given, and one of
three brains: a new one, the smartest's or a chosen creature's ("All 3 options")."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.control import Control
from src.evolve import _Knobs, world_now
from src.learner import Learner
from src.report import Reporter
from src.rules import ROOM_SLOTS
from src.species import FIRST, MIN_SPENT
from src.terrain import Obstacle, Terrain
from src.world import PlaceRefused
from tests.helpers import add, at, cap, make_world

GENES = {"size": 1.2, "grass_gut": 0.3, "meat_gut": 0.9, "gills": 0.0, "lungs": 1.0, "vision_width": 90.0,
         "vision_reach": 5.0, "hearing": 2.0, "fear_of_death": 3.0, "love_of_babies": 20.0, "learns_from_others": 0.4,
         "learning_speed": 0.001, "mutations": 2, "mutation_size": 0.05, "calls": 1, "kin": 0.5, "stealth": 0.2,
         "armour": 0.6, "layers": 2, "neurons": [12, 7]}


def learning_world():
    learner = Learner(ROOM_SLOTS, 0)
    return make_world(learner=learner, species="learner"), learner


def same(a, b):
    """Two networks with equal weights and size."""
    return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)


class TestPlace(unittest.TestCase):
    def test_placed_where_and_as_asked(self):
        w, _ = learning_world()
        spot = at(10.0, 10.0, w.radius)
        newcomers = w.newcomers
        c = w.place(list(spot * 3.0), GENES, {"kind": "new"})  # any length: the point on the ball it points to
        np.testing.assert_allclose(c.p, spot, atol=1e-12)
        for gene in ("size", "grass_gut", "meat_gut", "gills", "lungs", "vision_width", "vision_reach", "hearing",
                     "fear_of_death", "love_of_babies", "learns_from_others", "mutation_size", "kin", "stealth",
                     "armour"):
            self.assertEqual(getattr(c, gene), GENES[gene], gene)
        self.assertEqual((c.mutations, c.calls, c.generation), (2, 1, 0))
        self.assertEqual(w.newcomers, newcomers)  # not the soup's: Auto reads newcomers as "nearly died out"
        self.assertEqual(c.species, FIRST)
        self.assertIn(c, w.creatures)

    def test_a_new_brain_is_a_newcomers(self):
        w, learner = learning_world()
        c = w.place(list(at(10.0, 10.0, w.radius)), GENES, {"kind": "new"})
        self.assertEqual((c.layers, list(learner.current(c.slot).size[1:3])), (2, [12, 7]))
        self.assertAlmostEqual(learner.lr(c.slot), 0.001)

    def test_a_copied_brain_is_an_exact_copy(self):
        w, learner = learning_world()
        model = w.place(list(at(5.0, 5.0, w.radius)), {**GENES, "layers": 1, "neurons": [20]}, {"kind": "new"})
        model.species = 7
        c = w.place(list(at(10.0, 10.0, w.radius)), GENES, {"kind": "creature", "id": model.id})
        self.assertTrue(same(learner.current(c.slot), learner.current(model.slot)))
        self.assertEqual(c.layers, 1)  # the brain's own size, not the sliders'
        self.assertEqual(c.species, 7)  # the species of the creature whose brain it has

    def test_the_smartest_brain(self):
        w, learner = learning_world()
        made = [w.place(list(at(3.0 * k, 3.0, w.radius)), GENES, {"kind": "new"}) for k in range(1, 4)]
        for c, (earned, spent) in zip(made, ((20.0, 10.0), (90.0, 30.0), (50.0, MIN_SPENT / 2))):
            c.earned, c.spent = earned, spent  # the last earned most for each spent, but has hardly spent anything
        c = w.place(list(at(10.0, 10.0, w.radius)), GENES, {"kind": "smartest"})
        self.assertTrue(same(learner.current(c.slot), learner.current(made[1].slot)))

    def test_a_place_it_cannot_live_is_refused(self):
        w, _ = learning_world()
        w.terrain = Terrain([Obstacle("rock", (cap(20.0, 10.0, 2.0),)), Obstacle("pond", (cap(30.0, 10.0, 2.0),))],
                            [], w.radius)
        before = len(w.creatures)
        for spot, brain, words in ((at(20.0, 10.0, w.radius), {"kind": "new"}, "rock"),
                                   (at(30.0, 10.0, w.radius), {"kind": "new"}, "gills"),
                                   (at(10.0, 10.0, w.radius), {"kind": "creature", "id": 999}, "no longer")):
            with self.assertRaises(PlaceRefused) as refused:
                w.place(list(spot), GENES, brain)
            self.assertIn(words, str(refused.exception))
        w.rules["room"] = float(len(w.creatures))
        with self.assertRaises(PlaceRefused) as refused:
            w.place(list(at(10.0, 10.0, w.radius)), GENES, {"kind": "new"})
        self.assertIn("full", str(refused.exception))
        self.assertEqual(len(w.creatures), before)

    def test_the_runner_places_once(self):
        w, _ = learning_world()
        add(w, 1.0, 1.0)
        place = {"seq": 5, "spot": list(at(10.0, 10.0, w.radius)), "genes": GENES, "brain": {"kind": "new"}}
        control = Control("auto", None, False, place=place)
        with tempfile.TemporaryDirectory() as tmp:
            reporter = Reporter(Path(tmp))
            knobs = _Knobs(control, reporter)
            knobs.take(control, None, w)
            knobs.take(control, None, w)  # read again: placed once
            events = [json.loads(line) for line in (Path(tmp) / "report" / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len(w.creatures), 2)
        placed = world_now(w)["placed"]
        self.assertEqual((placed["seq"], placed["id"]), (5, w.creatures[-1].id))
        self.assertEqual([e["kind"] for e in events], ["placed"])


if __name__ == "__main__":
    unittest.main()
