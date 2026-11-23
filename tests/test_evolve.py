import contextlib
import csv
import io
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.evolve import parse_args, run, should_unlock, start_round
from src.learner import LEARNING_RATE, Learner, NetWeights
from src.levels import LEVELS
from src.rounds import Member, next_generation
from src.scoreboard import ROW_KEYS
from tests.helpers import make_world


def quiet_run(argv, levels=LEVELS):
    with contextlib.redirect_stdout(io.StringIO()):
        return run(parse_args(argv), levels)


def read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestEvolve(unittest.TestCase):
    def test_evolution_team_writes_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = quiet_run(["--seed", "1", "--steps", "2000", "--team", "evolution", "--out", folder])
            with open(path, newline="") as f:
                self.assertEqual(tuple(next(csv.reader(f))), ROW_KEYS)
            rows = read_rows(path)
        self.assertEqual([(r["step"], r["round"]) for r in rows], [("1000", "1"), ("2000", "2")])
        self.assertNotEqual(rows[1]["exam_ratio"], "")

    def test_same_seed_same_csv(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = quiet_run(["--seed", "3", "--steps", "1000", "--out", a]).read_text()
            second = quiet_run(["--seed", "3", "--steps", "1000", "--out", b]).read_text()
        self.assertEqual(first, second)

    def test_unlock_rule(self):
        self.assertTrue(should_unlock([1.6, 1.7], 10, 1, 5))
        self.assertFalse(should_unlock([1.6, 1.4], 10, 1, 5))
        self.assertFalse(should_unlock([None, 1.6], 10, 1, 5))
        self.assertFalse(should_unlock([1.6, 1.7], 9, 1, 5))
        self.assertFalse(should_unlock([1.6, 1.7], 10, 5, 5))
        self.assertFalse(should_unlock([1.7], 10, 1, 5))

    def test_baseline_replaces_dead(self):
        starving = (replace(LEVELS[0], max_food=0, food_per_step=0, step_cost=5.0),)
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "4", "--steps", "1000", "--policy", "random", "--out", folder],
                                       starving))
        self.assertGreater(int(rows[-1]["deaths_hunger"]), 0)
        self.assertEqual(rows[-1]["population"], "30")
        self.assertEqual(rows[-1]["round"], "")

    def test_child_inherits_inborn(self):
        learner = Learner(30, 0)
        world = make_world(LEVELS[0], learner=learner)
        rng = np.random.default_rng(0)
        members = [Member(NetWeights.random(rng), 1.0, LEARNING_RATE, world.next_id()) for _ in range(30)]
        start_round(world, members, "learner")
        for _ in range(70):
            world.step()
        inborn, learned = members[0].genome, learner.current(0)
        self.assertFalse(np.allclose(learned.w1, inborn.w1))  # learning changed slot 0's network

        scored = [(member, 30.0 - i) for i, member in enumerate(members)]
        nxt = next_generation(scored, random.Random(0), np.random.default_rng(1), world.next_id)
        self.assertIs(nxt[0].genome, inborn)
        child = nxt[8]  # the first child of the fittest member
        self.assertEqual(child.parent_id, members[0].id)
        self.assertGreater(np.mean(child.genome.w1 == inborn.w1), 0.7)
        self.assertLess(np.mean(np.isclose(child.genome.w1, learned.w1)), 0.3)

    def test_learning_team_replaces_dead(self):
        starving = (replace(LEVELS[0], max_food=0, food_per_step=0, step_cost=5.0),)
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "4", "--steps", "1000", "--team", "learning", "--out", folder],
                                       starving))
        self.assertGreater(int(rows[-1]["deaths_hunger"]), 0)
        self.assertEqual(rows[-1]["population"], "30")
        self.assertEqual(rows[-1]["round"], "")

    def test_both_team_writes_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "2000", "--team", "both", "--out", folder]))
        self.assertEqual([(r["step"], r["round"]) for r in rows], [("1000", "1"), ("2000", "2")])

    def test_same_seed_same_csv_learning(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = quiet_run(["--seed", "3", "--steps", "1000", "--team", "learning", "--out", a]).read_text()
            second = quiet_run(["--seed", "3", "--steps", "1000", "--team", "learning", "--out", b]).read_text()
        self.assertEqual(first, second)

    def test_baseline_never_unlocks(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "4000", "--policy", "random", "--out", folder]))
        self.assertEqual({r["level"] for r in rows}, {"1"})
        self.assertEqual({r["exam_ratio"] for r in rows}, {""})


if __name__ == "__main__":
    unittest.main()
