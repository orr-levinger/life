import contextlib
import csv
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.evolve import parse_args, run, should_unlock
from src.levels import LEVELS
from src.scoreboard import ROW_KEYS


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

    def test_baseline_never_unlocks(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "4000", "--policy", "random", "--out", folder]))
        self.assertEqual({r["level"] for r in rows}, {"1"})
        self.assertEqual({r["exam_ratio"] for r in rows}, {""})


if __name__ == "__main__":
    unittest.main()
