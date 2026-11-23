import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.evolve import parse_args, run, should_unlock
from src.exam import ExamResult
from src.scoreboard import ROW_KEYS


def quiet_run(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return run(parse_args(argv))


def read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestEvolve(unittest.TestCase):
    def test_instinct_run_writes_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = quiet_run(["--seed", "1", "--steps", "2000", "--species", "instinct", "--out", folder])
            with open(path, newline="") as f:
                self.assertEqual(tuple(next(csv.reader(f))), ROW_KEYS)
            rows = read_rows(path)
            seconds = float((Path(folder) / "time.txt").read_text())
        self.assertEqual([(r["step"], r["stage"]) for r in rows], [("1000", "1"), ("2000", "1")])
        self.assertGreater(int(rows[0]["newcomers"]), 0)  # the first creatures are newcomers
        self.assertNotEqual(rows[1]["exam_ratio"], "")
        self.assertGreater(seconds, 0.0)

    def test_same_seed_same_csv(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = quiet_run(["--seed", "3", "--steps", "1000", "--species", "instinct", "--out", a]).read_text()
            second = quiet_run(["--seed", "3", "--steps", "1000", "--species", "instinct", "--out", b]).read_text()
        self.assertEqual(first, second)

    def test_unlock_rule(self):
        self.assertTrue(should_unlock([1.6, 1.7], 10, 1, 4))
        self.assertFalse(should_unlock([1.6, 1.4], 10, 1, 4))
        self.assertFalse(should_unlock([None, 1.6], 10, 1, 4))
        self.assertFalse(should_unlock([1.6, 1.7], 9, 1, 4))
        self.assertFalse(should_unlock([1.6, 1.7], 10, 4, 4))
        self.assertFalse(should_unlock([1.7], 10, 1, 4))

    def test_stage_goes_up_after_two_passed_exams(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch("src.evolve.run_exam", return_value=ExamResult(0.4, 0.2, 2.0)):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "4000", "--species", "instinct", "--out", folder]))
        self.assertEqual([r["stage"] for r in rows], ["1", "1", "1", "2"])

    def test_baseline_never_changes_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "4000", "--policy", "random", "--out", folder]))
        self.assertEqual({r["stage"] for r in rows}, {"1"})
        self.assertEqual({r["exam_ratio"] for r in rows}, {""})

    def test_learning_run_starts_in_the_garden(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "2000", "--species", "learning", "--max-stage", "0",
                                        "--out", folder]))
        self.assertEqual([r["stage"] for r in rows], ["0", "0"])
        self.assertEqual({(r["deaths_hunger"], r["deaths_attack"], r["deaths_age"]) for r in rows}, {("0", "0", "0")})
        self.assertGreaterEqual(int(rows[-1]["population"]), 30)
        self.assertNotEqual(rows[1]["exam_ratio"], "")

    def test_same_seed_same_csv_learning(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = quiet_run(["--seed", "3", "--steps", "1000", "--species", "learning", "--out", a]).read_text()
            second = quiet_run(["--seed", "3", "--steps", "1000", "--species", "learning", "--out", b]).read_text()
        self.assertEqual(first, second)

    def test_species_or_policy_is_required(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--seed", "1"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--seed", "1", "--species", "instinct", "--policy", "random"])


if __name__ == "__main__":
    unittest.main()
