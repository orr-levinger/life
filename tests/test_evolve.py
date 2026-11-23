import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.dial import Dial
from src.evolve import FIRST_CREATURES, parse_args, run
from src.scoreboard import ROW_KEYS

EXAM_COLUMNS = ("intelligence", "wanderers")


def quiet_run(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return run(parse_args(argv))


def read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def deaths(row) -> int:
    return int(row["deaths_hunger"]) + int(row["deaths_attack"]) + int(row["deaths_age"])


def reviewing_as_if(died: str):
    """The dial's review, told that nobody died in the window ("nobody") or that everybody did ("everybody")."""
    real = Dial.review

    def review(dial, deaths, alive, newcomers):
        return real(dial, 0 if died == "nobody" else alive, alive, 0)
    return mock.patch.object(Dial, "review", review)


class TestEvolve(unittest.TestCase):
    def test_instinct_run_writes_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = quiet_run(["--seed", "1", "--steps", "2000", "--species", "instinct", "--out", folder])
            with open(path, newline="") as f:
                self.assertEqual(tuple(next(csv.reader(f))), ROW_KEYS)
            rows = read_rows(path)
            seconds = float((Path(folder) / "time.txt").read_text())
        self.assertEqual([r["step"] for r in rows], ["1000", "2000"])
        self.assertEqual(rows[0]["level"], "0")
        self.assertGreater(int(rows[0]["newcomers"]), 0)  # the first creatures are newcomers
        for column in EXAM_COLUMNS:
            self.assertEqual(rows[0][column], "")  # the exam is every 2,000 steps
            self.assertNotEqual(rows[1][column], "")
        self.assertGreater(seconds, 0.0)

    def test_run_writes_its_record(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "4", "--steps", "1000", "--species", "instinct", "--out", folder])
            record = json.loads((Path(folder) / "run.json").read_text())
        self.assertEqual((record["seed"], record["kind"], record["steps"]), (4, "instinct", 1000))
        self.assertRegex(record["started"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")

    def test_every_run_gets_its_own_dated_folder(self):
        self.assertRegex(parse_args(["--seed", "7", "--species", "learning"]).out, r"^runs/\d{8}-\d{6}-7-learning$")
        self.assertRegex(parse_args(["--seed", "7", "--species", "learning", "--inherit", "inborn"]).out,
                         r"-7-learning-inborn$")
        self.assertRegex(parse_args(["--seed", "7", "--policy", "rules", "--level", "3"]).out, r"-7-rules-L3$")

    def test_same_seed_same_csv(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = quiet_run(["--seed", "3", "--steps", "1000", "--species", "instinct", "--out", a]).read_text()
            second = quiet_run(["--seed", "3", "--steps", "1000", "--species", "instinct", "--out", b]).read_text()
        self.assertEqual(first, second)

    def test_the_dial_reviews_every_window(self):
        reviews = []
        real = Dial.review

        def review(dial, deaths, alive, newcomers):
            verdict = real(dial, deaths, alive, newcomers)
            reviews.append((deaths, alive, newcomers, dial.level))
            return verdict

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(Dial, "review", review):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--out", folder]))
        self.assertEqual([(died, alive) for died, alive, _, _ in reviews],
                         [(deaths(row), int(row["population"])) for row in rows])
        soup = [int(row["newcomers"]) for row in rows]
        soup[0] -= FIRST_CREATURES  # the first creatures are not the soup's
        self.assertEqual([newcomers for _, _, newcomers, _ in reviews], soup)
        self.assertEqual([int(row["level"]) for row in rows[1:]], [level for *_, level in reviews[:-1]])

    def test_dial_turns_when_nobody_dies(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("nobody"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--out", folder]))
        self.assertEqual([row["level"] for row in rows], ["0", "1", "2"])
        self.assertLess(float(rows[1]["max_food"]), float(rows[0]["max_food"]))  # the first nudge: less food at once
        self.assertLess(float(rows[2]["food_energy"]), float(rows[1]["food_energy"]))  # the second: less in each food
        self.assertEqual(rows[2]["max_food"], rows[1]["max_food"])  # one setting at a time

    def test_dial_takes_nudges_back_when_too_many_die(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("everybody"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--level", "2",
                                        "--out", folder]))
        self.assertEqual([row["level"] for row in rows], ["2", "1", "0"])

    def test_dial_stops_at_the_max_level(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("nobody"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--max-level", "1",
                                        "--out", folder]))
        self.assertEqual([row["level"] for row in rows], ["0", "1", "1"])

    def test_exam_does_not_change_the_world(self):
        argv = ["--seed", "2", "--steps", "3000", "--species", "instinct", "--out"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            examined = read_rows(quiet_run(argv + [a]))
            with mock.patch("src.evolve.EXAM_EVERY", 10 ** 9):
                unexamined = read_rows(quiet_run(argv + [b]))
        self.assertNotEqual(examined[1]["intelligence"], "")
        self.assertEqual({row["intelligence"] for row in unexamined}, {""})
        for with_exam, without in zip(examined, unexamined):
            for column in EXAM_COLUMNS:
                with_exam.pop(column), without.pop(column)
        self.assertEqual(examined, unexamined)

    def test_yardstick_world_stays_at_its_level(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--policy", "random", "--level", "2",
                                        "--out", folder]))
        self.assertEqual({row["level"] for row in rows}, {"2"})
        self.assertEqual({row["intelligence"] for row in rows}, {""})

    def test_learning_run_writes_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "2000", "--species", "learning", "--out", folder]))
        self.assertEqual([row["step"] for row in rows], ["1000", "2000"])
        self.assertEqual(rows[0]["level"], "0")
        self.assertGreaterEqual(int(rows[-1]["population"]), 10)
        self.assertNotEqual(rows[1]["intelligence"], "")
        self.assertNotEqual(rows[1]["energy"], "")

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
