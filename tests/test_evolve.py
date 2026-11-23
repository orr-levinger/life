import contextlib
import csv
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src import control as control_module
from src.control import Control, read_control, write_control
from src.dial import FEW, MANY, WELL_FED, Dial, value, with_value
from src.evolve import CONTROL_EVERY, FIRST_CREATURES, ROW_EVERY, parse_args, run
from src.scoreboard import ROW_KEYS
from src.settings import START

EXAM_COLUMNS = ("intelligence", "wanderers")
ROWS_PER_REVIEW = 4  # a row every 250 steps, a review every 1,000


def quiet_run(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return run(parse_args(argv))


def read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def levels(rows):
    return [round(float(row["level"]), 2) for row in rows]


def at_reviews(rows):
    """The rows written at Auto's reviews (steps 1,000, 2,000 …)."""
    return [row for row in rows if int(row["step"]) % 1000 == 0]


def reviewing_as_if(life: str):
    """Auto's review, told that the creatures are many and well fed ("easy") or few ("hard")."""
    real = Dial.review

    def review(dial, alive, energy, newcomers):
        return real(dial, MANY, WELL_FED, 0) if life == "easy" else real(dial, FEW - 1, WELL_FED, 0)
    return mock.patch.object(Dial, "review", review)


def owner_writes_at(step: int, control: Control):
    """The owner's control file appears while the run goes, at `step`."""
    real, looks = control_module.read_control, []

    def read(folder):
        looks.append(folder)
        if (len(looks) - 1) * CONTROL_EVERY == step:  # the first look is before step 1
            write_control(folder, control)
        return real(folder)
    return mock.patch("src.evolve.read_control", read)


def watched(folder, speed=10.0, until=None):
    """The owner jumped into this world (until `until`, a time; ten minutes from now when not given)."""
    lease = time.time() + 600 if until is None else until
    write_control(Path(folder), Control("auto", None, False, watch_until=lease, watch_speed=speed))


def saved_steps(folder):
    return sorted(int(p.stem.split("-")[1]) for p in (Path(folder) / "saves").glob("step-*.npz"))


def no_waiting():
    return mock.patch("src.evolve.time.sleep")


class TestEvolve(unittest.TestCase):
    def test_instinct_run_writes_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = quiet_run(["--seed", "1", "--steps", "2000", "--species", "instinct", "--out", folder])
            with open(path, newline="") as f:
                self.assertEqual(tuple(next(csv.reader(f))), ROW_KEYS)
            rows = read_rows(path)
            seconds = float((Path(folder) / "time.txt").read_text())
        self.assertEqual(levels(rows[:1]), [0])
        self.assertGreater(int(rows[0]["newcomers"]), 0)  # the first creatures are newcomers
        for column in EXAM_COLUMNS:
            self.assertEqual([row[column] != "" for row in rows], [False] * 7 + [True])  # the exam is every 2,000
        self.assertEqual({row["manual"] for row in rows}, {"0"})
        self.assertGreater(seconds, 0.0)

    def test_a_row_every_250_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "1000", "--species", "instinct", "--out", folder]))
        self.assertEqual([int(row["step"]) for row in rows], [250, 500, 750, 1000])
        self.assertEqual(ROW_EVERY, 250)

    def test_run_writes_its_record(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "4", "--steps", "1000", "--species", "instinct", "--out", folder])
            record = json.loads((Path(folder) / "run.json").read_text())
            control = read_control(Path(folder))
        self.assertEqual((record["seed"], record["kind"], record["steps"]), (4, "instinct", 1000))
        self.assertRegex(record["started"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")
        self.assertEqual(control, Control("auto", None, False))  # the page can find and steer every run

    def test_every_run_gets_its_own_dated_folder(self):
        self.assertRegex(parse_args(["--seed", "7", "--species", "learning"]).out, r"^runs/\d{8}-\d{6}-7-learning$")
        self.assertRegex(parse_args(["--seed", "7", "--species", "learning", "--inherit", "inborn"]).out,
                         r"-7-learning-inborn$")
        self.assertRegex(parse_args(["--seed", "7", "--policy", "rules", "--level", "3"]).out, r"-7-rules-L3$")

    def test_steps_or_forever_not_both(self):
        self.assertIsNone(parse_args(["--seed", "1", "--species", "instinct", "--forever"]).steps)
        self.assertEqual(parse_args(["--seed", "1", "--species", "instinct"]).steps, 100000)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--seed", "1", "--species", "instinct", "--forever", "--steps", "10"])

    def test_same_seed_same_csv(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = quiet_run(["--seed", "3", "--steps", "1000", "--species", "instinct", "--out", a]).read_text()
            second = quiet_run(["--seed", "3", "--steps", "1000", "--species", "instinct", "--out", b]).read_text()
        self.assertEqual(first, second)

    def test_auto_reviews_number_and_fedness(self):
        reviews = []
        real = Dial.review

        def review(dial, alive, energy, newcomers):
            verdict = real(dial, alive, energy, newcomers)
            reviews.append((alive, energy, newcomers, round(dial.level, 2)))
            return verdict

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(Dial, "review", review):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--out", folder]))
        windows = [rows[i:i + ROWS_PER_REVIEW] for i in range(0, len(rows), ROWS_PER_REVIEW)]
        self.assertEqual([alive for alive, _, _, _ in reviews], [int(w[-1]["population"]) for w in windows])
        for (_, energy, _, _), window in zip(reviews, windows):  # the average energy of the 1,000 steps
            self.assertAlmostEqual(energy, sum(float(r["energy"]) for r in window) / ROWS_PER_REVIEW)
        soup = [sum(int(r["newcomers"]) for r in w) for w in windows]
        soup[0] -= FIRST_CREATURES  # the first creatures are not the soup's
        self.assertEqual([newcomers for _, _, newcomers, _ in reviews], soup)
        self.assertEqual(levels([w[0] for w in windows[1:]]), [level for *_, level in reviews[:-1]])

    def test_dial_turns_when_life_is_easy(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--out", folder]))
        self.assertEqual(levels(at_reviews(rows)), [0, 1, 2])
        changed = at_reviews(rows)
        self.assertLess(float(changed[1]["max_food"]), float(changed[0]["max_food"]))  # first: less food at once
        self.assertLess(float(changed[2]["food_energy"]), float(changed[1]["food_energy"]))  # then: less per food
        self.assertEqual(changed[2]["max_food"], changed[1]["max_food"])  # one knob at a time

    def test_dial_takes_nudges_back_when_life_is_hard(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("hard"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--level", "2",
                                        "--out", folder]))
        self.assertEqual(levels(at_reviews(rows)), [2, 1, 0])

    def test_dial_stops_at_the_max_level(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--max-level", "1",
                                        "--out", folder]))
        self.assertEqual(levels(at_reviews(rows)), [0, 1, 1])

    def test_manual_knobs_reach_the_world(self):
        mine = with_value(START, "food_energy", 15)
        with tempfile.TemporaryDirectory() as folder, owner_writes_at(500, Control("manual", mine, False)):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "1000", "--species", "instinct", "--out", folder]))
        self.assertEqual([(float(r["food_energy"]), r["manual"]) for r in rows],
                         [(30.0, "0"), (15.0, "1"), (15.0, "1"), (15.0, "1")])
        self.assertAlmostEqual(float(rows[-1]["level"]), 13.51, places=2)  # 30 -> 15 is 13.5 nudges' worth

    def test_auto_leaves_manual_knobs_alone(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--mode", "manual",
                                        "--out", folder]))
        self.assertEqual(set(levels(rows)), {0})
        self.assertEqual({row["manual"] for row in rows}, {"1"})

    def test_back_to_auto_goes_on_from_the_owner_s_settings(self):
        mine = with_value(START, "max_food", 100)
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            write_control(Path(folder), Control("manual", mine, False))
            with owner_writes_at(1500, Control("auto", None, False)):
                rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct",
                                            "--out", folder]))
        by_step = {int(r["step"]): r for r in rows}
        self.assertEqual(float(by_step[1000]["max_food"]), 100)  # Manual: the owner's, untouched by Auto
        self.assertEqual(by_step[1500]["manual"], "0")
        self.assertEqual(float(by_step[2250]["max_food"]), 95)  # Auto's first nudge, from the owner's 100
        self.assertEqual(by_step[2250]["food_energy"], by_step[1000]["food_energy"])

    def test_stop_ends_a_run_without_a_limit(self):
        with tempfile.TemporaryDirectory() as folder, owner_writes_at(750, Control("auto", None, True)):
            rows = read_rows(quiet_run(["--seed", "1", "--forever", "--species", "instinct", "--out", folder]))
            record = json.loads((Path(folder) / "run.json").read_text())
            ended = (Path(folder) / "time.txt").exists(), (Path(folder) / "best").exists()
        self.assertIsNone(record["steps"])
        self.assertEqual([int(row["step"]) for row in rows], [250, 500])
        self.assertEqual(ended, (True, True))

    def test_exam_does_not_change_the_world(self):
        argv = ["--seed", "2", "--steps", "3000", "--species", "instinct", "--out"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            examined = read_rows(quiet_run(argv + [a]))
            with mock.patch("src.evolve.EXAM_EVERY", 10 ** 9):
                unexamined = read_rows(quiet_run(argv + [b]))
        self.assertNotEqual(at_reviews(examined)[1]["intelligence"], "")
        self.assertEqual({row["intelligence"] for row in unexamined}, {""})
        for with_exam, without in zip(examined, unexamined):
            for column in EXAM_COLUMNS:
                with_exam.pop(column), without.pop(column)
        self.assertEqual(examined, unexamined)

    def test_yardstick_world_stays_at_its_level(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--policy", "random", "--level", "2",
                                        "--out", folder]))
        self.assertEqual(set(levels(rows)), {2})
        self.assertEqual({row["intelligence"] for row in rows}, {""})

    def test_learning_run_writes_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "2000", "--species", "learning", "--out", folder]))
        self.assertEqual(int(rows[-1]["step"]), 2000)
        self.assertEqual(levels(rows[:1]), [0])
        self.assertGreaterEqual(int(rows[-1]["population"]), 10)
        self.assertNotEqual(rows[-1]["intelligence"], "")
        self.assertNotEqual(rows[-1]["energy"], "")

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


    def test_watched_world_writes_its_view(self):
        with tempfile.TemporaryDirectory() as folder, no_waiting():
            watched(folder)
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "250", "--species", "instinct", "--out", folder]))
            view = json.loads((Path(folder) / "view.json").read_text())
        self.assertEqual(view["step"], 250)
        self.assertEqual(len(view["creatures"]), int(rows[-1]["population"]))
        self.assertGreater(len(view["foods"]), 0)
        creature = view["creatures"][0]
        self.assertEqual(set(creature), {"x", "y", "radius", "energy", "heading", "did", "generation"})
        self.assertTrue(0.0 <= creature["energy"] <= 1.0)
        self.assertTrue(0.0 <= creature["x"] <= 50 and 0.0 <= creature["y"] <= 30)
        self.assertEqual(set(view["foods"][0]), {"x", "y", "radius"})

    def test_watching_never_changes_the_world(self):
        argv = ["--seed", "2", "--steps", "500", "--species", "instinct", "--out"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, no_waiting():
            watched(a, speed=30)
            seen = quiet_run(argv + [a]).read_text()
            unseen = quiet_run(argv + [b]).read_text()
        self.assertEqual(seen, unseen)

    def test_watch_runs_out_by_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            watched(folder, until=time.time() - 1)
            started = time.perf_counter()
            quiet_run(["--seed", "1", "--steps", "250", "--species", "instinct", "--out", folder])
            took = time.perf_counter() - started
            view = (Path(folder) / "view.json").exists()
        self.assertFalse(view)
        self.assertLess(took, 8.0)  # 250 steps at 10 a second would take 25

    def test_watched_world_waits(self):
        with tempfile.TemporaryDirectory() as folder:
            watched(folder, speed=25)
            started = time.perf_counter()
            quiet_run(["--seed", "1", "--steps", "50", "--species", "instinct", "--out", folder])
            took = time.perf_counter() - started
        self.assertGreaterEqual(took, 1.8)  # 50 steps at 25 a second

    def test_a_save_every_2000_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "1", "--steps", "4500", "--species", "instinct", "--out", folder])
            self.assertEqual(saved_steps(folder), [2000, 4000, 4500])  # and one at the end

    def test_stop_saves(self):
        with tempfile.TemporaryDirectory() as folder, owner_writes_at(750, Control("auto", None, True)):
            quiet_run(["--seed", "1", "--forever", "--species", "instinct", "--out", folder])
            self.assertEqual(saved_steps(folder), [749])  # the last step it took

    def test_resume_starts_a_new_world_from_the_save(self):
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            quiet_run(["--seed", "3", "--steps", "2000", "--species", "instinct", "--out", old])
            before = (Path(old) / "scoreboard.csv").read_text()
            save = Path(old) / "saves" / "step-0002000.npz"
            rows = read_rows(quiet_run(["--resume", str(save), "--steps", "2500", "--mode", "manual", "--out", new]))
            record = json.loads((Path(new) / "run.json").read_text())
            control = read_control(Path(new))
            after = (Path(old) / "scoreboard.csv").read_text()
            old_rows = read_rows(Path(old) / "scoreboard.csv")
        self.assertEqual((record["seed"], record["kind"], record["steps"]), (3, "instinct", 2500))
        self.assertEqual(record["from"], {"run": Path(old).name, "step": 2000})
        self.assertEqual([int(r["step"]) for r in rows], [250 * i for i in range(1, 11)])
        self.assertEqual(rows[:8], old_rows)  # the old world's history up to the save
        self.assertEqual((rows[8]["manual"], control.mode), ("1", "manual"))
        self.assertEqual(before, after)  # the old world is never changed

    def test_resume_needs_nothing_else(self):
        args = parse_args(["--resume", "runs/x/saves/step-0002000.npz"])
        self.assertEqual(args.resume, "runs/x/saves/step-0002000.npz")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--resume", "x.npz", "--species", "instinct", "--seed", "1"])

    def test_pause_takes_no_steps(self):
        real = control_module.read_control
        looks = []

        def read(folder):
            looks.append(time.perf_counter())
            if len(looks) == 11:  # step 250
                write_control(folder, Control("auto", None, False, paused=True))
            if len(looks) > 11 and looks[-1] - looks[11] > 1.0:  # a second later: Continue
                write_control(folder, Control("auto", None, False))
            return real(folder)

        with tempfile.TemporaryDirectory() as folder, mock.patch("src.evolve.read_control", read):
            started = time.perf_counter()
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "500", "--species", "instinct", "--out", folder]))
            took = time.perf_counter() - started
        self.assertGreaterEqual(took, 1.0)
        self.assertEqual([int(r["step"]) for r in rows], [250, 500])  # it went on afterwards

if __name__ == "__main__":
    unittest.main()
