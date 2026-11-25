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
from src.dial import Dial, rates, value, with_value
from src.evolve import parse_args, run
from src.scoreboard import ROW_KEYS
from src.settings import START
from src.rules import rule, starting_rules

TARGET_ALIVE, REVIEW_EVERY = rule("auto_target").start, round(rule("auto_every").start)
CONTROL_EVERY, ROW_EVERY = round(rule("control_every").start), round(rule("row_every").start)
FIRST_CREATURES = round(rule("first_creatures").start)

EXAM_COLUMNS = ("intelligence", "wanderers")


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
    """Auto's review, told that the creatures are steady 8 above the target ("easy": one nudge harder) or 8 below it
    ("hard": one nudge easier)."""
    real = Dial.review
    alive = TARGET_ALIVE + 8 if life == "easy" else TARGET_ALIVE - 8

    def review(dial, *measured):
        return real(dial, alive, 0.0, 0.0, 0)
    return mock.patch.object(Dial, "review", review)


def by_step(rows):
    return {int(row["step"]): row for row in rows}


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

    def test_auto_looks_every_500_steps(self):
        looks = []
        real = Dial.review

        def review(dial, *measured):
            looks.append(dial)
            return real(dial, *measured)

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(Dial, "review", review):
            quiet_run(["--seed", "1", "--steps", "2000", "--species", "instinct", "--out", folder])
        self.assertEqual((REVIEW_EVERY, len(looks)), (500, 4))

    def test_auto_reviews_the_rates(self):
        reviews = []
        real = Dial.review

        def review(dial, alive, alive_rate, energy_rate, newcomers):
            made = real(dial, alive, alive_rate, energy_rate, newcomers)
            reviews.append((alive, alive_rate, energy_rate, newcomers, round(dial.level, 2)))
            return made

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(Dial, "review", review):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "3000", "--species", "instinct", "--out", folder]))
        numbers = [{"step": int(r["step"]), "population": int(r["population"]), "energy": float(r["energy"])} for r in rows]
        for i, (alive, alive_rate, energy_rate, newcomers, level) in enumerate(reviews):
            at = 2 * i + 1  # the row written at this look: steps 500, 1,000, 1,500 …
            self.assertEqual(alive, numbers[at]["population"])
            expected = rates(numbers[max(0, at - 3):at + 1])  # the points of the last 1,000 steps
            self.assertAlmostEqual(alive_rate, expected[0])
            self.assertAlmostEqual(energy_rate, expected[1])
            soup = int(rows[at]["newcomers"]) + int(rows[at - 1]["newcomers"]) - (FIRST_CREATURES if i == 0 else 0)
            self.assertEqual(newcomers, soup)
            if at + 1 < len(rows):
                self.assertEqual(levels([rows[at + 1]]), [level])

    def test_dial_turns_when_life_is_easy(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            rows = by_step(read_rows(quiet_run(["--seed", "1", "--steps", "1500", "--species", "instinct",
                                                "--out", folder])))
        self.assertEqual(levels(rows.values()), [0, 0, 1, 1, 2, 2])  # one nudge at each look (500, 1,000)
        self.assertLess(float(rows[750]["max_food"]), float(rows[500]["max_food"]))  # first: less food at once
        self.assertLess(float(rows[1250]["food_energy"]), float(rows[1000]["food_energy"]))  # then: less per food
        self.assertEqual(rows[1250]["max_food"], rows[1000]["max_food"])  # one knob at a time

    def test_dial_takes_nudges_back_when_life_is_hard(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("hard"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "1500", "--species", "instinct", "--level", "2",
                                        "--out", folder]))
        self.assertEqual(levels(rows), [2, 2, 1, 1, 0, 0])

    def test_dial_stops_at_the_max_level(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "1500", "--species", "instinct", "--max-level", "1",
                                        "--out", folder]))
        self.assertEqual(levels(rows), [0, 0, 1, 1, 1, 1])

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
        rows = by_step(rows)
        self.assertEqual(float(rows[1000]["max_food"]), 100)  # Manual: the owner's, untouched by Auto
        self.assertEqual(rows[1500]["manual"], "0")
        self.assertEqual(float(rows[1750]["max_food"]), 95)  # Auto's first nudge (at 1,500), from the owner's 100
        self.assertEqual(rows[1750]["food_energy"], rows[1000]["food_energy"])

    def test_a_rule_change_reaches_the_running_world(self):
        from src.saves import load_world
        with tempfile.TemporaryDirectory() as folder, \
                owner_writes_at(500, Control("auto", None, False, rules={"eat_miss": 0.9, "tank": 150.0})):
            quiet_run(["--seed", "1", "--steps", "1000", "--species", "instinct", "--out", folder])
            now = json.loads((Path(folder) / "now.json").read_text())
            world, _, _ = load_world(Path(folder) / "saves" / f"step-{1000:07d}.npz", learner_seed=0)
        self.assertEqual((now["rules"]["eat_miss"], now["rules"]["tank"]), (0.9, 150.0))
        self.assertEqual((world.rules["eat_miss"], world.rules["tank"]), (0.9, 150.0))  # kept in its save

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
            with mock.patch("src.world.starting_rules", lambda: {**starting_rules(), "exam_every": 10.0 ** 9}):
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
        self.assertEqual(set(creature), {"id", "x", "y", "radius", "energy", "heading", "did", "generation", "age",
                                         "lifespan", "earned", "spent", "size", "facing", "vision_width",
                                         "vision_reach", "hearing", "fear_of_death", "love_of_babies",
                                         "learns_from_others", "neurons", "layers", "moved",
                                         "learning_speed", "intent", "target", "baby"})
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

    def test_a_running_world_says_how_it_is_now(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "1000", "--species", "instinct", "--out", folder]))
            now = json.loads((Path(folder) / "now.json").read_text())
        self.assertEqual(set(now), {"step", "alive", "energy", "food", "births", "deaths", "time", "rules", "genes"})
        low, mean, high = now["genes"]["size"]
        self.assertTrue(low <= mean <= high)
        self.assertEqual((now["step"], now["alive"]), (1000, int(rows[-1]["population"])))
        self.assertEqual(now["births"], sum(int(r["births"]) for r in rows))
        self.assertEqual(now["deaths"], sum(int(r["deaths_hunger"]) + int(r["deaths_attack"]) + int(r["deaths_age"]) for r in rows))
        self.assertTrue(0.0 <= now["energy"] <= 1.0)
        self.assertLess(abs(now["time"] - time.time()), 60)

    def test_now_never_changes_the_world(self):
        argv = ["--seed", "2", "--steps", "500", "--species", "instinct", "--out"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with mock.patch("src.world.starting_rules", lambda: {**starting_rules(), "now_seconds": 0.0}):  # every 25 steps
                with_now = quiet_run(argv + [a]).read_text()
            with mock.patch("src.evolve.world_now", return_value={}) as asked, mock.patch("src.evolve.write_json"):
                without = quiet_run(argv + [b]).read_text()  # never looked at, never written
        self.assertGreater(asked.call_count, 0)
        self.assertEqual(with_now, without)

    def test_update_goes_on_in_the_same_run(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "1", "--steps", "2000", "--species", "instinct", "--out", folder])
            (Path(folder) / "time.txt").unlink()  # the page server does this before handing over
            save = Path(folder) / "saves" / "step-0002000.npz"
            rows = read_rows(quiet_run(["--resume", str(save), "--continue", "--steps", "3000", "--out", folder]))
            record = json.loads((Path(folder) / "run.json").read_text())
            folders = [p.name for p in Path(folder).iterdir() if p.is_dir()]
        self.assertEqual([int(r["step"]) for r in rows], [250 * i for i in range(1, 13)])
        self.assertEqual([u["step"] for u in record["updates"]], [2000])
        self.assertEqual((record["seed"], record["kind"]), (1, "instinct"))
        self.assertEqual(sorted(folders), ["best", "saves"])  # no new run folder

    def test_update_drops_rows_after_the_save(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "1", "--steps", "2500", "--species", "instinct", "--out", folder])
            save = Path(folder) / "saves" / "step-0002000.npz"
            rows = read_rows(quiet_run(["--resume", str(save), "--continue", "--steps", "3000", "--out", folder]))
        steps = [int(r["step"]) for r in rows]
        self.assertEqual(steps, sorted(set(steps)))
        self.assertEqual(steps[-1], 3000)

    def test_updated_world_is_the_same_world(self):
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as updated:
            straight = (Path(quiet_run(["--seed", "4", "--steps", "3000", "--species", "instinct", "--out", whole]))).read_text()
            quiet_run(["--seed", "4", "--steps", "2000", "--species", "instinct", "--out", updated])
            save = Path(updated) / "saves" / "step-0002000.npz"
            again = quiet_run(["--resume", str(save), "--continue", "--steps", "3000", "--out", updated]).read_text()
        self.assertEqual(again, straight)

    def test_continue_needs_resume(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--seed", "1", "--species", "instinct", "--continue"])

    def test_hunters_switch_in_auto(self):
        with tempfile.TemporaryDirectory() as folder, owner_writes_at(500, Control("auto", None, False, hunters=True)):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "1000", "--species", "instinct", "--out", folder]))
        self.assertEqual([(r["hunters"], r["manual"]) for r in rows], [("0", "0"), ("1", "0"), ("1", "0"), ("1", "0")])

if __name__ == "__main__":
    unittest.main()
