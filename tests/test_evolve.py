import contextlib
import csv
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import control as control_module
from tests.helpers import SMALL_WORLD, small_world
from src.control import Control, read_control, write_control
from src.dial import Dial, rates, value, with_value
from src.evolve import _Knobs, _new_world, parse_args, run
from src.report import Reporter
from src.scoreboard import ROW_KEYS
from src.settings import START
from src.rules import rule, starting_rules

TARGET_ALIVE, REVIEW_EVERY = SMALL_WORLD["auto_target"], round(SMALL_WORLD["auto_every"])  # the small test world
CONTROL_EVERY, ROW_EVERY = round(SMALL_WORLD["control_every"]), round(SMALL_WORLD["row_every"])
FIRST_CREATURES = round(SMALL_WORLD["first_creatures"])

EXAM_COLUMNS = ("intelligence", "wanderers")


def quiet_run(argv):
    with small_world(), contextlib.redirect_stdout(io.StringIO()):  # the small test world, whatever the starting rules
        return run(parse_args(argv))


def read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def levels(rows):
    return [round(float(row["level"]), 2) for row in rows]


def at_reviews(rows):
    """The rows written at Auto's reviews (steps 200, 2,000 …)."""
    return [row for row in rows if int(row["step"]) % 200 == 0]


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
            path = quiet_run(["--seed", "1", "--steps", "400", "--species", "instinct", "--out", folder])
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
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "200", "--species", "instinct", "--out", folder]))
        self.assertEqual([int(row["step"]) for row in rows], [50, 100, 150, 200])
        self.assertEqual(ROW_EVERY, 50)

    def test_a_run_writes_its_water_picture(self):  # the page draws the planet's seas from it
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "4", "--steps", "50", "--species", "instinct", "--out", folder])
            self.assertEqual((Path(folder) / "water.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_run_writes_its_record(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "4", "--steps", "200", "--species", "instinct", "--out", folder])
            record = json.loads((Path(folder) / "run.json").read_text())
            control = read_control(Path(folder))
        self.assertEqual((record["seed"], record["kind"], record["steps"]), (4, "instinct", 200))
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
            first = quiet_run(["--seed", "3", "--steps", "200", "--species", "instinct", "--out", a]).read_text()
            second = quiet_run(["--seed", "3", "--steps", "200", "--species", "instinct", "--out", b]).read_text()
        self.assertEqual(first, second)

    def test_auto_looks_every_500_steps(self):
        looks = []
        real = Dial.review

        def review(dial, *measured):
            looks.append(dial)
            return real(dial, *measured)

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(Dial, "review", review):
            quiet_run(["--seed", "1", "--steps", "400", "--species", "instinct", "--out", folder])
        self.assertEqual((REVIEW_EVERY, len(looks)), (100, 4))

    def test_auto_reviews_the_rates(self):
        reviews = []
        real = Dial.review

        def review(dial, alive, alive_rate, energy_rate, newcomers):
            made = real(dial, alive, alive_rate, energy_rate, newcomers)
            reviews.append((alive, alive_rate, energy_rate, newcomers, round(dial.level, 2)))
            return made

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(Dial, "review", review):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "600", "--species", "instinct", "--out", folder]))
        numbers = [{"step": int(r["step"]), "population": int(r["population"]), "energy": float(r["energy"])} for r in rows]
        for i, (alive, alive_rate, energy_rate, newcomers, level) in enumerate(reviews):
            at = 2 * i + 1  # the row written at this look: steps 100, 200, 1,100 …
            self.assertEqual(alive, numbers[at]["population"])
            expected = rates(numbers[max(0, at - 3):at + 1])  # the points of the last 200 steps
            self.assertAlmostEqual(alive_rate, expected[0])
            self.assertAlmostEqual(energy_rate, expected[1])
            soup = int(rows[at]["newcomers"]) + int(rows[at - 1]["newcomers"]) - (FIRST_CREATURES if i == 0 else 0)
            self.assertEqual(newcomers, soup)
            if at + 1 < len(rows):
                self.assertEqual(levels([rows[at + 1]]), [level])

    def test_dial_turns_when_life_is_easy(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            rows = by_step(read_rows(quiet_run(["--seed", "1", "--steps", "300", "--species", "instinct",
                                                "--out", folder])))
        self.assertEqual(levels(rows.values()), [0, 0, 1, 1, 2, 2])  # one nudge at each look (100, 200)
        self.assertLess(float(rows[150]["max_food"]), float(rows[100]["max_food"]))  # first: less food at once
        self.assertLess(float(rows[250]["food_energy"]), float(rows[200]["food_energy"]))  # then: less per food
        self.assertEqual(rows[250]["max_food"], rows[200]["max_food"])  # one knob at a time

    def test_dial_takes_nudges_back_when_life_is_hard(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("hard"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "300", "--species", "instinct", "--level", "2",
                                        "--out", folder]))
        self.assertEqual(levels(rows), [2, 2, 1, 1, 0, 0])

    def test_dial_stops_at_the_max_level(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "300", "--species", "instinct", "--max-level", "1",
                                        "--out", folder]))
        self.assertEqual(levels(rows), [0, 0, 1, 1, 1, 1])

    def test_manual_knobs_reach_the_world(self):
        mine = with_value(START, "food_energy", 15)
        with tempfile.TemporaryDirectory() as folder, owner_writes_at(100, Control("manual", mine, False)):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "200", "--species", "instinct", "--out", folder]))
        self.assertEqual([(float(r["food_energy"]), r["manual"]) for r in rows],
                         [(30.0, "0"), (15.0, "1"), (15.0, "1"), (15.0, "1")])
        self.assertAlmostEqual(float(rows[-1]["level"]), 13.03, places=2)  # 30 -> 15: 8.5 nudges of 5 %, then 4.5 of 1 % of the range

    def test_auto_leaves_manual_knobs_alone(self):
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "600", "--species", "instinct", "--mode", "manual",
                                        "--out", folder]))
        self.assertEqual(set(levels(rows)), {0})
        self.assertEqual({row["manual"] for row in rows}, {"1"})

    def test_back_to_auto_goes_on_from_the_owner_s_settings(self):
        mine = with_value(START, "max_food", 100)
        with tempfile.TemporaryDirectory() as folder, reviewing_as_if("easy"):
            write_control(Path(folder), Control("manual", mine, False))
            with owner_writes_at(300, Control("auto", None, False)):
                rows = read_rows(quiet_run(["--seed", "1", "--steps", "600", "--species", "instinct",
                                            "--out", folder]))
        rows = by_step(rows)
        self.assertEqual(float(rows[200]["max_food"]), 100)  # Manual: the owner's, untouched by Auto
        self.assertEqual(rows[300]["manual"], "0")
        nudged = next(r for r in rows.values() if int(r["step"]) > 300 and float(r["max_food"]) != 100)
        self.assertAlmostEqual(float(nudged["max_food"]), 92.2)  # Auto's first nudge from the owner's 100: 1 % of the range (7.8) beats 5 % (5)
        self.assertEqual(nudged["food_energy"], rows[200]["food_energy"])  # one knob at a time

    def test_a_rule_change_reaches_the_running_world(self):
        from src.saves import load_world
        with tempfile.TemporaryDirectory() as folder, \
                owner_writes_at(100, Control("auto", None, False, rules={"diet_bite": 0.9, "tank": 150.0})):
            quiet_run(["--seed", "1", "--steps", "200", "--species", "instinct", "--out", folder])
            now = json.loads((Path(folder) / "now.json").read_text())
            world, _, _ = load_world(Path(folder) / "saves" / f"step-{200:07d}.npz", learner_seed=0)
        self.assertEqual((now["rules"]["diet_bite"], now["rules"]["tank"]), (0.9, 150.0))
        self.assertEqual((world.rules["diet_bite"], world.rules["tank"]), (0.9, 150.0))  # kept in its save

    def test_stop_ends_a_run_without_a_limit(self):
        with tempfile.TemporaryDirectory() as folder, owner_writes_at(150, Control("auto", None, True)):
            rows = read_rows(quiet_run(["--seed", "1", "--forever", "--species", "instinct", "--out", folder]))
            record = json.loads((Path(folder) / "run.json").read_text())
            ended = (Path(folder) / "time.txt").exists(), (Path(folder) / "best").exists()
        self.assertIsNone(record["steps"])
        self.assertEqual([int(row["step"]) for row in rows], [50, 100])
        self.assertEqual(ended, (True, True))

    def test_exam_does_not_change_the_world(self):
        argv = ["--seed", "2", "--steps", "600", "--species", "instinct", "--out"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            examined = read_rows(quiet_run(argv + [a]))
            with mock.patch.dict(SMALL_WORLD, {"exam_every": 10.0 ** 9}):  # the same small world, never examined
                unexamined = read_rows(quiet_run(argv + [b]))
        self.assertNotEqual(at_reviews(examined)[1]["intelligence"], "")
        self.assertEqual({row["intelligence"] for row in unexamined}, {""})
        for with_exam, without in zip(examined, unexamined):
            for column in EXAM_COLUMNS:
                with_exam.pop(column), without.pop(column)
        self.assertEqual(examined, unexamined)

    def test_yardstick_world_stays_at_its_level(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "600", "--policy", "random", "--level", "2",
                                        "--out", folder]))
        self.assertEqual(set(levels(rows)), {2})
        self.assertEqual({row["intelligence"] for row in rows}, {""})


class TestGeneChanges(unittest.TestCase):
    def test_a_gene_change_reaches_the_world_once(self):
        world, dial = mock.Mock(), mock.Mock()
        change = {"seq": 3, "id": 9, "genes": {"size": 1.1}}
        knobs = _Knobs(Control("auto", None, False), Reporter(Path("unused")))  # writes only when a creature is placed
        for _ in range(3):  # the control file is read again and again
            knobs.take(Control("auto", None, False, edit=change), dial, world)
        world.edit_genes.assert_called_once_with(9, {"size": 1.1})
        knobs.take(Control("auto", None, False, edit={**change, "seq": 4}), dial, world)
        self.assertEqual(world.edit_genes.call_count, 2)  # a new change, even of the same genes


class TestViewOnThePlanet(unittest.TestCase):
    def test_the_view_gives_points_on_the_planet(self):
        import numpy as np
        from src.evolve import world_view
        from src.terrain import Obstacle, Terrain
        from tests.helpers import add, cap, make_world
        w = make_world()
        c = add(w, 3.0, 4.0)
        w.terrain = Terrain([Obstacle("rock", (cap(10.0, 0.0, 2.0),))], [], w.radius)
        view = world_view(w)
        seen = view["creatures"][0]
        self.assertEqual((len(seen["p"]), len(seen["facing"])), (3, 3))
        self.assertAlmostEqual(float(np.linalg.norm(seen["p"])), 1.0, places=5)
        np.testing.assert_allclose(seen["p"], c.p, atol=1e-6)
        self.assertNotIn("x", seen)
        self.assertEqual((view["radius"], view["terrain"]["obstacles"][0]["kind"]), (w.radius, "rock"))
        self.assertEqual(len(view["terrain"]["obstacles"][0]["caps"][0]), 4)  # a point and an arc radius


class TestNextWorldRules(unittest.TestCase):
    def test_a_new_world_is_made_on_the_owners_rules(self):
        args = parse_args(["--seed", "1", "--species", "instinct", "--steps", "10"])
        world, _ = _new_world(args, {"first_creatures": 5.0, "planet_radius": 20.0})
        self.assertEqual((len(world.creatures), world.radius), (5, 20.0))

    def test_a_new_planet_has_seas(self):
        with small_world():
            args = parse_args(["--seed", "1", "--species", "instinct", "--steps", "10"])
            world, _ = _new_world(args, {})
        p = np.random.default_rng(2).normal(size=(20_000, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        self.assertAlmostEqual(float(world.terrain.water.is_water(p).mean()), world.rules["water_share"], delta=0.05)

    def test_the_view_has_the_followed_ones_pixels(self):  # the Pixels view: what it sees, as it sees it
        from src.evolve import world_view
        from tests.helpers import add, make_world
        w = make_world()
        me, other = add(w, 0.0, 0.0), add(w, 2.0, 0.0)
        me.eye_pixels = 4
        eye = world_view(w, follow=me.id)["eye"]
        self.assertEqual((eye["id"], eye["sharpness"], len(eye["pixels"]), len(eye["pixels"][0]), len(eye["ears"])),
                         (me.id, 4, 16, 6, 8))
        self.assertGreater(max(p[0] for p in eye["pixels"]), 0.0)  # it sees the other one
        self.assertNotIn("eye", world_view(w))  # nobody followed: nothing more to send
        self.assertNotIn("eye", world_view(w, follow=999))  # gone

    def test_a_new_world_has_the_starting_size(self):
        args = parse_args(["--seed", "1", "--species", "instinct", "--steps", "10"])
        world, _ = _new_world(args, {})  # the owner changed nothing: the starting rules' planet, radius 37
        self.assertEqual(world.radius, 37.0)
