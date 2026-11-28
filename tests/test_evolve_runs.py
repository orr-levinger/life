"""Running worlds: the second half of the evolve tests (split from tests/test_evolve.py on 2026-11-27 so the suite,
run four files at a time, stays under its 5 minutes; each test unchanged)."""
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
from tests.helpers import SMALL_WORLD
from src.control import Control, read_control, write_control
from src.evolve import parse_args, run
from tests.test_evolve import levels, no_waiting, owner_writes_at, quiet_run, read_rows, saved_steps, watched


class TestEvolveRuns(unittest.TestCase):
    def test_learning_run_writes_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "400", "--species", "learning", "--out", folder]))
        self.assertEqual(int(rows[-1]["step"]), 400)
        self.assertEqual(levels(rows[:1]), [0])
        self.assertGreaterEqual(int(rows[-1]["population"]), 10)
        self.assertNotEqual(rows[-1]["intelligence"], "")
        self.assertNotEqual(rows[-1]["energy"], "")

    def test_same_seed_same_csv_learning(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = quiet_run(["--seed", "3", "--steps", "200", "--species", "learning", "--out", a]).read_text()
            second = quiet_run(["--seed", "3", "--steps", "200", "--species", "learning", "--out", b]).read_text()
        self.assertEqual(first, second)

    def test_species_or_policy_is_required(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--seed", "1"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--seed", "1", "--species", "instinct", "--policy", "random"])


    def test_watched_world_writes_its_view(self):
        with tempfile.TemporaryDirectory() as folder, no_waiting():
            watched(folder)
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "50", "--species", "instinct", "--out", folder]))
            view = json.loads((Path(folder) / "view.json").read_text())
        self.assertEqual(view["step"], 50)
        self.assertEqual(len(view["creatures"]), int(rows[-1]["population"]))
        self.assertGreater(len(view["foods"]), 0)
        creature = view["creatures"][0]
        self.assertEqual(set(creature), {"id", "p", "radius", "energy", "heading", "did", "generation", "age",
                                         "lifespan", "earned", "spent", "size", "facing", "vision_width",
                                         "vision_reach", "hearing", "fear_of_death", "love_of_babies",
                                         "learns_from_others", "neurons", "layers", "moved", "bumped", "brain", "mutations", "mutation_size", "grass_gut", "meat_gut", "calls", "call", "kin", "gills", "lungs", "stealth", "armour",
                                         "learning_speed", "intent", "target", "baby"})
        self.assertTrue(0.0 <= creature["energy"] <= 1.0)
        self.assertAlmostEqual(sum(v * v for v in creature["p"]), 1.0, places=4)  # a point on the planet
        self.assertEqual(set(view["foods"][0]), {"p", "radius", "fruit", "meat"})
        self.assertEqual(set(view["terrain"]), {"radius", "obstacles", "trees"})
        self.assertEqual(view["radius"], 10.9)  # the small test planet

    def test_watching_never_changes_the_world(self):
        argv = ["--seed", "2", "--steps", "100", "--species", "instinct", "--out"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, no_waiting():
            watched(a, speed=30)
            seen = quiet_run(argv + [a]).read_text()
            unseen = quiet_run(argv + [b]).read_text()
        self.assertEqual(seen, unseen)

    def test_watch_runs_out_by_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            watched(folder, until=time.time() - 1)
            started = time.perf_counter()
            quiet_run(["--seed", "1", "--steps", "50", "--species", "instinct", "--out", folder])
            took = time.perf_counter() - started
            view = (Path(folder) / "view.json").exists()
        self.assertFalse(view)
        self.assertLess(took, 8.0)  # 50 steps at 10 a second would take 25

    def test_watched_world_waits(self):
        with tempfile.TemporaryDirectory() as folder:
            watched(folder, speed=25)
            started = time.perf_counter()
            quiet_run(["--seed", "1", "--steps", "50", "--species", "instinct", "--out", folder])
            took = time.perf_counter() - started
        self.assertGreaterEqual(took, 1.8)  # 50 steps at 25 a second

    def test_a_save_every_400_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "1", "--steps", "900", "--species", "instinct", "--out", folder])
            self.assertEqual(saved_steps(folder), [400, 800, 900])  # and one at the end

    def test_stop_saves(self):
        with tempfile.TemporaryDirectory() as folder, owner_writes_at(150, Control("auto", None, True)):
            quiet_run(["--seed", "1", "--forever", "--species", "instinct", "--out", folder])
            self.assertEqual(saved_steps(folder), [149])  # the last step it took

    def test_resume_starts_a_new_world_from_the_save(self):
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            quiet_run(["--seed", "3", "--steps", "400", "--species", "instinct", "--out", old])
            before = (Path(old) / "scoreboard.csv").read_text()
            save = Path(old) / "saves" / "step-0000400.npz"
            rows = read_rows(quiet_run(["--resume", str(save), "--steps", "500", "--mode", "manual", "--out", new]))
            record = json.loads((Path(new) / "run.json").read_text())
            control = read_control(Path(new))
            after = (Path(old) / "scoreboard.csv").read_text()
            old_rows = read_rows(Path(old) / "scoreboard.csv")
        self.assertEqual((record["seed"], record["kind"], record["steps"]), (3, "instinct", 500))
        self.assertEqual(record["from"], {"run": Path(old).name, "step": 400})
        self.assertEqual([int(r["step"]) for r in rows], [50 * i for i in range(1, 11)])
        self.assertEqual(rows[:8], old_rows)  # the old world's history up to the save
        self.assertEqual((rows[8]["manual"], control.mode), ("1", "manual"))
        self.assertEqual(before, after)  # the old world is never changed

    def test_resume_needs_nothing_else(self):
        args = parse_args(["--resume", "runs/x/saves/step-0000400.npz"])
        self.assertEqual(args.resume, "runs/x/saves/step-0000400.npz")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--resume", "x.npz", "--species", "instinct", "--seed", "1"])

    def test_pause_takes_no_steps(self):
        real = control_module.read_control
        looks = []

        def read(folder):
            looks.append(time.perf_counter())
            if len(looks) == 11:  # step 50
                write_control(folder, Control("auto", None, False, paused=True))
            if len(looks) > 11 and looks[-1] - looks[11] > 1.0:  # a second later: Continue
                write_control(folder, Control("auto", None, False))
            return real(folder)

        with tempfile.TemporaryDirectory() as folder, mock.patch("src.evolve.read_control", read):
            started = time.perf_counter()
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "100", "--species", "instinct", "--out", folder]))
            took = time.perf_counter() - started
        self.assertGreaterEqual(took, 1.0)
        self.assertEqual([int(r["step"]) for r in rows], [50, 100])  # it went on afterwards

    def test_a_running_world_says_how_it_is_now(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "200", "--species", "instinct", "--out", folder]))
            now = json.loads((Path(folder) / "now.json").read_text())
        self.assertEqual(set(now), {"step", "alive", "energy", "food", "births", "deaths", "time", "rules", "season", "genes",
                                    "placed"})  # placed: the owner's last creature to place, for the page
        low, mean, high = now["genes"]["size"]
        self.assertTrue(low <= mean <= high)
        self.assertEqual((now["step"], now["alive"]), (200, int(rows[-1]["population"])))
        self.assertEqual(now["births"], sum(int(r["births"]) for r in rows))
        self.assertEqual(now["deaths"], sum(int(r["deaths_hunger"]) + int(r["deaths_attack"]) + int(r["deaths_age"]) for r in rows))
        self.assertTrue(0.0 <= now["energy"] <= 1.0)
        self.assertLess(abs(now["time"] - time.time()), 60)

    def test_now_never_changes_the_world(self):
        argv = ["--seed", "2", "--steps", "100", "--species", "instinct", "--out"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with mock.patch.dict(SMALL_WORLD, {"now_seconds": 0.0}):  # every 25 steps
                with_now = quiet_run(argv + [a]).read_text()
            with mock.patch("src.evolve.world_now", return_value={}) as asked, mock.patch("src.evolve.write_json"):
                without = quiet_run(argv + [b]).read_text()  # never looked at, never written
        self.assertGreater(asked.call_count, 0)
        self.assertEqual(with_now, without)

    def test_update_goes_on_in_the_same_run(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "1", "--steps", "400", "--species", "instinct", "--out", folder])
            (Path(folder) / "time.txt").unlink()  # the page server does this before handing over
            save = Path(folder) / "saves" / "step-0000400.npz"
            rows = read_rows(quiet_run(["--resume", str(save), "--continue", "--steps", "600", "--out", folder]))
            record = json.loads((Path(folder) / "run.json").read_text())
            folders = [p.name for p in Path(folder).iterdir() if p.is_dir()]
        self.assertEqual([int(r["step"]) for r in rows], [50 * i for i in range(1, 13)])
        self.assertEqual([u["step"] for u in record["updates"]], [400])
        self.assertEqual((record["seed"], record["kind"]), (1, "instinct"))
        self.assertEqual(sorted(folders), ["best", "report", "saves"])  # no new run folder (its report: 2026-11-27)

    def test_update_drops_rows_after_the_save(self):
        with tempfile.TemporaryDirectory() as folder:
            quiet_run(["--seed", "1", "--steps", "500", "--species", "instinct", "--out", folder])
            save = Path(folder) / "saves" / "step-0000400.npz"
            rows = read_rows(quiet_run(["--resume", str(save), "--continue", "--steps", "600", "--out", folder]))
        steps = [int(r["step"]) for r in rows]
        self.assertEqual(steps, sorted(set(steps)))
        self.assertEqual(steps[-1], 600)

    def test_updated_world_is_the_same_world(self):
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as updated:
            straight = (Path(quiet_run(["--seed", "4", "--steps", "600", "--species", "instinct", "--out", whole]))).read_text()
            quiet_run(["--seed", "4", "--steps", "400", "--species", "instinct", "--out", updated])
            save = Path(updated) / "saves" / "step-0000400.npz"
            again = quiet_run(["--resume", str(save), "--continue", "--steps", "600", "--out", updated]).read_text()
        self.assertEqual(again, straight)

    def test_continue_needs_resume(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--seed", "1", "--species", "instinct", "--continue"])

    def test_hunters_switch_in_auto(self):
        with tempfile.TemporaryDirectory() as folder, owner_writes_at(100, Control("auto", None, False, hunters=True)):
            rows = read_rows(quiet_run(["--seed", "1", "--steps", "200", "--species", "instinct", "--out", folder]))
        self.assertEqual([(r["hunters"], r["manual"]) for r in rows], [("0", "0"), ("1", "0"), ("1", "0"), ("1", "0")])

if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
