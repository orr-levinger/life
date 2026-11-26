import contextlib
import csv
import io
import json
import math
import random
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.dial import Dial
from src.evolve import parse_args, run
from src.learner import Learner
from src.saves import FLAT_WORLD, SAVE_FORMAT, SaveError, load_world, save_format, save_world, saves_of
from src.rules import ROOM_SLOTS as MAX_CREATURES
from src.settings import START
from src.terrain import make_terrain
from tests.helpers import at, add, make_world, populate, small_world
from tests.test_learner import live
from src.brain import N_SENSES
from src.rules import rule

TARGET_ALIVE = rule("auto_target").start


def quiet_run(argv):
    with small_world(), contextlib.redirect_stdout(io.StringIO()):  # the small test world, whatever the starting rules
        return run(parse_args(argv))


def rows_of(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def picture(world):
    """Everything about a world that a save must keep."""
    return {
        "creatures": [(c.id, c.slot, tuple(c.p), tuple(c.facing), c.size, c.policy, c.parent_id, c.generation, c.energy, c.age,
                       c.lifespan, c.food_eaten, c.last_action, tuple(c.intended_vector), c.earned, c.spent, c.moved,
                       None if c.brain is None else tuple(a.tobytes() for a in (c.brain.w1, c.brain.b1, c.brain.w2, c.brain.b2)))
                      for c in world.creatures],
        "foods": [(tuple(f.p), f.energy, f.remaining_duration, f.radius) for f in world.foods],
        "counters": (world.step_count, world.births, world.newcomers, dict(world.deaths), world.total_bites,
                     world.energy_earned, world.energy_spent, world.creature_steps, world.next_id()),
        "settings": (world.settings, world.level, world.species, world.inherit),
        "random": (world.rng.random(), float(world.np_rng.random()), world.food_rng.random()),
    }


class TestSaves(unittest.TestCase):
    def test_save_and_load_give_the_same_world(self):
        world = make_world(replace(START, max_food=50), seed=3)
        populate(world, 25)
        world.prefill_food()
        for _ in range(300):
            world.step()
        world.set_settings(world.settings, 2.5)
        dial = Dial(world.settings)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, dial, {"note": "kept"})
            loaded, loaded_dial, extra = load_world(path, learner_seed=0)
        self.assertEqual(extra, {"note": "kept"})
        self.assertEqual(loaded_dial.settings, dial.settings)
        self.assertEqual(picture(loaded), picture(world))

    def test_a_resumed_world_goes_on_as_the_old_one_did(self):
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            whole = rows_of(quiet_run(["--seed", "4", "--steps", "800", "--species", "instinct", "--out", old]))
            save = Path(old) / "saves" / "step-0000400.npz"
            resumed = rows_of(quiet_run(["--resume", str(save), "--steps", "800", "--out", new]))
        self.assertEqual(resumed, whole)  # the first 2,000 steps copied, the next 2,000 lived again the same way

    def test_learners_keep_what_they_learned(self):
        learner = Learner(MAX_CREATURES, 0)
        world = make_world(learner=learner, species="learner", seed=1)
        populate(world, 12)
        for creature in world.creatures[:3]:
            live(learner, creature.slot, 65)
        learner.update()
        freed = world.creatures[5].slot
        learner.release(freed)  # a free slot in the middle
        world.creatures.pop(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            loaded, _, _ = load_world(path, learner_seed=0)
        before, after = learner.state(), loaded.learner.state()
        self.assertEqual(sorted(before), sorted(after))
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        self.assertEqual([c.slot for c in loaded.creatures], [c.slot for c in world.creatures])
        self.assertEqual(loaded.learner.allocate(learner.inborn(0), 1e-3), freed)  # the free slot is still free

    def test_auto_keeps_its_nudges(self):
        world = make_world(seed=2)
        dial = Dial()
        for _ in range(3):
            dial.review(TARGET_ALIVE + 8, 0.0, 0.0, 0)
        world.set_settings(dial.settings, dial.level)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, dial, {})
            _, loaded, _ = load_world(path, learner_seed=0)
        for kept in (dial, loaded):
            kept.review(TARGET_ALIVE - 8, 0.0, 0.0, 0)  # take one back
            kept.review(TARGET_ALIVE + 8, 0.0, 0.0, 0)  # and the next harder one comes from the same turn
        self.assertEqual(loaded.settings, dial.settings)
        self.assertEqual(loaded.level, dial.level)

    def test_a_save_of_another_format_is_refused(self):
        world = make_world()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = SAVE_FORMAT + 1
            arrays["about"] = np.array(json.dumps(about))
            np.savez(path, **arrays)
            with self.assertRaisesRegex(SaveError, f"save format {SAVE_FORMAT + 1}; this code reads format {SAVE_FORMAT}"):
                load_world(path, learner_seed=0)
            (Path(tmp) / "broken.npz").write_text("not a save")
            with self.assertRaisesRegex(SaveError, "cannot be read"):
                load_world(Path(tmp) / "broken.npz", learner_seed=0)

    def test_saves_of_lists_the_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_folder = Path(tmp)
            self.assertEqual(saves_of(run_folder), [])
            (run_folder / "saves").mkdir()
            for step in (4000, 2000, 16000):
                (run_folder / "saves" / f"step-{step:07d}.npz").write_bytes(b"")
            (run_folder / "saves" / "step-0004000.npz.tmp").write_bytes(b"")  # half written: not a save
            self.assertEqual(saves_of(run_folder), [2000, 4000, 16000])


    def test_a_save_keeps_the_last_steps_calls(self):
        world = make_world(seed=22)
        caller, gone = add(world, 10, 10), add(world, 20, 10)
        caller.calls = gone.calls = 3
        caller.apply_action(("CALL", (2, 0.5)), world)
        gone.apply_action(("CALL", (1, 1.0)), world)
        world.creatures.remove(gone)  # it called, then died: its call is still heard next step
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            loaded, _, _ = load_world(path, learner_seed=0)
        kept = [(c.id, tuple(np.round(c.p, 12)), kind, volume) for c, kind, volume in loaded.calls]
        self.assertEqual(kept, [(caller.id, tuple(np.round(caller.p, 12)), 2, 0.5), (gone.id, tuple(np.round(gone.p, 12)), 1, 1.0)])

    def test_a_sphere_save_resumes_the_same_world(self):
        world = make_world(replace(START, max_food=200), seed=23)
        world.rules["planet_radius"] = 12.0
        world.terrain = make_terrain({**world.rules, "obstacles": 6.0, "trees": 8.0, "forests": 2.0}, random.Random(2), 12.0)
        populate(world, 25)
        world.prefill_food()
        for _ in range(30):
            world.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            twin, _, _ = load_world(path, learner_seed=0)
        for _ in range(50):
            world.step()
            twin.step()
        self.assertEqual(picture(twin), picture(world))

    def test_a_flat_save_is_refused_in_plain_words(self):
        world = make_world(seed=24)
        populate(world, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 26
            arrays["about"] = np.array(json.dumps(about))
            np.savez(path, **arrays)
            with self.assertRaises(SaveError) as refused:
                load_world(path, learner_seed=0)
            self.assertIn(FLAT_WORLD, str(refused.exception))
            self.assertEqual(save_format(path), 26)

    def test_rules_are_kept_in_a_save(self):
        world = make_world(seed=8)
        populate(world, 3)
        world.rules["eat_miss"], world.rules["gene.size.high"] = 0.7, 1.6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            loaded, _, _ = load_world(path, learner_seed=0)
        self.assertEqual((loaded.rules["eat_miss"], loaded.rules["gene.size.high"]), (0.7, 1.6))
        self.assertEqual(loaded.rules, world.rules)


if __name__ == "__main__":
    unittest.main()
