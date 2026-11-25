import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.dial import Dial
from src.evolve import parse_args, run
from src.learner import Learner
from src.saves import SAVE_FORMAT, SaveError, load_world, save_world, saves_of
from src.rules import ROOM_SLOTS as MAX_CREATURES
from src.settings import START
from tests.helpers import make_world, populate
from tests.test_learner import live
from src.rules import rule

TARGET_ALIVE = rule("auto_target").start


def before_rules(about):
    """A save from before format 6: the world's size instead of its rules."""
    rules = about.pop("rules")
    return {**about, "width": rules["world_width"], "height": rules["world_height"]}


def quiet_run(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return run(parse_args(argv))


def rows_of(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def picture(world):
    """Everything about a world that a save must keep."""
    return {
        "creatures": [(c.id, c.slot, c.x, c.y, c.size, c.policy, c.parent_id, c.generation, c.energy, c.age,
                       c.lifespan, c.food_eaten, c.last_action, tuple(c.intended_vector), c.earned, c.spent,
                       None if c.brain is None else tuple(a.tobytes() for a in (c.brain.w1, c.brain.b1, c.brain.w2, c.brain.b2)))
                      for c in world.creatures],
        "foods": [(f.x, f.y, f.energy, f.remaining_duration, f.radius) for f in world.foods],
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
            whole = rows_of(quiet_run(["--seed", "4", "--steps", "4000", "--species", "instinct", "--out", old]))
            save = Path(old) / "saves" / "step-0002000.npz"
            resumed = rows_of(quiet_run(["--resume", str(save), "--steps", "4000", "--out", new]))
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


    def test_a_format_1_save_is_converted(self):
        world = make_world()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 1
            about = before_rules(about)
            del about["dial"]["hunters"]  # what a format 1 save held
            arrays["about"] = np.array(json.dumps(about))
            np.savez(path, **arrays)
            _, dial, _ = load_world(path, learner_seed=0)
        self.assertEqual(dial.state()["hunters"], None)

    def test_a_format_2_save_is_converted(self):
        world = make_world(seed=3)
        populate(world, 5)
        for _ in range(20):
            world.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 2
            about = before_rules(about)
            for creature in about["creatures"]:  # what a format 2 save held
                for key in ("earned", "spent", "vision_width", "vision_reach", "hearing", "facing", "fear_of_death",
                            "love_of_babies", "learns_from_others", "hurt"):
                    del creature[key]
            arrays["about"] = np.array(json.dumps(about))
            arrays["brain.w1"] = arrays["brain.w1"][:, :12, :]  # 12 inputs then
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        self.assertEqual({(c.earned, c.spent) for c in loaded.creatures}, {(0.0, 0.0)})
        self.assertEqual([c.id for c in loaded.creatures], [c.id for c in world.creatures])

    def test_a_format_3_save_gets_todays_senses(self):
        for species in ("brain", "learner"):
            learner = Learner(MAX_CREATURES, 0) if species == "learner" else None
            world = make_world(seed=3, learner=learner, species=species)
            populate(world, 5)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "step.npz"
                save_world(path, world, Dial(), {})
                with zipfile.ZipFile(path) as z:
                    arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
                about = json.loads(str(arrays["about"]))
                about["format"] = 3
                about = before_rules(about)
                for creature in about["creatures"]:  # what a format 3 save held
                    for key in ("vision_width", "vision_reach", "hearing", "facing", "fear_of_death", "love_of_babies",
                                "learns_from_others", "hurt"):
                        del creature[key]
                arrays["about"] = np.array(json.dumps(about))
                for name in list(arrays):  # 12 inputs then
                    if name in ("brain.w1",) or name.split(".")[-1] in ("w1", "wh") and name.startswith("learner."):
                        arrays[name] = arrays[name][..., :12, :]
                old = {name: arrays[name] for name in arrays if name != "about"}
                np.savez(path, **arrays)
                loaded, _, _ = load_world(path, learner_seed=0)
            for c in loaded.creatures:  # today's senses then, narrowed later to 120° seeing as much ground
                self.assertEqual((c.vision_width, c.hearing), (120.0, 1.5))
                self.assertAlmostEqual(c.vision_reach, 3.0 * math.sqrt(3.0))
            if species == "learner":
                for part in ("w1", "wh"):
                    weights = loaded.learner.state()[f"params.{part}"]
                    self.assertEqual(weights.shape[-2], 20)
                    self.assertTrue(np.allclose(weights[..., 12:, :], 0.0))
                    self.assertTrue(np.allclose(weights[..., :12, :], old[f"learner.params.{part}"]))

    def test_a_format_4_save_gets_the_starting_values(self):
        world = make_world(seed=6)
        populate(world, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 4
            about = before_rules(about)
            for creature in about["creatures"]:  # what a format 4 save held
                for key in ("fear_of_death", "love_of_babies", "learns_from_others", "hurt"):
                    del creature[key]
            arrays["about"] = np.array(json.dumps(about))
            arrays["brain.w1"] = arrays["brain.w1"][:, :17, :]  # 17 inputs then
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        self.assertEqual({(c.fear_of_death, c.love_of_babies, c.learns_from_others) for c in loaded.creatures},
                         {(10.0, 10.0, 0.5)})

    def test_the_owners_world_converts(self):
        world = make_world(seed=10)
        populate(world, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 6  # what a format 6 save held: cones to 360, one sense price, 21 inputs with a heard size
            rules = about["rules"]
            sense = rules.pop("vision_cost")
            del rules["hearing_cost"], rules["sprint_power"]
            rules.update({"sense_cost": sense, "gene.vision_width.high": 360.0, "gene.vision_width.new_high": 360.0,
                          "gene.vision_reach.high": 8.0, "gene.vision_reach.new_high": 8.0})
            about["creatures"][0].update({"vision_width": 360.0, "vision_reach": 3.0})
            about["creatures"][1].update({"vision_width": 100.0, "vision_reach": 4.0})
            arrays["about"] = np.array(json.dumps(about))
            w1 = arrays["brain.w1"]
            old = np.concatenate([w1[:, :16], np.full((w1.shape[0], 1, w1.shape[2]), 0.7), w1[:, 16:]], axis=1)
            arrays["brain.w1"] = old  # 21 inputs then: the heard size was input 17
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        first, second = loaded.creatures[0], loaded.creatures[1]
        self.assertEqual(first.vision_width, 120.0)
        self.assertAlmostEqual(first.vision_reach, 3.0 * math.sqrt(3.0))  # narrowed, it sees as much ground as before
        self.assertEqual((second.vision_width, second.vision_reach), (100.0, 4.0))  # within 120: untouched
        self.assertEqual(first.brain.w1.shape, (20, 16))
        self.assertTrue(np.allclose(first.brain.w1, w1[0]))  # the heard size's weights gone, the rest kept
        self.assertEqual((loaded.rules["vision_cost"], loaded.rules["hearing_cost"]), (sense, sense))
        self.assertEqual((loaded.rules["gene.vision_width.high"], loaded.rules["sprint_power"]), (120.0, 2.0))

    def test_a_format_9_save_drops_the_threat_rule(self):
        world = make_world(seed=3)
        populate(world, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"], about["rules"]["threat_ratio"] = 9, 1.5
            arrays["about"] = np.array(json.dumps(about))
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        self.assertNotIn("threat_ratio", loaded.rules)

    def test_a_format_10_save_gets_one_reach(self):
        world = make_world(seed=3)
        populate(world, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            del about["rules"]["reach"]
            about["format"], about["rules"]["eat_range"], about["rules"]["attack_range"] = 10, 2.5, 2.5
            arrays["about"] = np.array(json.dumps(about))
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        self.assertEqual(loaded.rules["reach"], 2.5)
        self.assertFalse({"eat_range", "attack_range"} & set(loaded.rules))

    def test_a_format_11_save_gets_the_mutation_genes(self):
        world = make_world(seed=3)
        populate(world, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 11
            for c in about["creatures"]:
                del c["mutations"], c["mutation_size"]
            about["rules"] = {k: v for k, v in about["rules"].items() if not k.startswith(("gene.mutations", "gene.mutation_size"))}
            about["rules"]["gene.size.change"] = 0.05
            arrays["about"] = np.array(json.dumps(about))
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        self.assertEqual({(c.mutations, c.mutation_size) for c in loaded.creatures}, {(1, 0.05)})
        self.assertNotIn("gene.size.change", loaded.rules)
        self.assertIn("gene.mutations.new_high", loaded.rules)

    def test_a_format_12_save_draws_every_newcomer_gene(self):
        world = make_world(seed=3)
        populate(world, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 12
            about["rules"] = {k: v for k, v in about["rules"].items() if not k.startswith("gene.fear.new_")}
            about["rules"]["gene.fear.start"] = 10.0
            arrays["about"] = np.array(json.dumps(about))
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        self.assertNotIn("gene.fear.start", loaded.rules)
        self.assertLess(loaded.rules["gene.fear.new_low"], loaded.rules["gene.fear.new_high"])

    def test_a_format_7_save_gets_room_to_grow(self):
        learner = Learner(MAX_CREATURES, 0)
        world = make_world(seed=11, learner=learner, species="learner")
        populate(world, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 7  # a format 7 brain: 1 layer of 32, no room to grow, no brain genes
            about["rules"] = {k: v for k, v in about["rules"].items()
                              if not k.startswith(("gene.neurons", "gene.layers", "neuron_cost", "brain_layer_share"))}
            arrays["about"] = np.array(json.dumps(about))
            for kind in ("params", "m", "v", "inborn"):
                arrays[f"learner.{kind}.w1"] = arrays[f"learner.{kind}.w1"][..., :32]
                arrays[f"learner.{kind}.b1"] = arrays[f"learner.{kind}.b1"][..., :32]
                arrays[f"learner.{kind}.wc"] = arrays[f"learner.{kind}.wc"][..., :32, :]
                arrays[f"learner.{kind}.wm"] = arrays[f"learner.{kind}.wm"][..., :32, :]
                for name in ("w2", "b2", "s2", "w3", "b3", "s3"):
                    del arrays[f"learner.{kind}.{name}"]
            del arrays["learner.size"], arrays["learner.inborn_size"]
            old_w1 = arrays["learner.params.w1"]
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        for c in loaded.creatures:
            brain = loaded.learner.current(c.slot)
            self.assertEqual(list(brain.size), [1, 32, 0, 0])
            self.assertEqual(brain.w1.shape, (20, 64))
            self.assertTrue(np.allclose(brain.w1[:, :32], old_w1[c.slot]))  # the old brain, room to grow beside it
            self.assertTrue(np.allclose(brain.w1[:, 32:], 0.0))
            self.assertEqual(c.neurons, 32)
        self.assertEqual(loaded.rules["neuron_cost"], 0.0005)

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

    def test_a_format_5_save_gets_the_starting_rules(self):
        from src.rules import starting_rules
        world = make_world(seed=9)
        populate(world, 3)
        world.rules["eat_miss"] = 0.7
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 5
            about = before_rules(about)
            arrays["about"] = np.array(json.dumps(about))
            arrays["brain.w1"] = np.insert(arrays["brain.w1"], 16, 0.0, axis=1)  # 21 inputs then: a heard size
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        self.assertEqual(loaded.rules, starting_rules())

    def test_a_format_4_brain_gets_four_silent_pain_inputs(self):
        world = make_world(seed=7)
        populate(world, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 4
            about = before_rules(about)
            for creature in about["creatures"]:  # what a format 4 save held
                for key in ("fear_of_death", "love_of_babies", "learns_from_others", "hurt"):
                    del creature[key]
            arrays["about"] = np.array(json.dumps(about))
            arrays["brain.w1"] = arrays["brain.w1"][:, :17, :]  # 17 inputs then
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        for i, creature in enumerate(loaded.creatures):
            self.assertEqual(creature.brain.w1.shape, (20, 16))
            self.assertTrue(np.allclose(creature.brain.w1[16:], 0.0))  # pain, silent
            self.assertTrue(np.allclose(creature.brain.w1[:16], arrays["brain.w1"][i][:16]))  # the heard size left later
            self.assertIsNone(creature.hurt)

    def test_wider_brains_act_as_before_with_zero_new_weights(self):
        world = make_world(seed=5)
        populate(world, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.npz"
            save_world(path, world, Dial(), {})
            with zipfile.ZipFile(path) as z:
                arrays = {name[:-4]: np.load(z.open(name)) for name in z.namelist()}
            about = json.loads(str(arrays["about"]))
            about["format"] = 3
            about = before_rules(about)
            for creature in about["creatures"]:
                for key in ("vision_width", "vision_reach", "hearing", "facing", "fear_of_death", "love_of_babies",
                                "learns_from_others", "hurt"):
                    del creature[key]
            arrays["about"] = np.array(json.dumps(about))
            arrays["brain.w1"] = arrays["brain.w1"][:, :12, :]
            np.savez(path, **arrays)
            loaded, _, _ = load_world(path, learner_seed=0)
        rng = np.random.default_rng(0)
        for i, creature in enumerate(loaded.creatures):
            x12 = rng.uniform(-1, 1, 12)
            w1, b1, w2, b2 = (arrays[f"brain.{part}"][i] for part in ("w1", "b1", "w2", "b2"))
            before = np.tanh(np.tanh(x12 @ w1 + b1) @ w2 + b2)
            now = creature.brain.forward(np.concatenate([x12, rng.uniform(-1, 1, 8)]))  # heard or felt: no effect yet
            self.assertTrue(np.allclose(before, now))


if __name__ == "__main__":
    unittest.main()
