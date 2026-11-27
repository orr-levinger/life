import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.control import (Control, knob_values, new_run_folder, read_control, settings_from, watch_speed_from,
                         write_control)
from src.dial import with_value
from src.settings import START


class TestControl(unittest.TestCase):
    def test_control_file_round_trip(self):
        mine = replace(with_value(with_value(START, "food_energy", 24), "lifespan", 7000), attacks=True)
        with tempfile.TemporaryDirectory() as tmp:
            write_control(Path(tmp), Control("manual", mine, False))
            back = read_control(Path(tmp))
            write_control(Path(tmp), Control("auto", None, True))
            stopped = read_control(Path(tmp))
            leftovers = sorted(p.name for p in Path(tmp).iterdir())
        self.assertEqual((back.mode, back.stop), ("manual", False))
        self.assertEqual(knob_values(back.settings), knob_values(mine))
        self.assertEqual(back.settings.food_energy, (16.0, 32.0))
        self.assertEqual(stopped, Control("auto", None, True))
        self.assertEqual(leftovers, ["control.json"])  # the temporary file was renamed into place

    def test_a_gene_change_round_trips(self):
        change = {"seq": 7, "id": 42, "genes": {"size": 1.2, "layers": 2, "neurons": [8, 4]}}
        with tempfile.TemporaryDirectory() as tmp:
            write_control(Path(tmp), Control("auto", None, False, edit=change))
            self.assertEqual(read_control(Path(tmp)).edit, change)

    def test_rule_changes_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_control(Path(tmp), Control("auto", None, False, rules={"diet_bite": 0.5, "tank": 120.0}))
            self.assertEqual(read_control(Path(tmp)).rules, {"diet_bite": 0.5, "tank": 120.0})
            write_control(Path(tmp), Control("auto", None, False))
            self.assertIsNone(read_control(Path(tmp)).rules)

    def test_settings_from_sets_only_the_named_knobs(self):
        settings = settings_from({"move_cost": 1.0, "hunters": True}, START)
        self.assertEqual(settings, replace(START, move_cost=1.0, attacks=True))

    def test_unknown_knob_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no knob called 'gravity'"):
            settings_from({"gravity": 1.0}, START)

    def test_value_outside_the_ends_is_refused(self):
        with self.assertRaisesRegex(ValueError, "food_energy must be between 3 and 100, not 120"):
            settings_from({"food_energy": 120}, START)
        with self.assertRaisesRegex(ValueError, "must be a number"):
            settings_from({"lifespan": "long"}, START)
        with self.assertRaisesRegex(ValueError, "must be a number"):
            settings_from({"lifespan": True}, START)
        with self.assertRaisesRegex(ValueError, "hunters must be true or false"):
            settings_from({"hunters": 1}, START)

    def test_manual_takes_values_within_the_worlds_ends(self):  # the world's knob ends (the owner, 2026-11-27)
        from src.rules import starting_rules
        rules = {**starting_rules(), "knob.fruit_rate.easy": 5.0}
        self.assertEqual(settings_from({"fruit_rate": 3.0}, START, rules).fruit_rate, 3.0)
        with self.assertRaisesRegex(ValueError, "fruit_rate must be between 0.002 and 5, not 6"):
            settings_from({"fruit_rate": 6.0}, START, rules)
        with self.assertRaisesRegex(ValueError, "between 0.002 and 1, not 3"):
            settings_from({"fruit_rate": 3.0}, START)  # a world on today's ends

    def test_a_bad_mode_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "mode must be one of"):
            write_control(Path(tmp), Control("turbo", None, False))

    def test_no_file_is_no_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_control(Path(tmp)))

    def test_run_folders_are_dated(self):
        folder = new_run_folder(Path("runs"), 3, "learning")
        self.assertRegex(folder.as_posix(), r"^runs/\d{8}-\d{6}-3-learning$")


    def test_watch_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_control(Path(tmp), Control("auto", None, False, watch_until=1234.5, watch_speed=20))
            back = read_control(Path(tmp))
            write_control(Path(tmp), Control("auto", None, False))
            unwatched = read_control(Path(tmp))
        self.assertEqual((back.watch_until, back.watch_speed), (1234.5, 20))
        self.assertEqual((unwatched.watch_until, unwatched.watch_speed), (None, 10))

    def test_watch_speed_outside_its_ends_is_refused(self):
        self.assertEqual(watch_speed_from(12), 12.0)
        with self.assertRaisesRegex(ValueError, "watch speed must be between 1 and 30 steps a second, not 50"):
            watch_speed_from(50)
        with self.assertRaisesRegex(ValueError, "watch speed must be a number"):
            watch_speed_from("fast")

    def test_paused_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_control(Path(tmp), Control("manual", None, False, paused=True))
            self.assertTrue(read_control(Path(tmp)).paused)
            write_control(Path(tmp), Control("manual", None, False))
            self.assertFalse(read_control(Path(tmp)).paused)

    def test_hunters_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_control(Path(tmp), Control("auto", None, False, hunters=True))
            self.assertIs(read_control(Path(tmp)).hunters, True)
            write_control(Path(tmp), Control("auto", None, False))
            self.assertIsNone(read_control(Path(tmp)).hunters)

if __name__ == "__main__":
    unittest.main()
