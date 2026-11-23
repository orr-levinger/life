import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.control import Control, knob_values, new_run_folder, read_control, settings_from, write_control
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

    def test_settings_from_sets_only_the_named_knobs(self):
        settings = settings_from({"move_cost": 1.0, "hunters": True}, START)
        self.assertEqual(settings, replace(START, move_cost=1.0, attacks=True))

    def test_unknown_knob_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no knob called 'gravity'"):
            settings_from({"gravity": 1.0}, START)

    def test_value_outside_the_ends_is_refused(self):
        with self.assertRaisesRegex(ValueError, "food_energy must be between 3 and 60, not 100"):
            settings_from({"food_energy": 100}, START)
        with self.assertRaisesRegex(ValueError, "must be a number"):
            settings_from({"lifespan": "long"}, START)
        with self.assertRaisesRegex(ValueError, "must be a number"):
            settings_from({"lifespan": True}, START)
        with self.assertRaisesRegex(ValueError, "hunters must be true or false"):
            settings_from({"hunters": 1}, START)

    def test_a_bad_mode_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "mode must be one of"):
            write_control(Path(tmp), Control("turbo", None, False))

    def test_no_file_is_no_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_control(Path(tmp)))

    def test_run_folders_are_dated(self):
        folder = new_run_folder(Path("runs"), 3, "learning")
        self.assertRegex(folder.as_posix(), r"^runs/\d{8}-\d{6}-3-learning$")


if __name__ == "__main__":
    unittest.main()
