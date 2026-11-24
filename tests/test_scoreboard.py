import unittest

from src.exam import ExamResult
from src.food import Food
from src.scoreboard import ROW_KEYS, Scoreboard
from src.senses import sense
from tests.helpers import add, make_world

STILL = ("MOVE", (0.0, 0.0))


def food_world():
    w = make_world()
    me = add(w, 10, 10)
    w.foods.append(Food(12, 10, remaining_duration=-1, energy=10.0))
    return w, me


def hunter_world():
    w = make_world()
    me = add(w, 10, 10, size=1.0)
    add(w, 12, 10, size=2.0)
    return w, me


class TestScoreboard(unittest.TestCase):
    def test_straight_at_food_scores_one(self):
        w, me = food_world()
        board = Scoreboard()
        board.observe(me, sense(me, w), ("MOVE", (1.0, 0.0)))
        self.assertAlmostEqual(board.row(w, None, False)["toward_food"], 1.0)

    def test_sideways_scores_zero(self):
        w, me = food_world()
        board = Scoreboard()
        board.observe(me, sense(me, w), ("MOVE", (0.0, 1.0)))
        self.assertAlmostEqual(board.row(w, None, False)["toward_food"], 0.0)

    def test_flee_counts_moving_away(self):
        w, me = hunter_world()
        board = Scoreboard()
        s = sense(me, w)
        board.observe(me, s, ("MOVE", (-1.0, 0.0)))
        board.observe(me, s, ("MOVE", (1.0, 0.0)))
        self.assertAlmostEqual(board.row(w, None, False)["flee"], 0.5)

    def test_standing_still_beside_a_hunter_is_not_fleeing(self):
        w, me = hunter_world()
        board = Scoreboard()
        board.observe(me, sense(me, w), STILL)
        self.assertEqual(board.row(w, None, False)["flee"], 0.0)

    def test_concentration_one_direction_is_one(self):
        w, me = food_world()
        board = Scoreboard()
        for _ in range(3):
            board.observe(me, sense(me, w), ("MOVE", (0.5, 0.0)))
        self.assertAlmostEqual(board.row(w, None, False)["direction_concentration"], 1.0)

    def test_concentration_four_directions_is_zero(self):
        w, me = food_world()
        board = Scoreboard()
        for move in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
            board.observe(me, sense(me, w), ("MOVE", move))
        self.assertAlmostEqual(board.row(w, None, False)["direction_concentration"], 0.0)

    def test_no_samples_is_none(self):
        row = Scoreboard().row(make_world(), None, False)
        for key in ("energy", "bites_per_creature_step", "toward_food", "flee", "direction_concentration",
                    "efficiency", "intelligence", "wanderers", "generation"):
            self.assertIsNone(row[key], key)

    def test_row_reports_efficiency_and_average_energy(self):
        w = make_world()
        half_full = add(w, 10, 10, size=1.0, energy=50.0)  # 50 of 100
        quarter_full = add(w, 30, 10, size=2.0, energy=50.0)  # 50 of 200
        board = Scoreboard()
        for creature in (half_full, quarter_full):
            board.observe(creature, sense(creature, w), STILL)
        w.energy_earned, w.energy_spent = 6.0, 4.0
        first = board.row(w, None, False)
        self.assertAlmostEqual(first["energy"], 0.375)  # each creature's share of its own maximum, averaged
        self.assertAlmostEqual(first["efficiency"], 1.5)  # energy earned ÷ energy spent in the window
        board.observe(half_full, sense(half_full, w), STILL)
        w.energy_earned, w.energy_spent = 7.0, 8.0
        second = board.row(w, None, False)
        self.assertAlmostEqual(second["energy"], 0.5)
        self.assertAlmostEqual(second["efficiency"], 0.25)  # 1 earned and 4 spent since the first row

    def test_row_resets_window(self):
        w, me = food_world()
        board = Scoreboard()
        board.observe(me, sense(me, w), ("MOVE", (1.0, 0.0)))
        w.step()
        w.births, w.newcomers = 3, 2
        first = board.row(w, ExamResult(1.7, 0.6), False)
        second = board.row(w, None, False)
        self.assertIsNotNone(first["toward_food"])
        self.assertIsNotNone(first["bites_per_creature_step"])
        self.assertEqual((first["intelligence"], first["wanderers"], first["births"], first["newcomers"]),
                         (1.7, 0.6, 3, 2))
        self.assertIsNone(second["toward_food"])
        self.assertEqual((second["intelligence"], second["births"], second["newcomers"]), (None, 0, 0))

    def test_row_has_exact_keys(self):
        self.assertEqual(ROW_KEYS, (
            "step", "level", "population", "energy", "births", "newcomers", "generation", "bites_per_creature_step",
            "toward_food", "flee", "direction_concentration", "efficiency", "intelligence", "wanderers",
            "deaths_hunger", "deaths_attack", "deaths_age", "max_food", "food_energy", "food_per_step", "move_cost",
            "step_cost", "lifespan", "hunters", "manual", "chase", "attacks", "vision_width", "vision_reach", "hearing",
            "fear_of_death", "love_of_babies", "learns_from_others", "neurons", "layers",
        ))
        w, me = food_world()
        me.generation = 4
        row = Scoreboard().row(w, None, False)
        self.assertEqual(tuple(row.keys()), ROW_KEYS)
        self.assertEqual((row["level"], row["population"], row["generation"]), (0, 1, 4))

    def test_row_shows_the_settings_in_force(self):
        w, _ = food_world()
        w.set_settings(w.settings, 7)
        row = Scoreboard().row(w, None, False)
        self.assertEqual(row["level"], 7)
        self.assertEqual((row["max_food"], row["food_energy"], row["food_per_step"]), (0, 30.0, 0))  # ranges: the middle
        self.assertEqual((row["move_cost"], row["step_cost"], row["lifespan"], row["hunters"]), (0.5, 0.05, 5000.0, 0))
        self.assertEqual((row["manual"], Scoreboard().row(w, None, True)["manual"]), (0, 1))


    def test_chase_counts_moving_toward_smaller(self):
        w = make_world()
        me = add(w, 10, 10, size=1.5)
        add(w, 12, 10, size=0.6)  # at most two-thirds my size: prey
        board = Scoreboard()
        s = sense(me, w)
        board.observe(me, s, ("MOVE", (1.0, 0.0)))  # toward it
        board.observe(me, s, ("MOVE", (-1.0, 0.0)))  # away
        board.observe(me, s, STILL)
        self.assertAlmostEqual(board.row(w, None, False)["chase"], 1 / 3)

    def test_attacks_are_counted(self):
        w, me = hunter_world()
        board = Scoreboard()
        s = sense(me, w)
        board.observe(me, s, ("ATTACK", w.creatures[1]))
        board.observe(me, s, ("ATTACK", w.creatures[1]))
        board.observe(me, s, STILL)
        first = board.row(w, None, False)
        self.assertEqual(first["attacks"], 2)
        self.assertEqual(board.row(w, None, False)["attacks"], 0)

    def test_scoreboard_has_the_senses(self):
        import csv
        import tempfile
        from pathlib import Path
        from src.scoreboard import migrate_scoreboard
        w = make_world()
        a, b = add(w, 10, 10), add(w, 20, 10)
        a.vision_width, a.vision_reach, a.hearing = 100.0, 2.0, 1.0
        b.vision_width, b.vision_reach, b.hearing = 300.0, 4.0, 2.0
        row = Scoreboard().row(w, None, False)
        self.assertEqual((row["vision_width"], row["vision_reach"], row["hearing"]), (200.0, 3.0, 1.5))
        from src.scoreboard import BEFORE_SENSES
        old_keys = list(BEFORE_SENSES)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoreboard.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=old_keys)
                writer.writeheader()
                writer.writerow({k: "1" for k in old_keys})
            self.assertTrue(migrate_scoreboard(path))
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(tuple(rows[0]), ROW_KEYS)
        self.assertEqual((rows[0]["hearing"], rows[0]["chase"]), ("", "1"))

    def test_scoreboard_has_the_values(self):
        import csv
        import tempfile
        from pathlib import Path
        from src.scoreboard import BEFORE_VALUES, migrate_scoreboard  # noqa: F401
        w = make_world()
        a, b = add(w, 10, 10), add(w, 20, 10)
        a.fear_of_death, a.love_of_babies, a.learns_from_others = 4.0, 8.0, 0.2
        b.fear_of_death, b.love_of_babies, b.learns_from_others = 6.0, 12.0, 0.4
        row = Scoreboard().row(w, None, False)
        self.assertEqual((row["fear_of_death"], row["love_of_babies"]), (5.0, 10.0))
        self.assertAlmostEqual(row["learns_from_others"], 0.3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoreboard.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(BEFORE_VALUES))
                writer.writeheader()
                writer.writerow({k: "1" for k in BEFORE_VALUES})
            self.assertTrue(migrate_scoreboard(path))
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(tuple(rows[0]), ROW_KEYS)
        self.assertEqual((rows[0]["fear_of_death"], rows[0]["hearing"]), ("", "1"))

    def test_scoreboard_reports_brain_size(self):
        w = make_world()
        a, b = add(w, 10, 10), add(w, 20, 10)
        a.neurons, a.layers, b.neurons, b.layers = 32, 1, 64, 2
        row = Scoreboard().row(w, None, False)
        self.assertEqual((row["neurons"], row["layers"]), (48.0, 1.5))

    def test_old_scoreboards_get_the_new_columns(self):
        import csv
        import tempfile
        from pathlib import Path
        from src.scoreboard import migrate_scoreboard
        from src.scoreboard import BEFORE_HUNTING
        old_keys = list(BEFORE_HUNTING)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scoreboard.csv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=old_keys)
                writer.writeheader()
                writer.writerow({k: "1" for k in old_keys})
            self.assertTrue(migrate_scoreboard(path))
            self.assertFalse(migrate_scoreboard(path))  # already in today's columns
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(tuple(rows[0]), ROW_KEYS)
        self.assertEqual((rows[0]["chase"], rows[0]["attacks"], rows[0]["step"]), ("", "", "1"))

if __name__ == "__main__":
    unittest.main()
