import unittest

from src.food import Food
from src.scoreboard import ROW_KEYS, Scoreboard
from src.senses import sense
from tests.helpers import add, make_world


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
        self.assertAlmostEqual(board.row(w, None)["toward_food"], 1.0)

    def test_sideways_scores_zero(self):
        w, me = food_world()
        board = Scoreboard()
        board.observe(me, sense(me, w), ("MOVE", (0.0, 1.0)))
        self.assertAlmostEqual(board.row(w, None)["toward_food"], 0.0)

    def test_flee_counts_moving_away(self):
        w, me = hunter_world()
        board = Scoreboard()
        s = sense(me, w)
        board.observe(me, s, ("MOVE", (-1.0, 0.0)))
        board.observe(me, s, ("MOVE", (1.0, 0.0)))
        self.assertAlmostEqual(board.row(w, None)["flee"], 0.5)

    def test_standing_still_beside_a_hunter_is_not_fleeing(self):
        w, me = hunter_world()
        board = Scoreboard()
        board.observe(me, sense(me, w), ("MOVE", (0.0, 0.0)))
        self.assertEqual(board.row(w, None)["flee"], 0.0)

    def test_concentration_one_direction_is_one(self):
        w, me = food_world()
        board = Scoreboard()
        for _ in range(3):
            board.observe(me, sense(me, w), ("MOVE", (0.5, 0.0)))
        self.assertAlmostEqual(board.row(w, None)["direction_concentration"], 1.0)

    def test_concentration_four_directions_is_zero(self):
        w, me = food_world()
        board = Scoreboard()
        for move in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
            board.observe(me, sense(me, w), ("MOVE", move))
        self.assertAlmostEqual(board.row(w, None)["direction_concentration"], 0.0)

    def test_no_samples_is_none(self):
        row = Scoreboard().row(make_world(), None)
        for key in ("bites_per_creature_step", "toward_food", "flee", "direction_concentration", "exam_ratio",
                    "generation"):
            self.assertIsNone(row[key], key)

    def test_row_resets_window(self):
        w, me = food_world()
        board = Scoreboard()
        board.observe(me, sense(me, w), ("MOVE", (1.0, 0.0)))
        w.step()
        w.births, w.newcomers = 3, 2
        first = board.row(w, 1.7)
        second = board.row(w, None)
        self.assertIsNotNone(first["toward_food"])
        self.assertIsNotNone(first["bites_per_creature_step"])
        self.assertEqual((first["exam_ratio"], first["births"], first["newcomers"]), (1.7, 3, 2))
        self.assertIsNone(second["toward_food"])
        self.assertEqual((second["births"], second["newcomers"]), (0, 0))

    def test_row_has_exact_keys(self):
        self.assertEqual(ROW_KEYS, (
            "step", "stage", "population", "births", "newcomers", "generation", "bites_per_creature_step",
            "toward_food", "flee", "direction_concentration", "exam_ratio", "deaths_hunger", "deaths_attack",
            "deaths_age",
        ))
        w, me = food_world()
        me.generation = 4
        row = Scoreboard().row(w, None)
        self.assertEqual(tuple(row.keys()), ROW_KEYS)
        self.assertEqual((row["stage"], row["population"], row["generation"]), (1, 1, 4))


if __name__ == "__main__":
    unittest.main()
