import csv
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

from src.check_proof import check
from src.scoreboard import ROW_KEYS

ROWS = 100  # 100,000 steps


def brain_rows(exam: float = 2.5, first_exam: float = 0.5, toward: float = 0.8, flee: float = 0.8,
               hunters_at: int = 50, garden_rows: int = 0, generation_end: int = 40, late_newcomers: int = 0,
               alive_at_end: bool = True) -> List[Dict]:
    """A run's rows: `garden_rows` rows in the Garden (stage 0), then stage 1, then Hunters (stage 3) from `hunters_at`."""
    rows = []
    for i in range(ROWS):
        step = (i + 1) * 1000
        stage = 3 if i >= hunters_at else 0 if i < garden_rows else 1
        rows.append({
            "step": step, "stage": stage, "population": 30, "births": 5,
            "newcomers": late_newcomers if i >= ROWS - 5 else 0,
            "generation": round(generation_end * (i + 1) / ROWS),
            "bites_per_creature_step": 0.3, "toward_food": toward, "flee": flee, "direction_concentration": 0.1,
            "exam_ratio": (first_exam if i < 10 else exam) if step % 2000 == 0 else "",
            "deaths_hunger": 3, "deaths_attack": 0, "deaths_age": 1,
        })
    if not alive_at_end:
        rows[-1]["population"] = 0
    return rows


def write(folder: Path, name: str, rows: List[Dict]) -> Path:
    path = folder / f"{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def verdicts(seed_rows: List[List[Dict]]) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as folder:
        brains = [write(Path(folder), f"brain{i}", rows) for i, rows in enumerate(seed_rows)]
        rules = [write(Path(folder), "rules1", brain_rows())]
        return {name: ok for name, ok, _ in check(brains, rules)}


def five(**changes) -> List[List[Dict]]:
    return [brain_rows(**changes) for _ in range(5)]


class TestCheckProof(unittest.TestCase):
    def test_all_pass(self):
        result = verdicts(five(garden_rows=10))
        self.assertEqual([result[k] for k in sorted(result)][:6], [True] * 6)

    def test_garden_not_judged_when_the_run_started_mortal(self):
        self.assertIsNone(verdicts(five())["1 learns in the garden"])

    def test_stuck_in_the_garden_fails(self):
        self.assertFalse(verdicts(five(garden_rows=ROWS, hunters_at=ROWS))["1 learns in the garden"])

    def test_late_newcomers_fail(self):
        self.assertFalse(verdicts(five(late_newcomers=2))["2 sustains itself"])

    def test_extinct_at_end_fails(self):
        self.assertFalse(verdicts(five(alive_at_end=False))["2 sustains itself"])

    def test_few_generations_fail(self):
        self.assertFalse(verdicts(five(generation_end=12))["3 generations"])

    def test_low_exam_score_fails(self):
        self.assertFalse(verdicts(five(exam=1.8))["4 exam score"])

    def test_flat_exam_score_fails(self):
        self.assertFalse(verdicts(five(exam=2.5, first_exam=2.5))["4 exam score"])

    def test_low_steering_fails(self):
        self.assertFalse(verdicts(five(toward=0.6))["5 steers to food"])

    def test_three_seeds_reaching_hunters_fail(self):
        seeds = five()
        seeds[0] = brain_rows(hunters_at=ROWS)
        seeds[1] = brain_rows(hunters_at=ROWS)
        self.assertFalse(verdicts(seeds)["6 flees hunters"])

    def test_low_flee_fails(self):
        self.assertFalse(verdicts(five(flee=0.6))["6 flees hunters"])

    def test_rules_reported_not_judged(self):
        self.assertIsNone(verdicts(five())["7 rules yardstick"])


if __name__ == "__main__":
    unittest.main()
