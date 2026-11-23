import csv
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

from src.check_proof import check
from src.scoreboard import ROW_KEYS

ROWS = 100  # 100,000 steps


def brain_rows(exam: float = 2.5, first_exam: float = 0.5, toward: float = 0.8, flee: float = 0.8,
               level_four_at: int = 50, alive_at_end: bool = True) -> List[Dict]:
    rows = []
    for i in range(ROWS):
        step = (i + 1) * 1000
        has_exam = step % 2000 == 0
        exam_value = first_exam if i < 10 else exam
        rows.append({
            "step": step, "level": 4 if i >= level_four_at else 1, "population": 30, "round": i + 1,
            "bites_per_creature_step": 0.3, "toward_food": toward, "flee": flee, "direction_concentration": 0.1,
            "exam_ratio": exam_value if has_exam else "", "deaths_hunger": 3, "deaths_attack": 0,
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
        result = verdicts(five())
        self.assertEqual([result[k] for k in sorted(result)][:4], [True, True, True, True])

    def test_low_exam_ratio_fails(self):
        self.assertFalse(verdicts(five(exam=1.8))["1 exam ratio"])

    def test_flat_exam_ratio_fails(self):
        self.assertFalse(verdicts(five(exam=2.5, first_exam=2.5))["1 exam ratio"])

    def test_low_toward_food_fails(self):
        self.assertFalse(verdicts(five(toward=0.6))["2 toward food"])

    def test_three_seeds_at_level_four_fails(self):
        seeds = five()
        seeds[0] = brain_rows(level_four_at=ROWS)
        seeds[1] = brain_rows(level_four_at=ROWS)
        self.assertFalse(verdicts(seeds)["3 flee at level 4+"])

    def test_low_flee_fails(self):
        self.assertFalse(verdicts(five(flee=0.6))["3 flee at level 4+"])

    def test_extinct_at_end_fails(self):
        self.assertFalse(verdicts(five(alive_at_end=False))["4 survives"])

    def test_rules_reported_not_judged(self):
        self.assertIsNone(verdicts(five())["5 rules yardstick"])


if __name__ == "__main__":
    unittest.main()
