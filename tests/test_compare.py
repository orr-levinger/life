import tempfile
import unittest
from pathlib import Path

from src.compare import compare, format_table
from tests.test_check_proof import ROWS, brain_rows, write


def run_folder(root: Path, name: str, rows, seconds=None) -> Path:
    folder = root / name
    folder.mkdir()
    path = write(folder, "scoreboard", rows)
    if seconds is not None:
        (folder / "time.txt").write_text(f"{seconds}\n")
    return path


class TestCompare(unittest.TestCase):
    def test_one_row_per_team(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = compare({
                "evolution": [run_folder(root, "1-evolution", brain_rows())],
                "both": [run_folder(root, "1-both", brain_rows())],
            })
        self.assertEqual([row["team"] for row in table], ["evolution", "both"])

    def test_means_over_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = compare({"both": [
                run_folder(root, "1-both", brain_rows(exam=2.0, toward=0.8, flee=0.9), seconds=120),
                run_folder(root, "2-both", brain_rows(exam=1.0, toward=0.6, flee=0.7, level_four_at=ROWS), seconds=240),
            ]})
        row = table[0]
        self.assertEqual(row["seeds"], 2)
        self.assertAlmostEqual(row["exam_ratio"], 1.5)
        self.assertAlmostEqual(row["toward_food"], 0.7)
        self.assertAlmostEqual(row["flee"], 0.9)  # only the seed that reached level 4 has a flee score there
        self.assertEqual(row["highest_level"], 4)
        self.assertAlmostEqual(row["minutes"], 3.0)

    def test_missing_values_show_dash(self):
        rows = brain_rows(level_four_at=ROWS)
        for r in rows:
            r["exam_ratio"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            table = compare({"learning": [run_folder(Path(tmp), "1-learning", rows)]})
        row = table[0]
        self.assertIsNone(row["exam_ratio"])
        self.assertIsNone(row["flee"])
        self.assertIsNone(row["minutes"])
        cells = [cell.strip() for cell in format_table(table).splitlines()[-1].split("|")]
        self.assertEqual(cells[0], "learning")
        self.assertEqual(cells.count("-"), 3)


if __name__ == "__main__":
    unittest.main()
