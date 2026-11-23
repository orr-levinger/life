"""Compare teams: one row per team, averaged over its seeds, from the runs' scoreboards.

    python -m src.compare --team evolution runs/1-evolution/scoreboard.csv ... --team both runs/1-both/scoreboard.csv ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Union

from .check_proof import EXAMS_COMPARED, FLEE_LEVEL, TOWARD_FOOD_ROWS
from .scoreboard import numbers, read_rows

Cell = Union[str, int, float, None]
COLUMNS = ("team", "seeds", "exam_ratio", "toward_food", "flee", "highest_level", "minutes")


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    known = [value for value in values if value is not None]
    return mean(known) if known else None


def _run_minutes(csv_path: Path) -> Optional[float]:
    timing = csv_path.parent / "time.txt"
    return float(timing.read_text()) / 60.0 if timing.exists() else None


def compare(runs: Dict[str, Sequence[Path]]) -> List[Dict[str, Cell]]:
    """Per team, the mean over its seeds of: the exam ratio over the last 5 exams, the toward-food score over the
    last 10 rows, the flee score over rows at level 4 or above; plus the highest level any seed reached and the
    minutes a run took. A score no seed could measure is None."""
    table = []
    for team, paths in runs.items():
        exam, toward, flee, levels, minutes = [], [], [], [], []
        for path in paths:
            rows = read_rows(path)
            exam.append(_mean(numbers(rows, "exam_ratio")[-EXAMS_COMPARED:]))
            toward.append(_mean(numbers(rows[-TOWARD_FOOD_ROWS:], "toward_food")))
            flee.append(_mean(numbers([r for r in rows if int(r["level"]) >= FLEE_LEVEL], "flee")))
            levels.append(max(int(r["level"]) for r in rows))
            minutes.append(_run_minutes(path))
        table.append({"team": team, "seeds": len(paths), "exam_ratio": _mean(exam), "toward_food": _mean(toward),
                      "flee": _mean(flee), "highest_level": max(levels), "minutes": _mean(minutes)})
    return table


def format_table(table: List[Dict[str, Cell]]) -> str:
    def show(value: Cell) -> str:
        return "-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = [" | ".join(f"{column:>13}" for column in COLUMNS)]
    lines += [" | ".join(f"{show(row[column]):>13}" for column in COLUMNS) for row in table]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare teams from their runs' scoreboards.")
    parser.add_argument("--team", nargs="+", action="append", required=True, metavar=("NAME", "CSV"),
                        help="a team's name followed by its runs' scoreboard.csv files; repeat per team")
    args = parser.parse_args(argv)
    runs = {name: [Path(p) for p in paths] for name, *paths in args.team}
    print(format_table(compare(runs)))


if __name__ == "__main__":
    main()
