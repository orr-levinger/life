"""The scoreboard: numbers that say whether the creatures are getting smarter, one row per window of steps."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .senses import THREAT_RATIO, Senses

if TYPE_CHECKING:
    from .creature import Creature
    from .policies import Action
    from .world import World

ROW_KEYS = (
    "step", "level", "population", "round", "bites_per_creature_step", "toward_food", "flee",
    "direction_concentration", "exam_ratio", "deaths_hunger", "deaths_attack",
)
DEATH_CAUSES = ("hunger", "attack")
Value = Union[int, float, None]


def read_rows(path: Path) -> List[Dict[str, str]]:
    """A run's scoreboard CSV as rows of text; an empty cell is a score with nothing to measure."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def numbers(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row[key] != ""]


def _cosine(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax * bx + ay * by) / (math.hypot(ax, ay) * math.hypot(bx, by))


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


class Scoreboard:
    """Scores and deaths cover the window since the last row."""

    def __init__(self) -> None:
        self._clear_samples()
        self._bites_seen = 0
        self._steps_seen = 0
        self._deaths_seen = {cause: 0 for cause in DEATH_CAUSES}

    def _clear_samples(self) -> None:
        self._toward_food: List[float] = []
        self._fled: List[float] = []
        self._unit_moves: List[tuple[float, float]] = []

    def observe(self, creature: Creature, senses: Senses, action: Action) -> None:
        kind, param = action
        moving = kind == "MOVE" and math.hypot(*param) > 0
        if moving:
            dx, dy = param
            length = math.hypot(dx, dy)
            self._unit_moves.append((dx / length, dy / length))
            if senses.food is not None:
                self._toward_food.append(_cosine(dx, dy, senses.food.x - creature.x, senses.food.y - creature.y))
        other = senses.other
        if other is not None and other.size >= THREAT_RATIO * creature.size:
            away = moving and _cosine(param[0], param[1], other.x - creature.x, other.y - creature.y) < 0
            self._fled.append(1.0 if away else 0.0)

    def row(self, world: World, exam_ratio: Optional[float], round_number: Optional[int]) -> Dict[str, Value]:
        steps = world.creature_steps - self._steps_seen
        bites = world.total_bites - self._bites_seen
        if self._unit_moves:
            mean_x = sum(u[0] for u in self._unit_moves) / len(self._unit_moves)
            mean_y = sum(u[1] for u in self._unit_moves) / len(self._unit_moves)
            concentration: Optional[float] = math.hypot(mean_x, mean_y)
        else:
            concentration = None
        deaths = {cause: world.deaths[cause] - seen for cause, seen in self._deaths_seen.items()}
        row: Dict[str, Value] = {
            "step": world.step_count,
            "level": world.level.number,
            "population": len(world.creatures),
            "round": round_number,
            "bites_per_creature_step": bites / steps if steps > 0 else None,
            "toward_food": _mean(self._toward_food),
            "flee": _mean(self._fled),
            "direction_concentration": concentration,
            "exam_ratio": exam_ratio,
            "deaths_hunger": deaths["hunger"],
            "deaths_attack": deaths["attack"],
        }
        self._clear_samples()
        self._bites_seen = world.total_bites
        self._steps_seen = world.creature_steps
        self._deaths_seen = {cause: world.deaths[cause] for cause in self._deaths_seen}
        return row
