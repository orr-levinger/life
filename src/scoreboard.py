"""The scoreboard: numbers that say how life is going and whether the creatures are getting smarter, one row per window
of steps."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .senses import THREAT_RATIO, Senses

if TYPE_CHECKING:
    from .creature import Creature
    from .exam import ExamResult
    from .policies import Action
    from .world import World

ROW_KEYS = (
    "step", "level", "population", "energy", "births", "newcomers", "generation", "bites_per_creature_step",
    "toward_food", "flee", "direction_concentration", "efficiency", "intelligence", "wanderers", "deaths_hunger",
    "deaths_attack", "deaths_age", "max_food", "food_energy", "food_per_step", "move_cost", "step_cost", "lifespan",
    "hunters", "manual",
)
DEATH_CAUSES = ("hunger", "attack", "age")
Value = Union[int, float, None]


def read_rows(path: Path) -> List[Dict[str, str]]:
    """A run's scoreboard CSV as rows of text; an empty cell is a score with nothing to measure."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def numbers(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row[key] != ""]


def last_steps(rows: List[Dict[str, str]], steps: int) -> List[Dict[str, str]]:
    """The rows of a run's last `steps` steps, however often its rows were written."""
    end = int(float(rows[-1]["step"])) if rows else 0
    return [row for row in rows if int(float(row["step"])) > end - steps]


def _cosine(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax * bx + ay * by) / (math.hypot(ax, ay) * math.hypot(bx, by))


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


class Scoreboard:
    """Scores, births, newcomers and deaths cover the window since the last row; `generation` is the deepest alive;
    `level` and the settings are the dial's, as in force during the window (ranges are shown by their middle);
    `energy` is the average, over the window, of each creature's energy as a share of its own maximum;
    `efficiency` is the living world's energy earned ÷ energy spent; `intelligence` and `wanderers` are the same
    measure from the exam in the fixed test world, in the rows that have an exam; `manual` is 1 while the owner holds
    the knobs, 0 while Auto does."""

    def __init__(self) -> None:
        self._clear_samples()
        self._seen = {"bites": 0, "steps": 0, "births": 0, "newcomers": 0, "earned": 0.0, "spent": 0.0}
        self._deaths_seen = {cause: 0 for cause in DEATH_CAUSES}

    def start_at(self, world: World) -> None:
        """The first window starts at the world's step now (a resumed world's counters do not start at zero)."""
        self._seen = {"bites": world.total_bites, "steps": world.creature_steps, "births": world.births,
                      "newcomers": world.newcomers, "earned": world.energy_earned, "spent": world.energy_spent}
        self._deaths_seen = {cause: world.deaths[cause] for cause in DEATH_CAUSES}

    def _clear_samples(self) -> None:
        self._toward_food: List[float] = []
        self._fled: List[float] = []
        self._unit_moves: List[tuple[float, float]] = []
        self._energy_shares: List[float] = []

    def observe(self, creature: Creature, senses: Senses, action: Action) -> None:
        kind, param = action
        self._energy_shares.append(creature.energy / creature.max_energy)
        moving = kind == "MOVE" and math.hypot(*param) > 0
        if moving:
            dx, dy = param
            length = math.hypot(dx, dy)
            self._unit_moves.append((dx / length, dy / length))
            if senses.food is not None and (senses.food.x, senses.food.y) != (creature.x, creature.y):
                self._toward_food.append(_cosine(dx, dy, senses.food.x - creature.x, senses.food.y - creature.y))
        other = senses.other
        if other is not None and other.size >= THREAT_RATIO * creature.size:
            away = moving and _cosine(param[0], param[1], other.x - creature.x, other.y - creature.y) < 0
            self._fled.append(1.0 if away else 0.0)

    def row(self, world: World, exam: Optional[ExamResult], manual: bool) -> Dict[str, Value]:
        now = {"bites": world.total_bites, "steps": world.creature_steps, "births": world.births,
               "newcomers": world.newcomers, "earned": world.energy_earned, "spent": world.energy_spent}
        window = {key: now[key] - self._seen[key] for key in now}
        if self._unit_moves:
            mean_x = sum(u[0] for u in self._unit_moves) / len(self._unit_moves)
            mean_y = sum(u[1] for u in self._unit_moves) / len(self._unit_moves)
            concentration: Optional[float] = math.hypot(mean_x, mean_y)
        else:
            concentration = None
        deaths = {cause: world.deaths[cause] - seen for cause, seen in self._deaths_seen.items()}
        settings = world.settings
        row: Dict[str, Value] = {
            "step": world.step_count,
            "level": world.level,
            "population": len(world.creatures),
            "energy": _mean(self._energy_shares),
            "births": window["births"],
            "newcomers": window["newcomers"],
            "generation": max((c.generation for c in world.creatures), default=None),
            "bites_per_creature_step": window["bites"] / window["steps"] if window["steps"] > 0 else None,
            "toward_food": _mean(self._toward_food),
            "flee": _mean(self._fled),
            "direction_concentration": concentration,
            "efficiency": window["earned"] / window["spent"] if window["spent"] > 0 else None,
            "intelligence": exam.intelligence if exam is not None else None,
            "wanderers": exam.wanderers if exam is not None else None,
            "deaths_hunger": deaths["hunger"],
            "deaths_attack": deaths["attack"],
            "deaths_age": deaths["age"],
            "max_food": settings.max_food,
            "food_energy": sum(settings.food_energy) / 2,
            "food_per_step": settings.food_per_step,
            "move_cost": settings.move_cost,
            "step_cost": settings.step_cost,
            "lifespan": sum(settings.lifespan) / 2,
            "hunters": int(settings.attacks),
            "manual": int(manual),
        }
        self._clear_samples()
        self._seen = now
        self._deaths_seen = {cause: world.deaths[cause] for cause in DEATH_CAUSES}
        return row
