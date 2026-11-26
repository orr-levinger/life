"""The scoreboard: numbers that say how life is going and whether the creatures are getting smarter, one row per window
of steps."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from .sphere import cross, bearing
from .senses import Senses

if TYPE_CHECKING:
    from .creature import Creature
    from .exam import ExamResult
    from .policies import Action
    from .world import World

ROW_KEYS = (
    "step", "level", "population", "energy", "births", "newcomers", "generation", "bites_per_creature_step",
    "toward_food", "flee", "direction_concentration", "efficiency", "intelligence", "wanderers", "deaths_hunger",
    "deaths_attack", "deaths_age", "max_food", "food_energy", "food_per_step", "move_cost", "step_cost", "lifespan",
    "hunters", "manual", "chase", "attacks", "vision_width", "vision_reach", "hearing", "fear_of_death",
    "love_of_babies", "learns_from_others", "neurons", "layers", "fruit_rate",
    "sprint_power", "eat_miss", "attack_cost", "attack_damage", "bump_cost", "vision_cost", "hearing_cost", "neuron_cost",
    "stealth_cost", "armour_cost",
)
BEFORE_COSTS = ROW_KEYS[:-10]  # the columns of scoreboards written before every cost was a knob (2026-11-26)
BEFORE_FRUIT = BEFORE_COSTS[:-1]  # the columns of scoreboards written before trees dropped fruit (2026-11-25)
BEFORE_BRAIN = BEFORE_FRUIT[:-2]  # the columns of scoreboards written before the brain's size was a gene (2026-11-24)
BEFORE_VALUES = BEFORE_BRAIN[:-3]  # the columns of scoreboards written before values were genes (2026-11-24)
BEFORE_SENSES = BEFORE_VALUES[:-3]  # the columns of scoreboards written before senses were genes (2026-11-24)
BEFORE_HUNTING = BEFORE_SENSES[:-2]  # the columns of scoreboards written before the chase and attacks columns
OLDER_LAYOUTS = (BEFORE_COSTS, BEFORE_FRUIT, BEFORE_BRAIN, BEFORE_VALUES, BEFORE_SENSES, BEFORE_HUNTING)  # read with the columns they lack empty
DEATH_CAUSES = ("hunger", "attack", "age")
Value = Union[int, float, None]


def read_rows(path: Path) -> List[Dict[str, str]]:
    """A run's scoreboard CSV as rows of text; an empty cell is a score with nothing to measure."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def migrate_scoreboard(path: Path) -> bool:
    """Gives a scoreboard written before the chase and attacks, the senses or the values columns the columns it
    lacks, empty. Returns whether it changed it. A scoreboard of any other columns is refused."""
    with open(path, newline="") as f:
        header = tuple(next(csv.reader(f), ()))
    if header == ROW_KEYS:
        return False
    if header not in OLDER_LAYOUTS:
        raise ValueError(f"{path} has columns this code does not know how to bring up to date")
    rows = read_rows(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_KEYS)
        writer.writeheader()
        writer.writerows({**{key: "" for key in ROW_KEYS}, **row} for row in rows)
    temporary.replace(path)
    return True


def numbers(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row[key] != ""]


def last_steps(rows: List[Dict[str, str]], steps: int) -> List[Dict[str, str]]:
    """The rows of a run's last `steps` steps, however often its rows were written."""
    end = int(float(rows[-1]["step"])) if rows else 0
    return [row for row in rows if int(float(row["step"])) > end - steps]


def _cosine(ax: float, ay: float, bx: float, by: float) -> float:
    lengths = math.hypot(ax, ay) * math.hypot(bx, by)
    return (ax * bx + ay * by) / lengths if lengths > 0 else 0.0  # one on the same spot: no direction


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


class Scoreboard:
    """Scores, births, newcomers and deaths cover the window since the last row; `generation` is the deepest alive;
    `level` and the settings are the dial's, as in force during the window (ranges are shown by their middle);
    `energy` is the average, over the window, of each creature's energy as a share of its own maximum;
    `efficiency` is the living world's energy earned ÷ energy spent; `intelligence` and `wanderers` are the same
    measure from the exam in the fixed test world, in the rows that have an exam; `manual` is 1 while the owner holds
    the knobs, 0 while Auto does; `chase` is the share of moves toward a smaller stranger in sight, `attacks` the
    attacks tried in the window."""

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
        self._chased: List[float] = []
        self._attacks = 0

    def observe(self, creature: Creature, senses: Senses, action: Action) -> None:
        kind, param = action
        self._energy_shares.append(creature.energy / creature.max_energy)
        moving = kind == "MOVE" and math.hypot(*param) > 0
        if moving:
            dx, dy = param
            way = dx * np.asarray(creature.facing) + dy * cross(creature.p, creature.facing)
            self._unit_moves.append(way / np.linalg.norm(way))  # the way it went, on the planet
            if senses.food is not None:
                ahead, left = bearing(creature.p, creature.facing, senses.food.p)
                if float(ahead) or float(left):  # food on its own spot has no direction
                    self._toward_food.append(_cosine(dx, dy, float(ahead), float(left)))
        other = senses.other  # any bigger one is a threat, any smaller one prey
        if other is not None and other.size > creature.size:
            away = moving and _cosine(param[0], param[1], *map(float, bearing(creature.p, creature.facing, other.p))) < 0
            self._fled.append(1.0 if away else 0.0)
        if other is not None and other.size < creature.size:
            toward = moving and _cosine(param[0], param[1], *map(float, bearing(creature.p, creature.facing, other.p))) > 0
            self._chased.append(1.0 if toward else 0.0)
        if kind == "ATTACK":
            self._attacks += 1

    def row(self, world: World, exam: Optional[ExamResult], manual: bool) -> Dict[str, Value]:
        now = {"bites": world.total_bites, "steps": world.creature_steps, "births": world.births,
               "newcomers": world.newcomers, "earned": world.energy_earned, "spent": world.energy_spent}
        window = {key: now[key] - self._seen[key] for key in now}
        if self._unit_moves:  # how much the ways moved agree on the planet: 1 all one way, 0 every way
            concentration: Optional[float] = float(np.linalg.norm(np.mean(self._unit_moves, axis=0)))
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
            "chase": _mean(self._chased),
            "attacks": self._attacks,
            "vision_width": _mean([c.vision_width for c in world.creatures]),
            "vision_reach": _mean([c.vision_reach for c in world.creatures]),
            "hearing": _mean([c.hearing for c in world.creatures]),
            "fear_of_death": _mean([c.fear_of_death for c in world.creatures]),
            "love_of_babies": _mean([c.love_of_babies for c in world.creatures]),
            "learns_from_others": _mean([c.learns_from_others for c in world.creatures]),
            "neurons": _mean([c.neurons for c in world.creatures if c.layers]),  # learners: the ones whose brains grow
            "layers": _mean([c.layers for c in world.creatures if c.layers]),
            "fruit_rate": settings.fruit_rate,
            "sprint_power": settings.sprint_power,
            "eat_miss": settings.eat_miss,
            "attack_cost": settings.attack_cost,
            "attack_damage": settings.attack_damage,
            "bump_cost": settings.bump_cost,
            "vision_cost": settings.vision_cost,
            "hearing_cost": settings.hearing_cost,
            "neuron_cost": settings.neuron_cost,
            "stealth_cost": settings.stealth_cost,
            "armour_cost": settings.armour_cost,
        }
        self._clear_samples()
        self._seen = now
        self._deaths_seen = {cause: world.deaths[cause] for cause in DEATH_CAUSES}
        return row
