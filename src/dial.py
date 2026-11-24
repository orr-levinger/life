"""The difficulty dial: the hardness of survival drives intelligence, and it is turned delicately.

Auto steers the number alive toward 64, 80 % of the room (the owner: "80 creatures means too easy, 40 too hard"). Every
look it takes one step, in nudges of 5 %, sized like gradient descent: alpha times how far the creatures are from 64,
plus alpha times how fast their number moves (and how fast their energy falls), rounded and clipped to 4. Harder nudges
turn one knob at a time; easier ones take back Auto's own newest harder nudges first.
In Manual the owner turns the knobs (`Dial.set`); Auto goes on from wherever they are.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .rules import Rules, starting_rules
from .settings import START, Settings

NUDGE = 0.05  # how much one nudge changes one knob
PER_CREATURES = 8  # one nudge per this many creatures away from the target
PER_ALIVE_RATE = 0.10  # one nudge per 10 % a 1,000 steps the number alive moves
PER_ENERGY_RATE = 0.05  # one nudge per 5 points of a full tank a 1,000 steps the average energy falls
MAX_NUDGES = 4  # the most one look may move, either way
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Knob:
    name: str  # a field of Settings; a range is turned by its middle
    harder: int  # -1: less is harder (food, lifespan); +1: more is harder (costs)
    easiest: float  # the ends of the knob's slider: it never passes them
    hardest: float

    @property
    def factor(self) -> float:
        """What one harder nudge multiplies the knob by (an easier nudge divides by it)."""
        return 1.0 + NUDGE * self.harder


KNOBS = (
    Knob("max_food", -1, easiest=400.0, hardest=20.0),
    Knob("food_energy", -1, easiest=60.0, hardest=3.0),
    Knob("move_cost", +1, easiest=0.1, hardest=3.0),
    Knob("step_cost", +1, easiest=0.01, hardest=0.5),
    Knob("lifespan", -1, easiest=10000.0, hardest=625.0),
    Knob("food_per_step", -1, easiest=6.0, hardest=0.2),
)


def knob_named(name: str) -> Knob:
    for knob in KNOBS:
        if knob.name == name:
            return knob
    raise ValueError(f"no knob called {name!r}; the knobs are {', '.join(k.name for k in KNOBS)}")


def value(settings: Settings, name: str) -> float:
    """A knob's value: the setting itself, or the middle of a range."""
    current = getattr(settings, name)
    return sum(current) / 2 if isinstance(current, tuple) else current


def _held(knob: Knob, number: float) -> float:
    low, high = sorted((knob.easiest, knob.hardest))
    return min(max(number, low), high)


def with_value(settings: Settings, name: str, number: float) -> Settings:
    """`settings` with one knob at `number`, held within the knob's ends; a range keeps its shape around its middle."""
    number = _held(knob_named(name), number)
    current = getattr(settings, name)
    if isinstance(current, tuple):
        scale = number / value(settings, name)
        return replace(settings, **{name: tuple(end * scale for end in current)})
    return replace(settings, **{name: number})


def difficulty(settings: Settings) -> float:
    """How many 5 % nudges harder than the easy start `settings` are (below zero: easier); hunters count as one."""
    nudges = sum(math.log(value(settings, k.name) / value(START, k.name)) / math.log(k.factor) for k in KNOBS)
    return nudges + (settings.attacks - START.attacks)


def settings_state(settings: Settings) -> Dict:
    return asdict(settings)


def settings_from_state(state: Dict) -> Settings:
    return Settings(**{name: tuple(given) if isinstance(given, list) else given for name, given in state.items()})


def slope(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """The slope of the straight line closest to (step, value) points (least squares), per 1,000 steps."""
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    return 1000.0 * sum((x - mean_x) * (y - mean_y) for x, y in points) / spread if spread else None


def rates(rows: Sequence[Mapping[str, float]]) -> Tuple[Optional[float], Optional[float]]:
    """From scoreboard rows: how fast the number alive moves (as a share of it) and how fast the average energy moves
    (in shares of a full tank), each per 1,000 steps."""
    alive = slope([(row["step"], row["population"]) for row in rows])
    mean_alive = sum(row["population"] for row in rows) / len(rows) if rows else 0
    energy = slope([(row["step"], row["energy"]) for row in rows if row["energy"] is not None])
    return (alive / mean_alive if alive is not None and mean_alive else None), energy


def step_for(alive: int, alive_rate: Optional[float], energy_rate: Optional[float], newcomers: int, target: float) -> int:
    """Auto's step, in nudges (+ harder, - easier): the distance from the target and the rates, each times its alpha.
    A rate not measured yet counts as steady; rising energy never eases."""
    if newcomers > 0:
        return -MAX_NUDGES  # the world nearly died out
    step = (alive - target) / PER_CREATURES
    step += (alive_rate or 0.0) / PER_ALIVE_RATE
    step += min(energy_rate or 0.0, 0.0) / PER_ENERGY_RATE
    return max(-MAX_NUDGES, min(MAX_NUDGES, round(step)))


def _at(knob: Knob, settings: Settings, end: float) -> bool:
    return abs(value(settings, knob.name) - end) <= TOLERANCE * max(1.0, abs(end))


class Dial:
    def __init__(self, settings: Settings = START, max_level: Optional[int] = None) -> None:
        self._settings = settings
        self._taken: List[Settings] = []  # the settings before each harder nudge Auto can still take back
        self._next_harder = 0  # whose turn it is, among the knobs
        self._next_easier = 0
        self._hunters: Optional[bool] = None  # the owner's hunters switch, in Auto: Auto leaves hunters as set
        self.max_level = max_level
        self.rules: Rules = starting_rules()  # the world's rules (its target and when hunters come): the runner hands them

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def level(self) -> float:
        """The difficulty now."""
        return difficulty(self._settings)

    def state(self) -> Dict:
        """What a save keeps: the knobs, the nudges Auto can take back, whose turn it is, and the top level."""
        return {"settings": settings_state(self._settings), "taken": [settings_state(s) for s in self._taken],
                "next_harder": self._next_harder, "next_easier": self._next_easier, "max_level": self.max_level,
                "hunters": self._hunters}

    @staticmethod
    def from_state(state: Dict) -> "Dial":
        dial = Dial(settings_from_state(state["settings"]), state["max_level"])
        dial._taken = [settings_from_state(s) for s in state["taken"]]
        dial._next_harder, dial._next_easier = state["next_harder"], state["next_easier"]
        dial._hunters = state["hunters"]
        return dial

    def set(self, settings: Settings) -> None:
        """The owner's settings. Auto goes on from here; the nudges it made before are no longer its to take back."""
        self._settings = settings
        self._taken = []
        self._hunters = None  # the owner's full settings say whether hunters are on

    def set_hunters(self, on: bool) -> None:
        """The owner's hunters switch while Auto turns the knobs: Auto never changes it afterwards."""
        self._hunters = on
        self._settings = replace(self._settings, attacks=on)

    def review(self, alive: int, alive_rate: Optional[float], energy_rate: Optional[float], newcomers: int) -> int:
        """One look: takes the step `step_for` says and returns the nudges it made (+ harder, - easier, 0 none)."""
        wanted, made = step_for(alive, alive_rate, energy_rate, newcomers, self.rules["auto_target"]), 0
        while made < wanted and self._harder() == "harder":
            made += 1
        while made > wanted and self._easier() == "easier":
            made -= 1
        return made

    def turn_to(self, level: float) -> None:
        """Nudge harder until the difficulty is `level` (for a world that begins harder than the easy start)."""
        while self.level < level - TOLERANCE and self._harder() == "harder":
            pass

    def _harder(self) -> str:
        if self.max_level is not None and self.level >= self.max_level - TOLERANCE:
            return "hold"
        if self._hunters is None and not self._settings.attacks and self.level >= self.rules["auto_hunters_at"] - TOLERANCE:
            return self._nudge(replace(self._settings, attacks=True))
        for _ in KNOBS:
            knob = KNOBS[self._next_harder % len(KNOBS)]
            self._next_harder += 1
            if not _at(knob, self._settings, knob.hardest):
                return self._nudge(with_value(self._settings, knob.name, value(self._settings, knob.name) * knob.factor))
        return "hold"  # every knob is at its hardest

    def _nudge(self, harder: Settings) -> str:
        self._taken.append(self._settings)
        self._settings = harder
        return "harder"

    def _easier(self) -> str:
        if self._taken:
            taken = self._taken.pop()
            self._settings = taken if self._hunters is None else replace(taken, attacks=self._hunters)
            return "easier"
        if self._hunters is None and self._settings.attacks:
            self._settings = replace(self._settings, attacks=False)
            return "easier"
        for _ in KNOBS:
            knob = KNOBS[self._next_easier % len(KNOBS)]
            self._next_easier += 1
            if not _at(knob, self._settings, knob.easiest):
                self._settings = with_value(self._settings, knob.name, value(self._settings, knob.name) / knob.factor)
                return "easier"
        return "hold"  # every knob is at its easiest
