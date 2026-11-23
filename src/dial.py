"""The difficulty dial: the hardness of survival drives intelligence, and it is turned delicately.

Auto looks at the world every window of steps. Many creatures, well fed: survival is easy, so one knob is nudged 5 %
harder, one knob at a time. Few creatures, or hungry ones, or the primordial soup had to add newcomers: the last harder
nudge is taken back, and with none to take back one knob is nudged 5 % easier. Anything else: nothing changes.
In Manual the owner turns the knobs (`Dial.set`); Auto goes on from wherever they are.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional

from .settings import START, Settings

NUDGE = 0.05  # how much one nudge changes one knob
HUNTERS_AT = 24  # at this difficulty the next harder nudge switches hunters on
MANY = 60  # this many creatures or more, and well fed: survival is easy
WELL_FED = 0.5  # average energy, as a share of full
FEW = 30  # fewer creatures than this: survival is too hard
HUNGRY = 0.35  # average energy below this share of full: survival is too hard
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


def _at(knob: Knob, settings: Settings, end: float) -> bool:
    return abs(value(settings, knob.name) - end) <= TOLERANCE * max(1.0, abs(end))


class Dial:
    def __init__(self, settings: Settings = START, max_level: Optional[int] = None) -> None:
        self._settings = settings
        self._taken: List[Settings] = []  # the settings before each harder nudge Auto can still take back
        self._next_harder = 0  # whose turn it is, among the knobs
        self._next_easier = 0
        self.max_level = max_level

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
                "next_harder": self._next_harder, "next_easier": self._next_easier, "max_level": self.max_level}

    @staticmethod
    def from_state(state: Dict) -> "Dial":
        dial = Dial(settings_from_state(state["settings"]), state["max_level"])
        dial._taken = [settings_from_state(s) for s in state["taken"]]
        dial._next_harder, dial._next_easier = state["next_harder"], state["next_easier"]
        return dial

    def set(self, settings: Settings) -> None:
        """The owner's settings. Auto goes on from here; the nudges it made before are no longer its to take back."""
        self._settings = settings
        self._taken = []

    def review(self, alive: int, energy: Optional[float], newcomers: int) -> str:
        """Look at one window: `alive` at its end, `energy` the creatures' average share of full energy over it (None
        when nobody was measured), `newcomers` the primordial soup had to add. Returns "harder", "easier" or "hold"."""
        if newcomers > 0 or alive < FEW or (energy is not None and energy < HUNGRY):
            return self._easier()
        if alive >= MANY and energy is not None and energy >= WELL_FED:
            return self._harder()
        return "hold"

    def turn_to(self, level: float) -> None:
        """Nudge harder until the difficulty is `level` (for a world that begins harder than the easy start)."""
        while self.level < level - TOLERANCE and self._harder() == "harder":
            pass

    def _harder(self) -> str:
        if self.max_level is not None and self.level >= self.max_level - TOLERANCE:
            return "hold"
        if not self._settings.attacks and self.level >= HUNTERS_AT - TOLERANCE:
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
            self._settings = self._taken.pop()
            return "easier"
        if self._settings.attacks:
            self._settings = replace(self._settings, attacks=False)
            return "easier"
        for _ in KNOBS:
            knob = KNOBS[self._next_easier % len(KNOBS)]
            self._next_easier += 1
            if not _at(knob, self._settings, knob.easiest):
                self._settings = with_value(self._settings, knob.name, value(self._settings, knob.name) / knob.factor)
                return "easier"
        return "hold"  # every knob is at its easiest
