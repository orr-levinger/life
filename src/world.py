"""The world: food, creatures, and one step of time."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, List, Optional, Set

import numpy as np

from .creature import Creature
from .food import Food
from .levels import MAX_AGE, Level
from .policies import Action, decide
from .senses import Senses, display_intent, sense

Observer = Callable[[Creature, Senses, Action], None]


class World:
    MIN_SPAWN_DISTANCE = 0.5  # gap kept between new food and anything already there
    FOOD_RADIUS_MARGIN = 0.4  # a new food's largest radius (energy 40 × 0.01): keeps candidates inside the world
    MAX_CREATURES = 40

    def __init__(self, width: int, height: int, level: Level, rng: random.Random, np_rng: np.random.Generator,
                 food_rng: random.Random, evolving: bool = True) -> None:
        """`evolving=False` is an exam world: no splits, no deaths, no aging, food placed without looking at creatures."""
        self.width = width
        self.height = height
        self.level = level
        self.rng = rng  # decisions and size mutation
        self.np_rng = np_rng  # brain mutation
        self.food_rng = food_rng  # food placement, separate so two exam copies get the same food
        self.evolving = evolving
        self.creatures: List[Creature] = []
        self.foods: List[Food] = []
        self.step_count = 0
        self.foods_created_this_step: Set[int] = set()
        self.deaths: Counter = Counter()
        self.total_bites = 0
        self.creature_steps = 0
        self.dead_this_step: List[Creature] = []
        self._next_id = 1

    def next_id(self) -> int:
        new_id = self._next_id
        self._next_id += 1
        return new_id

    def add_creature(self, creature: Creature) -> None:
        self.creatures.append(creature)

    def set_level(self, level: Level) -> None:
        self.level = level

    def kill(self, creature: Creature, cause: str) -> None:
        self.creatures.remove(creature)
        self.deaths[cause] += 1
        self.dead_this_step.append(creature)

    def spawn_food(self) -> None:
        """The same number of candidates every step, so two worlds with the same food seed draw the same food."""
        margin = self.FOOD_RADIUS_MARGIN
        for _ in range(self.level.food_per_step):
            x = self.food_rng.uniform(margin, self.width - margin)
            y = self.food_rng.uniform(margin, self.height - margin)
            energy = self.food_rng.uniform(*self.level.food_energy)
            if len(self.foods) >= self.level.max_food:
                continue
            food = Food(x, y, remaining_duration=-1, energy=energy)
            if self._fits(food):
                self.foods.append(food)
                self.foods_created_this_step.add(id(food))

    def _fits(self, food: Food) -> bool:
        neighbours = list(self.foods) + (list(self.creatures) if self.evolving else [])
        return all(
            math.hypot(food.x - other.x, food.y - other.y) >= food.radius + other.radius + self.MIN_SPAWN_DISTANCE
            for other in neighbours
        )

    def step(self, observer: Optional[Observer] = None) -> None:
        self.step_count += 1
        self.foods_created_this_step.clear()
        self.dead_this_step = []
        self.spawn_food()

        decisions = []
        for creature in self.creatures:
            senses = sense(creature, self)
            action = decide(creature, self, senses, self.rng)
            creature.intent = display_intent(creature, action, senses)
            creature.intended_vector = action[1] if action[0] == "MOVE" else (0.0, 0.0)
            if observer is not None:
                observer(creature, senses, action)
            decisions.append((creature, action))
        self.creature_steps += len(decisions)

        for phase in ("ATTACK", "EAT", "MOVE"):
            for creature, action in decisions:
                if action[0] == phase and creature in self.creatures:
                    creature.apply_action(action, self)

        if self.evolving:
            self._deaths_and_births()
        else:
            for creature in self.creatures:
                creature.energy = max(creature.energy, 0.0)

        self._decay_food()

    def _deaths_and_births(self) -> None:
        for creature in list(self.creatures):
            creature.age += 1
            if creature.energy <= 0:
                self.kill(creature, "hunger")
            elif creature.age >= MAX_AGE:
                self.kill(creature, "age")
        for creature in list(self.creatures):
            if creature.should_split() and len(self.creatures) + 2 <= self.MAX_CREATURES:
                self.creatures.remove(creature)
                self.creatures.extend(creature.split(self))

    def _decay_food(self) -> None:
        kept = []
        for food in self.foods:
            if id(food) not in self.foods_created_this_step:
                food.decay()
            if not food.is_expired() and food.energy > 0:
                kept.append(food)
        self.foods = kept
