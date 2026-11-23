"""The living world: food, creatures, and one step of time. Nothing here ranks creatures or picks parents:
a creature eats what it touches, has a baby when it has the energy, and dies of hunger, a hunter or old age."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING, Callable, List, Optional, Set

import numpy as np

from .brain import Brain
from .creature import Creature
from .food import Food
from .policies import Action, decide_all
from .senses import Senses, display_intent, sense_all
from .stages import (BABY_SHARE, BIRTH_AT, MAX_AGE, MAX_CREATURES, MUTATION_RATE, MUTATION_SCALE, SIZE_MUTATION,
                     SOUP_MIN, START_ENERGY, Stage)

if TYPE_CHECKING:  # the world never imports TensorFlow: a learner is handed to it
    from .learner import Learner

Observer = Callable[[Creature, Senses, Action], None]
INHERIT = ("learned", "inborn")


class World:
    MIN_SPAWN_DISTANCE = 0.5  # gap kept between new food and anything already there
    FOOD_RADIUS_MARGIN = 0.4  # a new food's largest radius (energy 40 × 0.01): keeps candidates inside the world
    PREFILL_STEPS = 100  # food spawn passes before a fresh world starts, so it doesn't begin empty

    def __init__(self, width: int, height: int, stage: Stage, species: str, rng: random.Random,
                 np_rng: np.random.Generator, food_rng: random.Random, evolving: bool = True,
                 learner: Optional[Learner] = None, inherit: str = "learned") -> None:
        """`species` is what this world's newcomers are. `evolving=False` is an exam world: nobody is born, dies,
        ages or learns, and food is placed without looking at creatures. `inherit` says what a learning baby gets:
        the parent's brain as it is now ("learned") or as the parent was born ("inborn")."""
        if species not in Creature.POLICIES:
            raise ValueError(f"species must be one of {Creature.POLICIES}, got {species!r}")
        if inherit not in INHERIT:
            raise ValueError(f"inherit must be one of {INHERIT}, got {inherit!r}")
        self.width = width
        self.height = height
        self.stage = stage
        self.species = species
        self.rng = rng  # decisions, placement and size mutation
        self.np_rng = np_rng  # brains and their mutation
        self.food_rng = food_rng  # food placement, separate so two exam copies get the same food
        self.evolving = evolving
        self.learner = learner
        self.inherit = inherit
        self.creatures: List[Creature] = []
        self.foods: List[Food] = []
        self.step_count = 0
        self.foods_created_this_step: Set[int] = set()
        self.deaths: Counter = Counter()
        self.births = 0
        self.newcomers = 0
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

    def set_stage(self, stage: Stage) -> None:
        self.stage = stage

    def kill(self, creature: Creature, cause: str) -> None:
        self.creatures.remove(creature)
        self.deaths[cause] += 1
        self.dead_this_step.append(creature)

    def add_newcomer(self) -> Creature:
        """A creature of this world's species with a random brain, at a random spot: the first creatures, and the
        primordial soup."""
        if self.species == "learner":
            raise NotImplementedError("learner newcomers arrive with the plan's Task 2")
        x, y = self.rng.uniform(1.0, self.width - 1.0), self.rng.uniform(1.0, self.height - 1.0)
        size = self.rng.uniform(Creature.MIN_SIZE, Creature.MAX_SIZE)
        brain = Brain.random(self.np_rng) if self.species == "brain" else None
        newcomer = Creature(self.next_id(), x, y, size, self.rng.uniform(*START_ENERGY), self.species, brain)
        self.creatures.append(newcomer)
        self.newcomers += 1
        return newcomer

    def spawn_food(self) -> None:
        """The same number of candidates every step, so two worlds with the same food seed draw the same food."""
        margin = self.FOOD_RADIUS_MARGIN
        for _ in range(self.stage.food_per_step):
            x = self.food_rng.uniform(margin, self.width - margin)
            y = self.food_rng.uniform(margin, self.height - margin)
            energy = self.food_rng.uniform(*self.stage.food_energy)
            if len(self.foods) >= self.stage.max_food:
                continue
            food = Food(x, y, remaining_duration=-1, energy=energy)
            if self._fits(food):
                self.foods.append(food)
                self.foods_created_this_step.add(id(food))

    def prefill_food(self) -> None:
        for _ in range(self.PREFILL_STEPS):
            self.spawn_food()

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

        acting = list(self.creatures)
        senses = sense_all(self, acting)
        actions = decide_all(self, acting, senses)
        for creature, sensed, action in zip(acting, senses, actions):
            creature.intent = display_intent(creature, action, sensed)
            creature.intended_vector = action[1] if action[0] == "MOVE" else (0.0, 0.0)
            if observer is not None:
                observer(creature, sensed, action)
        self.creature_steps += len(acting)
        energy_before = [creature.energy for creature in acting]

        for phase in ("ATTACK", "MOVE"):
            for creature, action in zip(acting, actions):
                if action[0] == phase and creature in self.creatures:
                    creature.apply_action(action, self)
        self._eat()

        mortal = self.evolving and self.stage.mortal
        for creature in list(self.creatures):
            if self.evolving:
                creature.age += 1
            if mortal and creature.energy <= 0:
                self.kill(creature, "hunger")
            elif mortal and creature.age >= MAX_AGE:
                self.kill(creature, "age")
            elif creature.energy < 0:
                creature.energy = 0.0  # where nothing can die, an empty creature simply stays empty

        if self.learner is not None and self.evolving:
            self._teach(acting, energy_before)
        if self.evolving:
            self._births()
            if self.stage.mortal and len(self.creatures) < SOUP_MIN:
                self.add_newcomer()
        self._decay_food()

    def _eat(self) -> None:
        """Eating on contact: each creature takes one bite of the nearest food it can reach. All distances are
        computed at once; a food eaten up by an earlier creature this step is skipped by the next."""
        if not self.foods or not self.creatures:
            return
        foods, eaters = list(self.foods), list(self.creatures)
        distance = np.hypot(np.array([f.x for f in foods])[None, :] - np.array([c.x for c in eaters])[:, None],
                            np.array([f.y for f in foods])[None, :] - np.array([c.y for c in eaters])[:, None])
        reachable = distance <= Creature.eat_reach(np.array([c.radius for c in eaters])[:, None],
                                                   np.array([f.radius for f in foods])[None, :])
        for i, creature in enumerate(eaters):
            in_reach = np.flatnonzero(reachable[i])
            for j in in_reach[np.argsort(distance[i, in_reach], kind="stable")]:
                if foods[j].energy > 0:
                    creature.bite(foods[j], self)
                    break

    def _teach(self, acting: List[Creature], energy_before: List[float]) -> None:
        """Each learner creature's reward for this step is its change in energy; dying ends its experience."""
        alive = set(map(id, self.creatures))
        for creature, before in zip(acting, energy_before):
            if creature.policy == "learner":
                self.learner.record(creature.slot, creature.energy - before, id(creature) not in alive)
        self.learner.update()

    def _births(self) -> None:
        """A creature with enough energy has a baby; the safety limit only pauses births."""
        for parent in list(self.creatures):
            if len(self.creatures) >= MAX_CREATURES:
                return
            if parent.energy >= BIRTH_AT * parent.max_energy:
                self._birth(parent)

    def _birth(self, parent: Creature) -> None:
        if parent.policy == "learner":
            raise NotImplementedError("learner births arrive with the plan's Task 2")
        gift = BABY_SHARE * parent.energy
        size = parent.size * (1.0 + self.rng.uniform(-SIZE_MUTATION, SIZE_MUTATION))
        size = min(max(size, Creature.MIN_SIZE), Creature.MAX_SIZE)
        x = min(max(parent.x + self.rng.uniform(-1.0, 1.0), 0.0), self.width)
        y = min(max(parent.y + self.rng.uniform(-1.0, 1.0), 0.0), self.height)
        brain = parent.brain.mutate(self.np_rng, MUTATION_RATE, MUTATION_SCALE) if parent.brain is not None else None
        parent.energy -= gift
        self.creatures.append(Creature(self.next_id(), x, y, size, gift, parent.policy, brain,
                                       parent_id=parent.id, generation=parent.generation + 1))
        self.births += 1

    def _decay_food(self) -> None:
        kept = []
        for food in self.foods:
            if id(food) not in self.foods_created_this_step:
                food.decay()
            if not food.is_expired() and food.energy > 0:
                kept.append(food)
        self.foods = kept
