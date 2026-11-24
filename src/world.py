"""The living world: food, creatures, and one step of time. Nothing here ranks creatures or picks parents:
a creature does what its brain chooses, has a baby when it has the energy, and dies only with low energy: when its
energy runs out (or a hunter takes it), or when it is past its lifespan and poorly fed."""
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
from .senses import Senses, display_intent, perceived, sense_all
from .settings import (BABY_SHARE, BIRTH_AT, FEAR_LIMITS, GENE_MUTATION, HEARING_LIMITS, LEARN_LIMITS, LOVE_LIMITS,
                       LOW_ENERGY, LR_MUTATION, MAX_CREATURES, MUTATION_RATE, MUTATION_SCALE, SIZE_MUTATION, SOUP_MIN,
                       START_ENERGY, VALUE_START, VISION_REACH_LIMITS, VISION_WIDTH_LIMITS, Settings)

if TYPE_CHECKING:  # the world never imports TensorFlow: a learner is handed to it
    from .learner import Learner

Observer = Callable[[Creature, Senses, Action], None]
INHERIT = ("learned", "inborn")


class World:
    MIN_SPAWN_DISTANCE = 0.5  # gap kept between new food and anything already there
    FOOD_RADIUS_MARGIN = 0.4  # a new food's largest radius (energy 40 × 0.01): keeps candidates inside the world
    PREFILL_STEPS = 100  # food spawn passes before a fresh world starts, so it doesn't begin empty

    def __init__(self, width: int, height: int, settings: Settings, species: str, rng: random.Random,
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
        self.settings = settings  # how hard survival is right now; the dial changes it
        self.level = 0  # how many of the dial's nudges are in force
        self.species = species
        self.rng = rng  # decisions, placement and size mutation
        self.np_rng = np_rng  # brains and their mutation
        self.food_rng = food_rng  # food placement, separate so two exam copies get the same food
        self.evolving = evolving
        self.learner = learner
        self.inherit = inherit
        self._had_baby = set()  # id() of each learner that had a baby last step: its love of babies joins this step's reward
        self.creatures: List[Creature] = []
        self.foods: List[Food] = []
        self.step_count = 0
        self.foods_created_this_step: Set[int] = set()
        self.deaths: Counter = Counter()
        self.births = 0
        self.newcomers = 0
        self.total_bites = 0
        self.energy_earned = 0.0  # by eating and hunting
        self.energy_spent = 0.0  # on moving, living, eating at nothing and attacking
        self.creature_steps = 0
        self.dead_this_step: List[Creature] = []
        self._next_id = 1

    def next_id(self) -> int:
        new_id = self._next_id
        self._next_id += 1
        return new_id

    def add_creature(self, creature: Creature) -> None:
        self.creatures.append(creature)

    def set_settings(self, settings: Settings, level: int) -> None:
        self.settings = settings
        self.level = level

    def kill(self, creature: Creature, cause: str) -> None:
        self.creatures.remove(creature)
        self.deaths[cause] += 1
        self.dead_this_step.append(creature)

    def add_newcomer(self) -> Creature:
        """A creature of this world's species with a random brain, at a random spot: the first creatures, and the
        primordial soup."""
        x, y = self.rng.uniform(1.0, self.width - 1.0), self.rng.uniform(1.0, self.height - 1.0)
        size = self.rng.uniform(Creature.MIN_SIZE, Creature.MAX_SIZE)
        brain = Brain.random(self.np_rng) if self.species == "brain" else None
        slot = self.learner.newcomer(self.np_rng) if self.species == "learner" else None
        newcomer = Creature(self.next_id(), x, y, size, self.rng.uniform(*START_ENERGY), self.species, brain, slot=slot)
        newcomer.lifespan = round(self.rng.uniform(*self.settings.lifespan))
        newcomer.vision_width = self.rng.uniform(*VISION_WIDTH_LIMITS)
        newcomer.vision_reach = self.rng.uniform(*VISION_REACH_LIMITS)
        newcomer.hearing = self.rng.uniform(*HEARING_LIMITS)
        newcomer.fear_of_death, newcomer.love_of_babies, newcomer.learns_from_others = VALUE_START
        self.creatures.append(newcomer)
        self.newcomers += 1
        return newcomer

    def spawn_food(self) -> None:
        """The same number of candidates every step, so two worlds with the same food seed draw the same food."""
        margin = self.FOOD_RADIUS_MARGIN
        whole = int(self.settings.food_per_step)  # e.g. 2.4 per step: 2 every step and a third 40 % of the time
        extra = self.food_rng.random() < self.settings.food_per_step - whole
        for _ in range(whole + extra):
            x = self.food_rng.uniform(margin, self.width - margin)
            y = self.food_rng.uniform(margin, self.height - margin)
            energy = self.food_rng.uniform(*self.settings.food_energy)
            if len(self.foods) >= int(self.settings.max_food):
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
        for creature in acting:
            creature.hurt = None  # pain is felt for one step: sensed just now, gone unless hit again
        actions = decide_all(self, acting, senses)
        for creature, sensed, action in zip(acting, senses, actions):
            creature.intent = display_intent(creature, action, sensed)
            creature.intended_vector = action[1] if action[0] == "MOVE" else (0.0, 0.0)
            if observer is not None:
                observer(creature, sensed, action)
        self.creature_steps += len(acting)
        energy_before = [creature.energy for creature in acting]

        for phase in ("ATTACK", "EAT", "MOVE"):
            for creature, action in zip(acting, actions):
                if action[0] == phase and creature in self.creatures:
                    creature.apply_action(action, self)

        for creature in list(self.creatures):
            if self.evolving:
                creature.age += 1
            if not self.evolving:
                creature.energy = max(creature.energy, 0.0)  # nobody dies in an exam
            elif creature.energy <= 0:
                self.kill(creature, "hunger")
            elif (creature.lifespan is not None and creature.age >= creature.lifespan
                  and creature.energy < LOW_ENERGY * creature.max_energy):
                self.kill(creature, "age")  # past its lifespan, only a creature with low energy dies

        if self.learner is not None and self.evolving:
            self._teach(acting, energy_before)  # before births: the energy given to a baby is never a penalty
            for dead in self.dead_this_step:
                if dead.policy == "learner":
                    self.learner.release(dead.slot)
        if self.evolving:
            self._births()
            if len(self.creatures) < SOUP_MIN:
                self.add_newcomer()
        self._decay_food()

    def _teach(self, acting: List[Creature], energy_before: List[float]) -> None:
        """Each learner creature's reward for this step is its change in energy, valued by its own genes: plus its love
        of babies for a baby it had the step before, minus its fear of death if it died (which ends its experience).
        Each learner's step is also a lesson for every other living learner that sees or hears it."""
        alive = set(map(id, self.creatures))
        events = {}
        for creature, before in zip(acting, energy_before):
            if creature.policy == "learner":
                died, baby = id(creature) not in alive, id(creature) in self._had_baby
                events[id(creature)] = (creature.energy - before, died, baby)
                self.learner.record(creature.slot, self._valued(creature, *events[id(creature)]), died)
        self._had_baby.clear()
        self._show(acting, events)
        self.learner.update()

    @staticmethod
    def _valued(valuer: Creature, change: float, died: bool, baby: bool) -> float:
        """A step's reward as `valuer`'s genes value it: the change in energy, minus its fear of death if the one who
        lived it died, plus its love of babies if that one had a baby."""
        return change - valuer.fear_of_death * died + valuer.love_of_babies * baby

    def _show(self, acting: List[Creature], events: dict) -> None:
        """Every living learner that sees or hears a learner that acted takes its step as a lesson, valued by the
        watcher's own genes and weighed by its own learning from others."""
        actors = [c for c in acting if c.policy == "learner"]
        watchers = [c for c in self.creatures if c.policy == "learner"]
        if not actors or not watchers:
            return
        seen = perceived(watchers, actors)
        for i, j in zip(*np.nonzero(seen)):
            watcher, actor = watchers[j], actors[i]
            if watcher is not actor:
                reward = self._valued(watcher, *events[id(actor)])
                self.learner.watch(watcher.slot, actor.slot, reward=reward, weight=watcher.learns_from_others)

    def _births(self) -> None:
        """A creature with enough energy has a baby; the safety limit only pauses births."""
        for parent in list(self.creatures):
            if len(self.creatures) >= MAX_CREATURES:
                return
            if parent.energy >= BIRTH_AT * parent.max_energy:
                self._birth(parent)

    def _birth(self, parent: Creature) -> None:
        """The baby gets a share of the parent's energy and the parent's brain with mutations (a learner's brain as
        the world's `inherit` says: as it is now, or as the parent was born)."""
        slot = None
        if parent.policy == "learner":
            if not self.learner.has_room():
                return
            lr_factor = math.exp(self.rng.gauss(0.0, LR_MUTATION))
            slot = self.learner.baby(parent.slot, self.np_rng, lr_factor, self.inherit)
        gift = BABY_SHARE * parent.energy
        size = parent.size * (1.0 + self.rng.uniform(-SIZE_MUTATION, SIZE_MUTATION))
        size = min(max(size, Creature.MIN_SIZE), Creature.MAX_SIZE)
        x = min(max(parent.x + self.rng.uniform(-1.0, 1.0), 0.0), self.width)
        y = min(max(parent.y + self.rng.uniform(-1.0, 1.0), 0.0), self.height)
        brain = parent.brain.mutate(self.np_rng, MUTATION_RATE, MUTATION_SCALE) if parent.brain is not None else None
        parent.energy -= gift
        baby = Creature(self.next_id(), x, y, size, gift, parent.policy, brain,
                        parent_id=parent.id, generation=parent.generation + 1, slot=slot)
        baby.lifespan = round(self.rng.uniform(*self.settings.lifespan))
        baby.vision_width = self._inherited(parent.vision_width, VISION_WIDTH_LIMITS)
        baby.vision_reach = self._inherited(parent.vision_reach, VISION_REACH_LIMITS)
        baby.hearing = self._inherited(parent.hearing, HEARING_LIMITS)
        baby.fear_of_death = self._inherited(parent.fear_of_death, FEAR_LIMITS)
        baby.love_of_babies = self._inherited(parent.love_of_babies, LOVE_LIMITS)
        baby.learns_from_others = self._inherited(parent.learns_from_others, LEARN_LIMITS)
        self.creatures.append(baby)
        self.births += 1
        if parent.policy == "learner":
            self._had_baby.add(id(parent))

    def _inherited(self, gene: float, limits: tuple) -> float:
        """A parent's sense or value gene for its baby: × (1 ± up to GENE_MUTATION), kept in its limits."""
        return min(max(gene * (1.0 + self.rng.uniform(-GENE_MUTATION, GENE_MUTATION)), limits[0]), limits[1])

    def _decay_food(self) -> None:
        kept = []
        for food in self.foods:
            if id(food) not in self.foods_created_this_step:
                food.decay()
            if not food.is_expired() and food.energy > 0:
                kept.append(food)
        self.foods = kept
