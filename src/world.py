"""The living world: food, creatures, and one step of time. Nothing here ranks creatures or picks parents:
a creature does what its brain chooses, has a baby when it has the energy, and dies only with low energy: when its
energy runs out (or a hunter takes it), or when it is past its lifespan and poorly fed."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .brain import Brain
from .creature import Creature, alike, full_tank, kin_genes, similarity
from .sphere import arc, exp_map, local_frame, random_point, walk
from .food import Food
from .policies import Action, decide_all
from .senses import Senses, display_intent, perceived, sense_all
from .species import SpeciesBook
from .rules import EDITABLE_GENES, Rules, gene_limits, starting_rules
from .terrain import Terrain
from .settings import Settings

if TYPE_CHECKING:  # the world never imports TensorFlow: a learner is handed to it
    from .learner import Learner

Observer = Callable[[Creature, Senses, Action], None]
INHERIT = ("learned", "inborn")
FOOD_SPOT_TRIES = 50  # tries for a food spot by the pond edges' thickness
FREE_SPOT_TRIES = 1000  # tries to find a newcomer a spot that no obstacle covers
GENES = ("size", "vision_width", "vision_reach", "hearing", "fear", "love", "learns", "learning_speed",
         "layers", "neurons", "weights", "mutations", "mutation_size", "grass_gut", "meat_gut", "calls", "kin", "gills",
         "lungs", "stealth", "armour",
         "eye_pixels")  # a baby changes its parent's `mutations` of them


class World:
    MIN_SPAWN_DISTANCE = 0.5  # gap kept between new food and anything already there
    PREFILL_STEPS = 100  # food spawn passes before a fresh world starts, so it doesn't begin empty

    def __init__(self, radius: float, settings: Settings, species: str, rng: random.Random,
                 np_rng: np.random.Generator, food_rng: random.Random, evolving: bool = True,
                 learner: Optional[Learner] = None, inherit: str = "learned", rules: Optional[Rules] = None) -> None:
        """`species` is what this world's newcomers are. `evolving=False` is an exam world: nobody is born, dies,
        ages or learns, and food is placed without looking at creatures. `inherit` says what a learning baby gets:
        the parent's brain as it is now ("learned") or as the parent was born ("inborn"). `rules` are the world's own
        (default: the starting rules); `radius` is its planet's (its rule `planet_radius`)."""
        if species not in Creature.POLICIES:
            raise ValueError(f"species must be one of {Creature.POLICIES}, got {species!r}")
        if inherit not in INHERIT:
            raise ValueError(f"inherit must be one of {INHERIT}, got {inherit!r}")
        self.rules: Rules = starting_rules() if rules is None else rules
        if learner is not None:
            learner.rules = self.rules  # learning's numbers are the world's rules
        self.rules["planet_radius"] = float(radius)
        self.settings = settings  # how hard survival is right now; the dial changes it
        self.level = 0  # how many of the dial's nudges are in force
        self.species_book = SpeciesBook()  # the tree of species (src/species.py)
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
        self.terrain = Terrain.empty()  # its obstacles and trees (a new world's are made by the runner)
        self.step_count = 0
        self.foods_created_this_step: Set[int] = set()
        self.deaths: Counter = Counter()
        self.births = 0
        self.newcomers = 0
        self.calls: List[Tuple[Creature, int, float]] = []  # the last step's calls: caller, kind, volume
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

    @property
    def radius(self) -> float:
        """Its planet's radius: distances along the ground are angles × this."""
        return self.rules["planet_radius"]

    def cost(self, name: str) -> float:
        """What `name` costs in this world now: its rule (Auto and the owner turn it between 0 and nature)."""
        return self.rules[name]

    def change_rules(self, changes: Rules) -> None:
        """The owner's rule changes, at once (a running planet keeps its own size)."""
        self.rules.update({k: v for k, v in changes.items() if k != "planet_radius"})

    def edit_genes(self, creature_id: int, genes: Dict[str, Any]) -> None:
        """The owner's change to a living creature's genes (checked by rules.check_genes); a creature gone is left be.
        Its body follows a new size; a new brain size is its learner's."""
        creature = next((c for c in self.creatures if c.id == creature_id), None)
        if creature is None:
            return
        for name, value in genes.items():
            if name == "learning_speed":
                if creature.policy == "learner":
                    self.learner.set_lr(creature.slot, float(value))
            elif name == "mutations":
                creature.mutations = int(value)
            elif name in EDITABLE_GENES:
                setattr(creature, name, float(value))
        if "size" in genes:
            creature.max_energy = full_tank(creature.size, self.rules)
            creature.radius = creature.size * self.rules["body_radius"]
            creature.energy = min(creature.energy, creature.max_energy)
        if "layers" in genes and creature.policy == "learner":
            self.learner.reshape(creature.slot, int(genes["layers"]), [int(n) for n in genes["neurons"]])
            self._brain_size(creature)

    def add_creature(self, creature: Creature) -> None:
        creature.rules = self.rules
        self.creatures.append(creature)

    def set_settings(self, settings: Settings, level: int) -> None:
        self.settings = settings
        self.level = level

    def kill(self, creature: Creature, cause: str) -> None:
        """A death, of any cause (hunger, old age, a hit): it leaves a corpse, meat worth its size × the corpse's
        energy, where it died."""
        corpse = Food(creature.p.copy(), remaining_duration=round(self.rules["corpse_steps"]),
                      energy=creature.size * self.rules["corpse_energy"], meat=True)
        self.foods.append(corpse)
        self.foods_created_this_step.add(id(corpse))
        self.creatures.remove(creature)
        self.species_book.died(creature.species, cause)
        self.deaths[cause] += 1
        self.dead_this_step.append(creature)

    def add_newcomer(self) -> Creature:
        """A creature of this world's species with a random brain, at a random spot: the first creatures, and the
        primordial soup."""
        r = self.rules
        spot = self._free_spot(r["gene.size.new_high"] * r["body_radius"])
        size = self.rng.uniform(r["gene.size.new_low"], r["gene.size.new_high"])
        brain = Brain.random(self.np_rng) if self.species == "brain" else None
        drawn = lambda gene: self.rng.uniform(r[f"gene.{gene}.new_low"], r[f"gene.{gene}.new_high"])  # noqa: E731
        whole = lambda gene: self.rng.randint(round(r[f"gene.{gene}.new_low"]), round(r[f"gene.{gene}.new_high"]))  # noqa: E731
        slot = None
        if self.species == "learner":
            neurons = [whole("neurons") for _ in range(whole("layers"))]  # every layer its own count
            slot = self.learner.newcomer(self.np_rng, drawn("learning_speed"), neurons, r["learn_spread"])
        energy = self.rng.uniform(r["newcomer_energy_low"], r["newcomer_energy_high"])
        newcomer = Creature(self.next_id(), spot, size, energy, self.species, brain, slot=slot, rules=r)
        newcomer.facing = local_frame(spot)[0] if self.rng.random() < 0.5 else -local_frame(spot)[1]  # any way
        newcomer.lifespan = self._lifespan(size)
        newcomer.vision_width = self.rng.uniform(r["gene.vision_width.new_low"], r["gene.vision_width.new_high"])
        newcomer.vision_reach = self.rng.uniform(r["gene.vision_reach.new_low"], r["gene.vision_reach.new_high"])
        newcomer.hearing = self.rng.uniform(r["gene.hearing.new_low"], r["gene.hearing.new_high"])
        newcomer.fear_of_death, newcomer.love_of_babies = drawn("fear"), drawn("love")
        newcomer.learns_from_others = drawn("learns")
        newcomer.grass_gut, newcomer.meat_gut = drawn("grass_gut"), drawn("meat_gut")
        newcomer.calls = whole("calls")
        newcomer.kin = drawn("kin")
        newcomer.gills, newcomer.lungs = drawn("gills"), drawn("lungs")
        newcomer.stealth = drawn("stealth")
        newcomer.armour = drawn("armour")
        newcomer.eye_pixels = whole("eye_pixels")
        newcomer.mutations, newcomer.mutation_size = whole("mutations"), drawn("mutation_size")
        self._brain_size(newcomer)
        self.creatures.append(newcomer)
        self.newcomers += 1
        return newcomer

    def _free_spot(self, radius: float) -> np.ndarray:
        """A random spot anywhere on the planet where a body of `radius` touches no obstacle."""
        for _ in range(FREE_SPOT_TRIES):
            p = random_point(self.rng)
            if not self.terrain.blocked(p, radius):
                return p
        raise RuntimeError("the obstacles leave no room for a newcomer")

    def spawn_food(self) -> None:
        """The same number of candidates every step, so two worlds with the same food seed draw the same food."""
        season = self.season()
        per_step = self.settings.food_per_step * season
        whole = int(per_step)  # e.g. 2.4 per step: 2 every step and a third 40 % of the time
        extra = self.food_rng.random() < per_step - whole
        ponds = self.terrain.has_water()
        fruit = sum(f.fruit for f in self.foods)  # fruit has its own room (max_fruit); the rest share max_food
        for _ in range(whole + extra):
            p = self._land_spot() if ponds else random_point(self.food_rng)
            energy = self.food_rng.uniform(*self.settings.food_energy)
            wet = ponds and self.food_rng.random() < self.rules["pond_food_share"]  # some grows in a pond
            if wet:
                p = self.terrain.pond_spot(self.food_rng)
            if len(self.foods) - fruit >= int(self.settings.max_food):
                continue
            food = Food(p, remaining_duration=-1, energy=energy)
            if self._fits(food, wet):
                self.foods.append(food)
                self.foods_created_this_step.add(id(food))
        for tree in self.terrain.trees:  # each tree drops a fruit now and then, near its trunk
            if self.food_rng.random() >= self.settings.fruit_rate * season:
                continue
            angle, away = self.food_rng.uniform(0.0, 2.0 * math.pi), self.food_rng.uniform(0.0, self.rules["fruit_spread"])
            trunk = np.array(tree.p)
            e1, e2 = local_frame(trunk)
            p = exp_map(trunk, e1, e2, away * math.cos(angle), away * math.sin(angle), self.radius)
            dropped = Food(p, remaining_duration=-1, energy=self.food_rng.uniform(*self.settings.food_energy), fruit=True)
            if fruit < int(self.rules["max_fruit"]) and self._fits(dropped):
                self.foods.append(dropped)
                self.foods_created_this_step.add(id(dropped))
                fruit += 1
    def _land_spot(self) -> np.ndarray:
        """A spot for new food on the land, food growing `pond_edge_food` times as thick near a pond's edge: a spot
        away from the edges is kept 1 time in `pond_edge_food` (the same food a step, more of it by the water). On a
        planet with seas, a spot in the water is tried again (the pond share is what grows in the water)."""
        thicker, width = self.rules["pond_edge_food"], self.rules["pond_edge_width"]
        for _ in range(FOOD_SPOT_TRIES):
            p = random_point(self.food_rng)
            if self.terrain.water is not None and self.terrain.in_pond(p):
                continue
            if thicker <= 1.0 or self.terrain.near_pond(p, width) or self.food_rng.random() < 1.0 / thicker:
                return p
        return p

    def season(self) -> float:
        """How much food and fruit grow now, against the year's middle: 1 + depth × sin(2π step ÷ year). An exam world
        has no seasons."""
        if not self.evolving:
            return 1.0
        return 1.0 + self.rules["season_depth"] * math.sin(2.0 * math.pi * self.step_count / self.rules["year_steps"])

    def prefill_food(self) -> None:
        for _ in range(self.PREFILL_STEPS):
            self.spawn_food()

    def _fits(self, food: Food, wet: bool = False) -> bool:
        """Food fits where nothing else is and no obstacle stands (a pond's own food, where no rock stands)."""
        if self.terrain.blocked(food.p, food.radius, swims=wet):
            return False
        neighbours = list(self.foods) + (list(self.creatures) if self.evolving else [])
        if not neighbours:
            return True
        where = np.stack([n.p for n in neighbours])
        room = np.array([n.radius for n in neighbours]) + food.radius + self.MIN_SPAWN_DISTANCE
        return bool(np.all(arc(food.p[None, :], where, self.radius) >= room))

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

        self.calls = []  # this step's calls, heard in the next step's senses
        for phase in ("CALL", "ATTACK", "EAT", "MOVE"):
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
                  and creature.energy < self.rules["low_energy"] * creature.max_energy):
                self.kill(creature, "age")  # past its lifespan, only a creature with low energy dies

        if self.learner is not None and self.evolving:
            self._teach(acting, energy_before)  # before births: the energy given to a baby is never a penalty
            for dead in self.dead_this_step:
                if dead.policy == "learner":
                    self.learner.release(dead.slot)
        if self.evolving:
            self._births()
            if len(self.creatures) < self.rules["soup"]:
                self.add_newcomer()
        self._decay_food()

    def _teach(self, acting: List[Creature], energy_before: List[float]) -> None:
        """Each learner creature's reward for this step is its change in energy, valued by its own genes: plus its love
        of babies for a baby it had the step before, minus its fear of death if it died (which ends its experience).
        Each learner's step is also a lesson for every other living learner that sees or hears it."""
        alive = set(map(id, self.creatures))
        events = {}
        shared = self._kin_shares(acting, energy_before)
        for creature, before in zip(acting, energy_before):
            if creature.policy == "learner":
                died, baby = id(creature) not in alive, id(creature) in self._had_baby
                events[id(creature)] = (creature.energy - before, died, baby)
                reward = self._valued(creature, *events[id(creature)]) + shared.get(id(creature), 0.0)
                self.learner.record(creature.slot, reward, died)
        self._had_baby.clear()
        self._show(acting, events)
        self.learner.update()

    def _kin_shares(self, acting: List[Creature], energy_before: List[float]) -> Dict[int, float]:
        """Each caring learner's share of what happened to those it perceives this step: its kin gene × Σ how alike
        each one is × that one's change in energy (never its own)."""
        carers = [c for c in acting if c.policy == "learner" and c.kin > 0]
        if not carers:
            return {}
        changes = np.array([c.energy - before for c, before in zip(acting, energy_before)])
        sees = perceived(carers, acting, self).T & (np.array([id(c) for c in carers])[:, None]
                                                    != np.array([id(c) for c in acting])[None, :])
        carer, actor = np.nonzero(sees)  # only the pairs that perceive each other: how alike is asked of those alone
        how_alike = np.ones(len(carer))
        genes = kin_genes(self.rules)  # with no gene to tell them apart, all are alike
        if genes and len(carer):
            spans = np.array([self.rules[f"gene.{g}.high"] - self.rules[f"gene.{g}.low"] for g in genes])
            mine = np.array([[getattr(c, g) for g in genes] for c in carers]) / spans
            theirs = np.array([[getattr(c, g) for g in genes] for c in acting]) / spans
            how_alike = alike(mine[carer], theirs[actor], genes)
        share = np.bincount(carer, weights=how_alike * changes[actor], minlength=len(carers))
        return {id(c): c.kin * s for c, s in zip(carers, share)}

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
        seen = perceived(watchers, actors, self)
        for i, j in zip(*np.nonzero(seen)):
            watcher, actor = watchers[j], actors[i]
            if watcher is not actor:
                reward = self._valued(watcher, *events[id(actor)])
                self.learner.watch(watcher.slot, actor.slot, reward=reward, weight=watcher.learns_from_others)

    def _births(self) -> None:
        """A creature with enough energy has a baby; the safety limit only pauses births."""
        for parent in list(self.creatures):
            if len(self.creatures) >= self.rules["room"]:
                return
            if parent.energy >= self.rules["birth_at"] * parent.max_energy:
                self._birth(parent)

    def _birth(self, parent: Creature) -> None:
        """The baby gets a share of the parent's energy and the parent's brain with mutations (a learner's brain as
        the world's `inherit` says: as it is now, or as the parent was born)."""
        slot = None
        picked = set(self.rng.sample(GENES, min(parent.mutations, len(GENES))))  # the genes this baby changes;
        change = parent.mutation_size  # each by up to this share; it copies every other exactly
        if parent.policy == "learner":
            if not self.learner.has_room():
                return
            lr = self._inherited(self.learner.lr(parent.slot), "learning_speed", picked, change)
            brain_genes = [g for g in ("layers", "neurons", "weights") if g in picked]
            slot = self.learner.baby(parent.slot, self.np_rng, lr, self.inherit, brain_genes, self.rng, self.rules)
        gift = self.rules["baby_share"] * parent.energy
        size = self._inherited(parent.size, "size", picked, change)
        ahead, left = self.rng.uniform(-1.0, 1.0), self.rng.uniform(-1.0, 1.0)  # beside its parent
        spot, facing = walk(parent.p, parent.facing, ahead, left, math.hypot(ahead, left), self.radius)
        if self.terrain.blocked(spot, size * self.rules["body_radius"]):  # never born inside an obstacle
            spot, facing = parent.p.copy(), parent.facing
        brain = parent.brain
        if brain is not None and "weights" in picked:  # an instinct brain changes only when its weights are the gene
            brain = brain.mutate(self.np_rng, self.rules["gene.weights.share"], self.rules["gene.weights.size"])
        parent.energy -= gift
        parent.had_baby_at = self.step_count
        baby = Creature(self.next_id(), spot, size, gift, parent.policy, brain,
                        parent_id=parent.id, generation=parent.generation + 1, slot=slot, rules=self.rules)
        baby.lifespan = self._lifespan(size)
        baby.facing = facing
        baby.vision_width = self._inherited(parent.vision_width, "vision_width", picked, change)
        baby.vision_reach = self._inherited(parent.vision_reach, "vision_reach", picked, change)
        baby.hearing = self._inherited(parent.hearing, "hearing", picked, change)
        baby.fear_of_death = self._inherited(parent.fear_of_death, "fear", picked, change)
        baby.love_of_babies = self._inherited(parent.love_of_babies, "love", picked, change)
        baby.learns_from_others = self._inherited(parent.learns_from_others, "learns", picked, change)
        baby.mutation_size = self._inherited(parent.mutation_size, "mutation_size", picked, change)
        baby.grass_gut = self._inherited(parent.grass_gut, "grass_gut", picked, change)
        baby.meat_gut = self._inherited(parent.meat_gut, "meat_gut", picked, change)
        baby.kin = self._inherited(parent.kin, "kin", picked, change)
        baby.gills = self._inherited(parent.gills, "gills", picked, change)
        baby.lungs = self._inherited(parent.lungs, "lungs", picked, change)
        baby.stealth = self._inherited(parent.stealth, "stealth", picked, change)
        baby.armour = self._inherited(parent.armour, "armour", picked, change)
        low, high = gene_limits(self.rules, "mutations")
        step = self.rng.choice((-1, 1)) if "mutations" in picked else 0  # a count: one more or one fewer
        baby.mutations = int(min(max(parent.mutations + step, low), high))
        low, high = gene_limits(self.rules, "calls")
        step = self.rng.choice((-1, 1)) if "calls" in picked else 0  # a count: one call more or one fewer
        baby.calls = int(min(max(parent.calls + step, low), high))
        low, high = gene_limits(self.rules, "eye_pixels")
        step = self.rng.choice((-1, 1)) if "eye_pixels" in picked else 0  # a pixel sharper or blurrier
        baby.eye_pixels = int(min(max(parent.eye_pixels + step, low), high))
        baby.species = parent.species  # a baby is in its parent's species
        self.species_book.born(parent.species)
        self._brain_size(baby)
        self.creatures.append(baby)
        self.births += 1
        if parent.policy == "learner":
            self._had_baby.add(id(parent))

    def _brain_size(self, creature: Creature) -> None:
        """A learner's brain size, from its network (what it pays for each step)."""
        if creature.policy == "learner":
            creature.neurons, creature.layers = self.learner.active_neurons(creature.slot), self.learner.layers(creature.slot)

    def count_neurons(self) -> None:
        """Every creature's brain size, from the learner (after a load, or creatures added by hand)."""
        for creature in self.creatures:
            self._brain_size(creature)

    def _lifespan(self, size: float) -> int:
        """Steps a body of `size` lives: drawn from the world's lifespan, longer for a bigger body."""
        return round(self.rng.uniform(*self.settings.lifespan) * size ** self.rules["lifespan_power"])

    def _inherited(self, value: float, gene: str, picked: Set[str], change: float) -> float:
        """A parent's gene for its baby: a picked gene ± up to `change` (the parent's mutation size) of the gene's whole
        range, never less than the smallest change's share of it; any other copied; kept in the gene's limits (which
        may have changed since the parent was born)."""
        low, high = gene_limits(self.rules, gene)
        if gene in picked:
            step = max(change, self.rules["gene_min_change"]) * (high - low)
            value += self.rng.uniform(-step, step)
        return min(max(value, low), high)

    def _decay_food(self) -> None:
        kept = []
        for food in self.foods:
            if id(food) not in self.foods_created_this_step:
                food.decay()
            if not food.is_expired() and food.energy > 0:
                kept.append(food)
        self.foods = kept
