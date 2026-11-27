"""A creature: its body, its energy, and what its actions do to it and to the world."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple


import numpy as np

from .sphere import arc, bearing, local_frame, walk
from .rules import HAND_MADE, Rules, rule, starting_rules
from .settings import TODAY_SENSES

if TYPE_CHECKING:
    from .brain import Brain
    from .food import Food
    from .world import World


# Every gene another creature shows to one that sees it, or that it hits (the owner, 2026-11-27: "they should see all
# the DNA of the other creature"); each scaled 0-1 by the gene's range, learning speed on a log scale.
SEEN_GENES = ("size", "diet", "water", "stealth", "armour", "vision_width", "vision_reach", "hearing", "calls", "kin",
              "fear", "love", "learns", "learning_speed", "mutations", "mutation_size", "neurons", "layers")
_GENE_ATTRS = {"fear": "fear_of_death", "love": "love_of_babies", "learns": "learns_from_others"}


def genes_table(creatures: Sequence["Creature"], world: Any) -> np.ndarray:
    """[creature, gene]: the genes each of `creatures` shows (SEEN_GENES), each 0-1 by its range."""
    rules, out = world.rules, np.zeros((len(creatures), len(SEEN_GENES)))
    if not len(creatures):
        return out
    for k, gene in enumerate(SEEN_GENES):
        low, high = rules[f"gene.{gene}.low"], rules[f"gene.{gene}.high"]
        if gene == "learning_speed":
            learner = getattr(world, "learner", None)
            values = np.array([learner.lr(c.slot) if learner is not None and c.policy == "learner" and c.slot is not None
                               else 0.0 for c in creatures])
            if low > 0 and high > low:
                share = (np.log10(np.maximum(values, 1e-300)) - math.log10(low)) / (math.log10(high) - math.log10(low))
                out[:, k] = np.where(values > 0, np.clip(share, 0.0, 1.0), 0.0)
            continue
        attr = _GENE_ATTRS.get(gene, gene)
        values = np.fromiter((getattr(c, attr, 0.0) or 0.0 for c in creatures), dtype=float, count=len(creatures))
        out[:, k] = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else 0.0
    return out


def genes_seen(other: "Creature", world: Any) -> np.ndarray:
    """The genes one creature shows (see `genes_table`)."""
    return genes_table([other], world)[0]


KIN_GENES = ("size", "diet", "water", "stealth", "armour", "vision_width", "vision_reach", "hearing")


def kin_genes(rules: Rules) -> list:
    """The genes that tell creatures apart in this world: those it has, each with a range."""
    return [g for g in KIN_GENES if f"gene.{g}.low" in rules and rules[f"gene.{g}.high"] > rules[f"gene.{g}.low"]]


def similarity(a: Creature, b: Creature, rules: Rules) -> float:
    """How alike two creatures' genes are: 1 minus the mean difference, each gene as a share of its range (1 =
    twins, 0 = as unlike as the genes allow). A gene the world does not have yet, or one the owner closed to a single
    value, is left out."""
    genes = kin_genes(rules)
    if not genes:
        return 1.0
    return 1.0 - sum(abs(getattr(a, g) - getattr(b, g)) / (rules[f"gene.{g}.high"] - rules[f"gene.{g}.low"])
                     for g in genes) / len(genes)


def is_family(a: Creature, b: Creature) -> bool:
    """Parent, child, or the same non-zero parent (the hand-written yardstick spares family; the world does not)."""
    return a.id == b.parent_id or b.id == a.parent_id or (a.parent_id != 0 and a.parent_id == b.parent_id)



def full_tank(size: float, rules: Rules) -> float:
    """The energy a body of `size` holds: more than its size (the tank × size to the tank power)."""
    return rules["tank"] * size ** rules["tank_power"]


class Creature:
    SENSE_RANGE = 3.0  # the rules policy's own fixed sense range; every other creature senses by its genes
    POLICIES = ("brain", "learner", "random", "rules")

    def __init__(self, id: int, p: np.ndarray, size: float, energy: float, policy: str,
                 brain: Optional[Brain] = None, parent_id: int = 0, generation: int = 0,
                 slot: Optional[int] = None, vision_width: float = TODAY_SENSES[0],
                 vision_reach: float = TODAY_SENSES[1], hearing: float = TODAY_SENSES[2],
                 fear_of_death: float = HAND_MADE["fear"], love_of_babies: float = HAND_MADE["love"],
                 learns_from_others: float = HAND_MADE["learns"], rules: Optional[Rules] = None) -> None:
        """`rules` are its world's (a creature outside any world follows the starting rules); its tank and radius are
        set at birth from them, its speeds and attack follow them as they change."""
        if policy not in self.POLICIES:
            raise ValueError(f"policy must be one of {self.POLICIES}, got {policy!r}")
        if policy == "brain" and brain is None:
            raise ValueError("a brain creature needs a brain")
        if policy == "learner" and slot is None:
            raise ValueError("a learner creature needs its slot in the world's learner")
        self.id = id
        self.slot = slot  # where a learner creature's network lives in the world's learner
        self.p = np.asarray(p, dtype=float)  # where it is: a point on the planet (a unit vector)
        self.size = size
        self.policy = policy
        self.brain = brain
        self.parent_id = parent_id
        self.generation = generation

        self.rules = starting_rules() if rules is None else rules
        self.max_energy = full_tank(size, self.rules)
        self.energy = min(energy, self.max_energy)
        self.radius = size * self.rules["body_radius"]
        self.vision_width = vision_width  # degrees of its vision cone: a gene
        self.vision_reach = vision_reach  # how far it sees: a gene
        self.hearing = hearing  # how far it hears, all around: a gene
        self.facing = local_frame(self.p)[0]  # where its cone points: a ground direction, the way it last moved
        self.fear_of_death = fear_of_death  # how bad dying feels to its learning: a gene
        self.love_of_babies = love_of_babies  # how good a baby feels to its learning: a gene
        self.learns_from_others = learns_from_others  # how much a lesson from another counts: a gene
        self.mutations = round(HAND_MADE["mutations"])  # how many genes each of its babies changes: a gene
        self.mutation_size = HAND_MADE["mutation_size"]  # how much each of those changes: a gene
        self.diet = HAND_MADE["diet"]  # 0 lives on plants, 1 on meat: a gene
        self.calls = round(HAND_MADE["calls"])  # how many kinds of call it can make: a gene
        self.kin = HAND_MADE["kin"]  # how much what happens to those like it counts as its own: a gene
        self.stealth = HAND_MADE["stealth"]  # how hard it is to notice: a gene
        self.armour = HAND_MADE["armour"]  # how well it takes a hit, slower and dearer to keep: a gene
        self.eye_pixels = HAND_MADE["eye_pixels"]  # how sharp its eye is (real pixels, 1-16), dearer when sharper: a gene
        self.water = HAND_MADE["water"]  # 0 cannot enter a pond; above 0 it swims at this share of its speed: a gene
        self.called: Optional[Tuple[int, float]] = None  # this step's call: its kind and how far it carries (display)
        self.target: Optional[int] = None  # the id of the one it attacked this step (display only)
        self.had_baby_at: Optional[int] = None  # the step it last had a baby (display only)
        self.neurons = 0  # the neurons its brain uses (a learner's, which the world sets); each costs energy
        self.layers = 0
        self.moved = False  # it moved in its last step: only then can it be heard
        self.bumped = False  # its last move ran into an obstacle (display only)
        self.hurt: Optional[Tuple[float, ...]] = None  # a hit, a step: share, where, how far, whose genes

        self.age = 0
        self.lifespan: Optional[int] = None  # steps it lives; the world sets it when the creature is born
        self.food_eaten = 0.0
        self.earned = 0.0  # energy it got by eating and hunting, over its life: with `spent`, its own intelligence
        self.spent = 0.0  # energy it used on moving, living, eating at nothing and attacking
        self.last_action = "NONE"
        self.intent = "REST"  # display only
        self.intended_vector: Tuple[float, float] = (0.0, 0.0)  # display only: its last move (ahead, left)

    @property
    def velocity(self) -> float:
        """Cruising speed: small ones are faster."""
        return self.rules["cruise_speed"] / self.size ** self.rules["speed_power"] * self._armoured()

    @property
    def top_speed(self) -> float:
        """Sprinting: big ones are faster, for a price; never below cruising."""
        return max(self.velocity, self.rules["sprint_speed"] * math.sqrt(self.size) * self._armoured())

    def _armoured(self) -> float:
        """Its speed's share for its armour: 1 - armour × the armour slowness."""
        return 1.0 - self.armour * self.rules["armour_slow"]

    def attack_damage(self, world: Any) -> float:
        """What its hit takes, by the world's cost: × its size, more for a meat eater."""
        return self.size * world.cost("attack_damage") * (1.0 + self.diet * self.rules["diet_bite"])

    def attack_cost(self, world: Any) -> float:
        """What an attack costs it, by the world's cost: × its size."""
        return self.size * world.cost("attack_cost")

    def can_reach(self, food: Food) -> bool:
        """The one eat-range rule: close enough to take a bite."""
        reach = (self.radius + food.radius) * self.rules["reach"]
        return float(arc(self.p, food.p, self.rules["planet_radius"])) <= reach  # along the ground

    def vision_area(self) -> float:
        return self.vision_width / 360.0 * math.pi * self.vision_reach ** 2

    def hearing_area(self) -> float:
        return math.pi * self.hearing ** 2

    def sense_area(self) -> float:
        """The ground its senses cover: its cone's area plus its hearing circle's."""
        return self.vision_area() + self.hearing_area()

    def _spend(self, amount: float, world: World) -> None:
        self.spent += amount
        world.energy_spent += amount

    def _earn(self, amount: float, world: World) -> None:
        self.earned += amount
        world.energy_earned += amount

    def apply_action(self, action: Tuple[str, Any], world: World) -> None:
        kind, param = action
        self.target = param.id if kind == "ATTACK" and param is not None else None  # whom it went for (display only)
        self.moved = self.bumped = False
        self.called = None
        if kind == "MOVE":
            self._move(param, world)
        elif kind == "EAT":
            self._eat(param, world)
        elif kind == "ATTACK":
            self._attack(param, world)
        elif kind == "CALL":
            self._call(*param, world)
        else:
            raise ValueError(f"unknown action {kind!r}")
        cost = world.cost
        living = (cost("step_cost") * self.size ** self.rules["living_power"]  # a bigger body pays more, less than its size
                  * (1.0 + self.stealth * cost("stealth_cost"))  # and stealth costs
                  * (1.0 + self.armour * cost("armour_cost"))  # and armour costs
                  + cost("vision_cost") * self.vision_area() + cost("hearing_cost") * self.hearing_area()
                  + cost("neuron_cost") * self.neurons + cost("pixel_cost") * self.eye_pixels) * self._home_factor(world)  # and each neuron; by where it is
        self.energy = min(self.energy - living, self.max_energy)
        self._spend(living, world)

    def _home_factor(self, world: World) -> float:
        """What its water gene makes of its moving and living costs where it is: less in water, more on land."""
        if world.terrain.in_pond(self.p):
            return 1.0 - self.water * self.rules["swim_saving"]
        return 1.0 + self.water * self.rules["land_cost"]

    def _pace(self, world: World) -> float:
        """Its share of its speed where it is: its water in a pond, 1 - water × the land penalty on land."""
        if world.terrain.in_pond(self.p):
            return self.water
        return 1.0 - self.water * self.rules["land_penalty"]

    def _call(self, kind: int, volume: float, world: World) -> None:
        """A call of `kind` (one its genes let it make) at `volume` 0-1: everyone within volume × the call reach hears
        it next step. It costs nothing but the step and being heard."""
        if not 0 <= kind < self.calls:
            self.last_action = "CALL_MISS"
            return
        volume = min(max(volume, 0.0), 1.0)
        world.calls.append((self, kind, volume))
        self.called = (kind, volume * self.rules["call_reach"])
        self.last_action = "CALL"

    def _eat(self, food: Optional[Food], world: World) -> None:
        """The creature chose to eat: one bite of `food` if it is still there and within reach, otherwise a miss."""
        if food is None or food not in world.foods or food.energy <= 0 or not self.can_reach(food):
            self.energy -= world.cost("eat_miss")
            self._spend(world.cost("eat_miss"), world)
            self.last_action = "EAT_MISS"
            return
        amount = min(self.rules["bite"], food.energy)
        food.energy -= amount
        gained = amount * (self.diet if food.meat else 1.0 - self.diet)  # what its diet makes of the bite
        self.energy += gained
        self.food_eaten += gained
        world.total_bites += 1
        self._earn(gained, world)
        if food.energy <= 0:
            world.foods.remove(food)
        self.last_action = "EAT"

    def _move(self, vector: Tuple[float, float], world: World) -> None:
        """A move in its own view: (ahead, left), as long as it wants to go, capped by its pace there."""
        dx, dy = vector
        distance = math.hypot(dx, dy)
        pace = self._pace(world)  # in water × its water, on land slower the more it is a swimmer
        home = self._home_factor(world)  # a move is priced where it starts
        top, cruise = self.top_speed * pace, self.velocity * pace
        if distance > top:
            dx, dy = dx * top / distance, dy * top / distance
            distance = top
        to, facing = walk(self.p, self.facing, dx, dy, distance, world.radius)
        if distance > 0:
            self.facing = facing  # it faces the way it moved, even when an obstacle stops it
        if distance > 0 and world.terrain.blocked(to, self.radius, swims=self.water > 0):  # it runs into an obstacle: it stays, and it hurts
            self.bumped = True
            self.energy -= world.cost("bump_cost")
            self._spend(world.cost("bump_cost"), world)
            self.last_action = "MOVE"
            return
        self.p = to
        self.moved = distance > 0  # a step that went somewhere makes a sound
        sprint = max(1.0, distance / max(cruise, 1e-9)) ** world.cost("sprint_power")  # faster than cruising costs more
        cost = distance * world.cost("move_cost") * self.size * sprint * home  # a bigger body pays more to move
        self.energy -= cost
        self._spend(cost, world)
        self.last_action = "MOVE"

    def _attack(self, target: Creature, world: World) -> None:
        if not world.settings.attacks:
            self.last_action = "ATTACK_OFF"
            return
        self.energy -= self.attack_cost(world)
        self._spend(self.attack_cost(world), world)
        reach = (self.radius + target.radius) * self.rules["reach"]
        if (target not in world.creatures or target.energy <= 0
                or float(arc(self.p, target.p, world.radius)) > reach):
            self.last_action = "ATTACK_MISS"
            return
        taken = min(target.energy, self.attack_damage(world) * (1.0 - target.armour * self.rules["armour_strength"]))
        target.energy -= taken
        ahead, left = bearing(target.p, target.facing, self.p)  # in the one hit's own view: where the hitter stands
        target.hurt = (taken / target.max_energy, float(ahead), float(left),  # how much, and where the hitter stands
                       float(arc(self.p, target.p, world.radius)), *map(float, genes_seen(self, world)))  # how far, who
        gained = self.rules["hunter_share"] * taken * self.diet  # a hit feeds a meat eater
        self.energy += gained
        self._earn(gained, world)
        self.last_action = "ATTACK"
        target.last_action = "HIT_BY_ATTACK"
        if target.energy <= 0:
            from .food import Food
            corpse = Food(target.p.copy(), remaining_duration=round(self.rules["corpse_steps"]),
                          energy=target.size * self.rules["corpse_energy"], meat=True)
            world.foods.append(corpse)
            world.foods_created_this_step.add(id(corpse))
            world.kill(target, "attack")

    def get_color(self) -> str:
        return "purple" if self.policy in ("brain", "learner") else "green"

    def __repr__(self) -> str:
        return (f"<Creature {self.id} {self.policy} at {np.round(self.p, 3)} size={self.size:.2f} "
                f"energy={self.energy:.2f}/{self.max_energy:.2f}>")
