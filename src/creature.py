"""A creature: its body, its energy, and what its actions do to it and to the world."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from .rules import HAND_MADE, Rules, rule, starting_rules
from .settings import TODAY_SENSES

if TYPE_CHECKING:
    from .brain import Brain
    from .food import Food
    from .world import World


def is_family(a: Creature, b: Creature) -> bool:
    """Parent, child, or the same non-zero parent: family can never be attacked."""
    return a.id == b.parent_id or b.id == a.parent_id or (a.parent_id != 0 and a.parent_id == b.parent_id)


class Creature:
    SENSE_RANGE = 3.0  # the rules policy's own fixed sense range; every other creature senses by its genes
    POLICIES = ("brain", "learner", "random", "rules")

    def __init__(self, id: int, x: float, y: float, size: float, energy: float, policy: str,
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
        self.x = float(x)
        self.y = float(y)
        self.size = size
        self.policy = policy
        self.brain = brain
        self.parent_id = parent_id
        self.generation = generation

        self.rules = starting_rules() if rules is None else rules
        self.max_energy = size * self.rules["tank"]
        self.energy = min(energy, self.max_energy)
        self.radius = size * self.rules["body_radius"]
        self.vision_width = vision_width  # degrees of its vision cone: a gene
        self.vision_reach = vision_reach  # how far it sees: a gene
        self.hearing = hearing  # how far it hears, all around: a gene
        self.facing: Tuple[float, float] = (1.0, 0.0)  # where its cone points: the direction of its last move
        self.fear_of_death = fear_of_death  # how bad dying feels to its learning: a gene
        self.love_of_babies = love_of_babies  # how good a baby feels to its learning: a gene
        self.learns_from_others = learns_from_others  # how much a lesson from another counts: a gene
        self.mutations = round(HAND_MADE["mutations"])  # how many genes each of its babies changes: a gene
        self.mutation_size = HAND_MADE["mutation_size"]  # how much each of those changes: a gene
        self.target: Optional[int] = None  # the id of the one it attacked this step (display only)
        self.had_baby_at: Optional[int] = None  # the step it last had a baby (display only)
        self.neurons = 0  # the neurons its brain uses (a learner's, which the world sets); each costs energy
        self.layers = 0
        self.moved = False  # it moved in its last step: only then can it be heard
        self.bumped = False  # its last move ran into an obstacle (display only)
        self.hurt: Optional[Tuple[float, float, float]] = None  # one step after a hit: share taken, direction to the hitter

        self.age = 0
        self.lifespan: Optional[int] = None  # steps it lives; the world sets it when the creature is born
        self.food_eaten = 0.0
        self.earned = 0.0  # energy it got by eating and hunting, over its life: with `spent`, its own intelligence
        self.spent = 0.0  # energy it used on moving, living, eating at nothing and attacking
        self.last_action = "NONE"
        self.intent = "REST"  # display only
        self.intended_vector: Tuple[float, float] = (0.0, 0.0)  # display only

    @property
    def velocity(self) -> float:
        """Cruising speed: small ones are faster."""
        return self.rules["cruise_speed"] / self.size

    @property
    def top_speed(self) -> float:
        """Sprinting: big ones are faster, for a price; never below cruising."""
        return max(self.velocity, self.rules["sprint_speed"] * math.sqrt(self.size))

    @property
    def attack_damage(self) -> float:
        return self.size * self.rules["attack_damage"]

    @property
    def attack_cost(self) -> float:
        return self.size * self.rules["attack_cost"]

    def can_reach(self, food: Food) -> bool:
        """The one eat-range rule: close enough to take a bite."""
        reach = (self.radius + food.radius) * self.rules["reach"]
        return math.hypot(food.x - self.x, food.y - self.y) <= reach

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
        if kind == "MOVE":
            self._move(param, world)
        elif kind == "EAT":
            self._eat(param, world)
        elif kind == "ATTACK":
            self._attack(param, world)
        else:
            raise ValueError(f"unknown action {kind!r}")
        living = (world.settings.step_cost * self.size  # being alive: a bigger body pays more
                  + self.rules["vision_cost"] * self.vision_area() + self.rules["hearing_cost"] * self.hearing_area()
                  + self.rules["neuron_cost"] * self.neurons)  # and each neuron of its brain
        self.energy = min(self.energy - living, self.max_energy)
        self._spend(living, world)

    def _eat(self, food: Optional[Food], world: World) -> None:
        """The creature chose to eat: one bite of `food` if it is still there and within reach, otherwise a miss."""
        if food is None or food not in world.foods or food.energy <= 0 or not self.can_reach(food):
            self.energy -= self.rules["eat_miss"]
            self._spend(self.rules["eat_miss"], world)
            self.last_action = "EAT_MISS"
            return
        amount = min(self.rules["bite"], food.energy)
        food.energy -= amount
        self.energy += amount
        self.food_eaten += amount
        world.total_bites += 1
        self._earn(amount, world)
        if food.energy <= 0:
            world.foods.remove(food)
        self.last_action = "EAT"

    def _move(self, vector: Tuple[float, float], world: World) -> None:
        dx, dy = vector
        distance = math.hypot(dx, dy)
        if distance > self.top_speed:
            dx, dy = dx * self.top_speed / distance, dy * self.top_speed / distance
            distance = self.top_speed
        if distance > 0:
            self.facing = (dx / distance, dy / distance)
        x, y = self.x, self.y
        to_x = min(max(self.x + dx, self.radius), world.width - self.radius)
        to_y = min(max(self.y + dy, self.radius), world.height - self.radius)
        if world.terrain.blocked(to_x, to_y, self.radius):  # it runs into an obstacle: it stays, and it hurts
            self.bumped = True
            self.energy -= self.rules["bump_cost"]
            self._spend(self.rules["bump_cost"], world)
            self.last_action = "MOVE"
            return
        self.x, self.y = to_x, to_y
        self.moved = (self.x, self.y) != (x, y)  # a step that went somewhere makes a sound
        sprint = max(1.0, distance / self.velocity) ** self.rules["sprint_power"]  # faster than cruising costs more
        cost = distance * world.settings.move_cost * self.size * sprint  # a bigger body pays more to move
        self.energy -= cost
        self._spend(cost, world)
        self.last_action = "MOVE"

    def _attack(self, target: Creature, world: World) -> None:
        if not world.settings.attacks:
            self.last_action = "ATTACK_OFF"
            return
        self.energy -= self.attack_cost
        self._spend(self.attack_cost, world)
        reach = (self.radius + target.radius) * self.rules["reach"]
        if (target not in world.creatures or target.energy <= 0 or is_family(self, target)
                or math.hypot(target.x - self.x, target.y - self.y) > reach):
            self.last_action = "ATTACK_MISS"
            return
        taken = min(target.energy, self.attack_damage)
        target.energy -= taken
        dx, dy = self.x - target.x, self.y - target.y
        distance = math.hypot(dx, dy)
        target.hurt = (taken / target.max_energy, dx / distance if distance else 0.0, dy / distance if distance else 0.0)
        self.energy += self.rules["hunter_share"] * taken
        self._earn(self.rules["hunter_share"] * taken, world)
        self.last_action = "ATTACK"
        target.last_action = "HIT_BY_ATTACK"
        if target.energy <= 0:
            from .food import Food
            corpse = Food(target.x, target.y, remaining_duration=round(self.rules["corpse_steps"]),
                          energy=target.size * self.rules["corpse_energy"])
            world.foods.append(corpse)
            world.foods_created_this_step.add(id(corpse))
            world.kill(target, "attack")

    def get_color(self) -> str:
        return "purple" if self.policy in ("brain", "learner") else "green"

    def __repr__(self) -> str:
        return (f"<Creature {self.id} {self.policy} x={self.x:.2f} y={self.y:.2f} size={self.size:.2f} "
                f"energy={self.energy:.2f}/{self.max_energy:.2f}>")
