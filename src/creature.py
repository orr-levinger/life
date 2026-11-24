"""A creature: its body, its energy, and what its actions do to it and to the world."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

from .settings import EAT_MISS_COST, HUNTER_SHARE

if TYPE_CHECKING:
    from .brain import Brain
    from .food import Food
    from .world import World


def is_family(a: Creature, b: Creature) -> bool:
    """Parent, child, or the same non-zero parent: family can never be attacked."""
    return a.id == b.parent_id or b.id == a.parent_id or (a.parent_id != 0 and a.parent_id == b.parent_id)


class Creature:
    RADIUS_FACTOR = 0.2
    SENSE_RANGE = 3.0
    ATTACK_RANGE_FACTOR = 3.0
    EAT_RANGE_FACTOR = 3.0
    POLICIES = ("brain", "learner", "random", "rules")
    MIN_SIZE = 0.5
    MAX_SIZE = 2.0
    CORPSE_ENERGY_PER_SIZE = 8.0
    CORPSE_STEPS = 5

    def __init__(self, id: int, x: float, y: float, size: float, energy: float, policy: str,
                 brain: Optional[Brain] = None, parent_id: int = 0, generation: int = 0,
                 slot: Optional[int] = None) -> None:
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

        self.max_energy = size * 100.0
        self.energy = min(energy, self.max_energy)
        self.velocity = 1.0 / size
        self.radius = size * self.RADIUS_FACTOR
        self.attack_damage = size * 5.0
        self.attack_cost = size * 1.0

        self.age = 0
        self.lifespan: Optional[int] = None  # steps it lives; the world sets it when the creature is born
        self.food_eaten = 0.0
        self.earned = 0.0  # energy it got by eating and hunting, over its life: with `spent`, its own intelligence
        self.spent = 0.0  # energy it used on moving, living, eating at nothing and attacking
        self.last_action = "NONE"
        self.intent = "REST"  # display only
        self.intended_vector: Tuple[float, float] = (0.0, 0.0)  # display only

    def can_reach(self, food: Food) -> bool:
        """The one eat-range rule: close enough to take a bite."""
        reach = (self.radius + food.radius) * self.EAT_RANGE_FACTOR
        return math.hypot(food.x - self.x, food.y - self.y) <= reach

    def _spend(self, amount: float, world: World) -> None:
        self.spent += amount
        world.energy_spent += amount

    def _earn(self, amount: float, world: World) -> None:
        self.earned += amount
        world.energy_earned += amount

    def apply_action(self, action: Tuple[str, Any], world: World) -> None:
        kind, param = action
        if kind == "MOVE":
            self._move(param, world)
        elif kind == "EAT":
            self._eat(param, world)
        elif kind == "ATTACK":
            self._attack(param, world)
        else:
            raise ValueError(f"unknown action {kind!r}")
        self.energy = min(self.energy - world.settings.step_cost, self.max_energy)
        self._spend(world.settings.step_cost, world)

    def _eat(self, food: Optional[Food], world: World) -> None:
        """The creature chose to eat: one bite of `food` if it is still there and within reach, otherwise a miss."""
        if food is None or food not in world.foods or food.energy <= 0 or not self.can_reach(food):
            self.energy -= EAT_MISS_COST
            self._spend(EAT_MISS_COST, world)
            self.last_action = "EAT_MISS"
            return
        amount = min(1.0, food.energy)
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
        if distance > self.velocity:
            dx, dy = dx * self.velocity / distance, dy * self.velocity / distance
            distance = self.velocity
        self.x = min(max(self.x + dx, self.radius), world.width - self.radius)
        self.y = min(max(self.y + dy, self.radius), world.height - self.radius)
        self.energy -= distance * world.settings.move_cost
        self._spend(distance * world.settings.move_cost, world)
        self.last_action = "MOVE"

    def _attack(self, target: Creature, world: World) -> None:
        if not world.settings.attacks:
            self.last_action = "ATTACK_OFF"
            return
        self.energy -= self.attack_cost
        self._spend(self.attack_cost, world)
        attack_range = (self.radius + target.radius) * self.ATTACK_RANGE_FACTOR
        if (target not in world.creatures or target.energy <= 0 or is_family(self, target)
                or math.hypot(target.x - self.x, target.y - self.y) > attack_range):
            self.last_action = "ATTACK_MISS"
            return
        taken = min(target.energy, self.attack_damage)
        target.energy -= taken
        self.energy += HUNTER_SHARE * taken
        self._earn(HUNTER_SHARE * taken, world)
        self.last_action = "ATTACK"
        target.last_action = "HIT_BY_ATTACK"
        if target.energy <= 0:
            from .food import Food
            corpse = Food(target.x, target.y, remaining_duration=self.CORPSE_STEPS,
                          energy=target.size * self.CORPSE_ENERGY_PER_SIZE)
            world.foods.append(corpse)
            world.foods_created_this_step.add(id(corpse))
            world.kill(target, "attack")

    def get_color(self) -> str:
        return "purple" if self.policy in ("brain", "learner") else "green"

    def __repr__(self) -> str:
        return (f"<Creature {self.id} {self.policy} x={self.x:.2f} y={self.y:.2f} size={self.size:.2f} "
                f"energy={self.energy:.2f}/{self.max_energy:.2f}>")
