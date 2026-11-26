"""Saves: a whole world at one step, written to one file, from which a new world can go on.

A save holds everything that makes the world what it is at that step: every creature, every brain as it is now (a
learning animal's learned and inborn networks, learning speed and learning state), every piece of food, the knobs and
Auto's state, the counters, and the random generators' state. Only a learning animal's unfinished experience (fewer
steps than one rollout) is not kept.

Each save says its format. A change of code that changes what a save holds brings the conversion of older saves with
it; a save this code cannot read is refused, never half-loaded. Saves from before the sphere (format 26 and older)
hold a flat world and cannot be resumed (the brains learned east and north, which a ball does not have).
"""
from __future__ import annotations

import json
import os
import random
import zipfile
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .brain import Brain
from .creature import Creature
from .dial import Dial, settings_from_state, settings_state
from .food import Food
from .rules import ROOM_SLOTS
from .terrain import Terrain
from .world import World

SAVE_FORMAT = 27  # 27: the planet (points on a sphere, facings in the ground, caps); every older format held a
# flat world (see the git history for formats 1-26)
FLAT_WORLD = "a flat world from before the sphere: it cannot be resumed"
SAVES = "saves"  # the folder of a run's saves
BRAIN_PARTS = ("w1", "b1", "w2", "b2")
LEARNER = "learner."

class SaveError(Exception):
    """A save that cannot be loaded: broken, or of a format this code does not read."""


def save_path(run: Path, step: int) -> Path:
    return run / SAVES / f"step-{step:07d}.npz"


def saves_of(run: Path) -> List[int]:
    """The steps a run has saves of, in order."""
    return sorted(int(path.stem.split("-")[1]) for path in (run / SAVES).glob("step-*.npz"))


def random_state(rng: random.Random) -> list:
    return list(rng.getstate())


def set_random_state(rng: random.Random, state: list) -> None:
    version, internal, gauss = state
    rng.setstate((version, tuple(internal), gauss))


def save_world(path: Path, world: World, dial: Dial, extra: Dict[str, Any]) -> None:
    """Writes the world at its current step (to a temporary file, then renamed, so a save is never half written)."""
    brains = [c.brain for c in world.creatures if c.brain is not None]
    brain_of, next_brain = [], 0
    for creature in world.creatures:
        brain_of.append(None if creature.brain is None else next_brain)
        next_brain += creature.brain is not None
    about = {
        "format": SAVE_FORMAT, "step": world.step_count, "species": world.species, "inherit": world.inherit,
        "rules": dict(world.rules), "settings": settings_state(world.settings), "level": world.level,
        "terrain": world.terrain.state(),
        "dial": dial.state(),
        "counters": {"births": world.births, "newcomers": world.newcomers, "deaths": dict(world.deaths),
                     "total_bites": world.total_bites, "energy_earned": world.energy_earned,
                     "energy_spent": world.energy_spent, "creature_steps": world.creature_steps,
                     "next_id": world._next_id},  # the id the next creature will get
        "random": {"rng": random_state(world.rng), "food_rng": random_state(world.food_rng),
                   "np_rng": world.np_rng.bit_generator.state},
        "creatures": [{"id": c.id, "slot": c.slot, "p": list(map(float, c.p)), "size": c.size, "policy": c.policy,
                       "parent_id": c.parent_id, "generation": c.generation, "energy": c.energy, "age": c.age,
                       "lifespan": c.lifespan, "food_eaten": c.food_eaten, "last_action": c.last_action,
                       "intent": c.intent, "heading": list(c.intended_vector), "brain": brain,
                       "earned": c.earned, "spent": c.spent, "vision_width": c.vision_width,
                       "vision_reach": c.vision_reach, "hearing": c.hearing, "facing": list(map(float, c.facing)),
                       "fear_of_death": c.fear_of_death, "love_of_babies": c.love_of_babies,
                       "learns_from_others": c.learns_from_others, "mutations": c.mutations,
                       "mutation_size": c.mutation_size, "diet": c.diet, "calls": c.calls, "kin": c.kin, "water": c.water, "moved": c.moved, "stealth": c.stealth, "armour": c.armour,
                       "hurt": None if c.hurt is None else list(c.hurt)}
                      for c, brain in zip(world.creatures, brain_of)],
        "foods": [{"p": list(map(float, f.p)), "energy": f.energy, "remaining_duration": f.remaining_duration, "fruit": f.fruit,
                   "meat": f.meat, "radius": f.radius} for f in world.foods],  # a food keeps the radius it was made with as it is eaten
        "calls": [{"id": c.id, "p": list(map(float, c.p)), "size": c.size, "diet": c.diet, "water": c.water, "stealth": c.stealth,
                   "kind": kind, "volume": volume} for c, kind, volume in world.calls],  # heard in the next step
        "extra": extra,
    }
    arrays: Dict[str, np.ndarray] = {"about": np.array(json.dumps(about))}
    if brains:
        arrays.update({f"brain.{part}": np.stack([getattr(b, part) for b in brains]) for part in BRAIN_PARTS})
    if world.learner is not None:
        arrays.update({LEARNER + key: value for key, value in world.learner.state().items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def _read(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        about = json.loads(str(arrays["about"]))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as problem:
        raise SaveError(f"{path} cannot be read: {problem}") from problem
    if isinstance(about.get("format"), int) and about["format"] < SAVE_FORMAT:
        raise SaveError(f"{path}: {FLAT_WORLD}")
    if about.get("format") != SAVE_FORMAT:
        raise SaveError(f"{path} is save format {about.get('format')}; this code reads format {SAVE_FORMAT}")
    arrays["about"] = about
    return arrays


def _food(saved: Dict[str, Any]) -> Food:
    food = Food(np.array(saved["p"]), saved["remaining_duration"], saved["energy"], fruit=saved["fruit"], meat=saved["meat"])
    food.radius = saved["radius"]
    return food


def save_format(path: Path) -> int:
    """The format a save says it is (0 when it cannot be read): what the page checks before it stops a world."""
    try:
        with np.load(path) as data:
            return int(json.loads(str(data["about"])).get("format", 0))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return 0


def save_header(path: Path) -> Dict[str, Any]:
    """The step a save is of, and the seed and kind of the run that made it."""
    about = _read(path)["about"]
    return {"step": about["step"], "seed": about["extra"]["seed"], "kind": about["extra"]["kind"]}


def load_world(path: Path, learner_seed: int) -> Tuple[World, Dial, Dict[str, Any]]:
    """The saved world, Auto's state, and the extra the runner kept with it."""
    arrays = _read(path)
    about = arrays.pop("about")
    learner = None
    if about["species"] == "learner":
        from .learner import Learner  # TensorFlow is loaded only for learning animals

        learner = Learner(ROOM_SLOTS, learner_seed)
        learner.load_state({key[len(LEARNER):]: value for key, value in arrays.items() if key.startswith(LEARNER)})
    rules = about["rules"]
    world = World(rules["planet_radius"], settings_from_state(about["settings"], rules), about["species"],
                  random.Random(), np.random.default_rng(), random.Random(), learner=learner, inherit=about["inherit"],
                  rules=dict(rules))
    world.set_settings(world.settings, about["level"])
    set_random_state(world.rng, about["random"]["rng"])
    set_random_state(world.food_rng, about["random"]["food_rng"])
    world.np_rng.bit_generator.state = about["random"]["np_rng"]
    counters = about["counters"]
    world.step_count = about["step"]
    world.births, world.newcomers, world.total_bites = counters["births"], counters["newcomers"], counters["total_bites"]
    world.deaths.update(counters["deaths"])
    world.energy_earned, world.energy_spent = counters["energy_earned"], counters["energy_spent"]
    world.creature_steps = counters["creature_steps"]
    world._next_id = counters["next_id"]
    for saved in about["creatures"]:
        brain = None if saved["brain"] is None else Brain(*(arrays[f"brain.{part}"][saved["brain"]] for part in BRAIN_PARTS))
        creature = Creature(saved["id"], np.array(saved["p"]), saved["size"], saved["energy"], saved["policy"], brain,
                            parent_id=saved["parent_id"], generation=saved["generation"], slot=saved["slot"],
                            rules=world.rules)
        creature.energy, creature.age, creature.lifespan = saved["energy"], saved["age"], saved["lifespan"]
        creature.food_eaten, creature.last_action, creature.intent = saved["food_eaten"], saved["last_action"], saved["intent"]
        creature.intended_vector = tuple(saved["heading"])
        creature.earned, creature.spent = saved["earned"], saved["spent"]
        creature.vision_width, creature.vision_reach = saved["vision_width"], saved["vision_reach"]
        creature.hearing, creature.facing = saved["hearing"], np.array(saved["facing"])
        creature.fear_of_death, creature.love_of_babies = saved["fear_of_death"], saved["love_of_babies"]
        creature.learns_from_others = saved["learns_from_others"]
        creature.mutations, creature.mutation_size = saved["mutations"], saved["mutation_size"]
        creature.diet, creature.calls, creature.kin = saved["diet"], saved["calls"], saved["kin"]
        creature.water, creature.moved = saved["water"], saved["moved"]  # a mover is heard next step
        creature.stealth, creature.armour = saved["stealth"], saved["armour"]
        creature.hurt = None if saved["hurt"] is None else tuple(saved["hurt"])
        world.add_creature(creature)
    world.foods = [_food(f) for f in about["foods"]]
    world.terrain = Terrain.from_state(about["terrain"])
    living = {c.id: c for c in world.creatures}
    world.calls = [(living.get(saved["id"]) or SimpleNamespace(p=np.array(saved["p"]), **{k: saved[k] for k in (
                        "id", "size", "diet", "water", "stealth")}),
                    saved["kind"], saved["volume"]) for saved in about.get("calls", [])]  # a caller since gone still heard
    world.count_neurons()
    return world, Dial.from_state(about["dial"], rules), about["extra"]
