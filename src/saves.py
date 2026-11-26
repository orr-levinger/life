"""Saves: a whole world at one step, written to one file, from which a new world can go on.

A save holds everything that makes the world what it is at that step: every creature, every brain as it is now (a
learning animal's learned and inborn networks, learning speed and learning state), every piece of food, the knobs and
Auto's state, the counters, and the random generators' state. Only a learning animal's unfinished experience (fewer
steps than one rollout) is not kept.

Each save says its format. A change of code that changes what a save holds brings the conversion of older saves with
it; a save this code cannot read is refused, never half-loaded.
"""
from __future__ import annotations

import json
import math
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
from .rules import HAND_MADE, ROOM_SLOTS, RULES, rule, starting_rules
from .terrain import Terrain, make_terrain
from .settings import START, TODAY_SENSES
from .world import World

SAVE_FORMAT = 26  # 2: Auto's state keeps the owner's hunters switch; 3: each creature's energy earned and spent;
# 4: each creature's senses (a vision cone and hearing) and 17 brain inputs; 5: each creature's values and its pain,
# and 21 brain inputs; 6: the world's rules; 7: cones at most 120°, hearing without size (20 inputs), vision and
# hearing priced apart, the sprint's power; 8: the brain's size in the genes, room for 3 layers of 64; 9: every number
# a rule, the three brain genes (2026-11-24); 10: no rule decides who counts as bigger (2026-11-25); 11: one reach for eating and attacking; 12: how
# many genes a baby changes and by how much are genes; 13: a newcomer draws every gene from a range; 14: terrain
# (obstacles and trees), fruit, the senses of an obstacle in view and of being under a tree (25 inputs); 15: a gene
# changes by a share of its range, at least the smallest change; 16: a learner keeps a slot for each of up to 300
# creatures (a world's own rules keep its size and room); 17: size by curves (six powers); 18: the diet
# gene, meat (a corpse), the senses of meat seen and of its own diet (27 inputs); 19: calls (a gene, 10 kinds
# closed above it, a volume), the heard call (44 inputs); 20: care for kin (a gene), family seen and attacked like
# anyone, how alike the one it sees is (45 inputs); 21: water (a gene; ponds let swimmers in, grow food, hide
# nothing), the senses of being in water and of water ahead (47 inputs), whether each moved last step; 22: stealth;
# 23: armour; 24: seasons; 25: forests and rich pond edges (rules only: a world keeps its ground);
# 26: the exam world's own size
SAVES = "saves"  # the folder of a run's saves
BRAIN_PARTS = ("w1", "b1", "w2", "b2")
LEARNER = "learner."
INPUTS_HEARD = 5  # format 4: the brain hears (inputs 13-17)
INPUTS_PAIN = 4  # format 5: the brain feels pain (inputs 18-21)
INPUTS_TERRAIN = 5  # format 14: an obstacle in view (4 inputs) and being under a tree (1)
INPUTS_DIET = 2  # format 18: the nearest food it sees is meat, its own diet
INPUTS_WATER = 2  # format 21: in water, the nearest obstacle in view is a pond
INPUTS_KIN = 1  # format 20: how alike the nearest creature it sees is
INPUTS_CALL = 17  # format 19: the loudest call heard: its kind (10), how loud, from where (2), the voice (4)
HEARD_SIZE = 16  # format 7: the heard size (input 17, from 0: 16) leaves the brain


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
        "creatures": [{"id": c.id, "slot": c.slot, "x": c.x, "y": c.y, "size": c.size, "policy": c.policy,
                       "parent_id": c.parent_id, "generation": c.generation, "energy": c.energy, "age": c.age,
                       "lifespan": c.lifespan, "food_eaten": c.food_eaten, "last_action": c.last_action,
                       "intent": c.intent, "heading": list(c.intended_vector), "brain": brain,
                       "earned": c.earned, "spent": c.spent, "vision_width": c.vision_width,
                       "vision_reach": c.vision_reach, "hearing": c.hearing, "facing": list(c.facing),
                       "fear_of_death": c.fear_of_death, "love_of_babies": c.love_of_babies,
                       "learns_from_others": c.learns_from_others, "mutations": c.mutations,
                       "mutation_size": c.mutation_size, "diet": c.diet, "calls": c.calls, "kin": c.kin, "water": c.water, "moved": c.moved, "stealth": c.stealth, "armour": c.armour,
                       "hurt": None if c.hurt is None else list(c.hurt)}
                      for c, brain in zip(world.creatures, brain_of)],
        "foods": [{"x": f.x, "y": f.y, "energy": f.energy, "remaining_duration": f.remaining_duration, "fruit": f.fruit,
                   "meat": f.meat, "radius": f.radius} for f in world.foods],  # a food keeps the radius it was made with as it is eaten
        "calls": [{"id": c.id, "x": c.x, "y": c.y, "size": c.size, "diet": c.diet, "water": c.water, "stealth": c.stealth,
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
    if about.get("format") == 1:
        about = _from_format_1(about)
    if about.get("format") == 2:
        about = _from_format_2(about)
    if about.get("format") == 3:
        about = _from_format_3(about)
        _widen_inputs(arrays, INPUTS_HEARD)
    if about.get("format") == 4:
        about = _from_format_4(about)
        _widen_inputs(arrays, INPUTS_PAIN)
    if about.get("format") == 5:
        about = _from_format_5(about)
    if about.get("format") == 6:
        about = _from_format_6(about)
        _drop_input(arrays, HEARD_SIZE)
    if about.get("format") == 7:
        about = _from_format_7(about)
        _room_to_grow(arrays)
    if about.get("format") == 8:
        about = _from_format_8(about)
    if about.get("format") == 9:
        about = _from_format_9(about)
    if about.get("format") == 10:
        about = _from_format_10(about)
    if about.get("format") == 11:
        about = _from_format_11(about)
    if about.get("format") == 12:
        about = _from_format_12(about)
    if about.get("format") == 13:
        about = _from_format_13(about)
        _widen_inputs(arrays, INPUTS_TERRAIN)
    if about.get("format") == 14:
        about = {**about, "format": 15, "rules": {**about["rules"], "gene_min_change": rule("gene_min_change").start}}
    if about.get("format") == 15:
        about = {**about, "format": 16}  # its rules stay as they are: its size and its room
        _more_slots(arrays)
    if about.get("format") == 16:
        powers = ("tank_power", "living_power", "speed_power", "lifespan_power", "heard_power", "seen_power")
        about = {**about, "format": 17, "rules": {**about["rules"], **{p: rule(p).start for p in powers}}}
    if about.get("format") == 17:
        about = _from_format_17(about)
        _widen_inputs(arrays, INPUTS_DIET)
    if about.get("format") == 18:
        rules = {**about["rules"], **{r.name: r.start for r in RULES if r.name.startswith("gene.calls") or r.name == "call_reach"}}
        about = {**about, "format": 19, "rules": rules, "creatures": [{**c, "calls": 0} for c in about["creatures"]]}
        _widen_inputs(arrays, INPUTS_CALL)
        _calls_closed(arrays)
    if about.get("format") == 19:
        rules = {**about["rules"], **{r.name: r.start for r in RULES if r.name.startswith("gene.kin")}}
        about = {**about, "format": 20, "rules": rules, "creatures": [{**c, "kin": 0.5} for c in about["creatures"]]}
        _widen_inputs(arrays, INPUTS_KIN)
    if about.get("format") == 20:
        new = [r for r in RULES if r.name.startswith("gene.water") or r.name in ("land_penalty", "pond_food_share")]
        about = {**about, "format": 21, "rules": {**about["rules"], **{r.name: r.start for r in new}},
                 "creatures": [{**c, "water": 0.0, "moved": False} for c in about["creatures"]]}  # they cannot swim
        _widen_inputs(arrays, INPUTS_WATER)
    if about.get("format") == 21:
        new = [r for r in RULES if r.name.startswith("gene.stealth") or r.name in ("stealth_strength", "stealth_cost")]
        about = {**about, "format": 22, "rules": {**about["rules"], **{r.name: r.start for r in new}},
                 "creatures": [{**c, "stealth": 0.0} for c in about["creatures"]]}  # seen and heard as before
    if about.get("format") == 22:
        new = [r for r in RULES if r.name.startswith("gene.armour") or r.name.startswith("armour_")]
        about = {**about, "format": 23, "rules": {**about["rules"], **{r.name: r.start for r in new}},
                 "creatures": [{**c, "armour": 0.0} for c in about["creatures"]]}  # hit, fast and fed as before
    if about.get("format") == 23:
        about = {**about, "format": 24,
                 "rules": {**about["rules"], **{name: rule(name).start for name in ("year_steps", "season_depth")}}}
    if about.get("format") == 24:
        regions = ("forests", "forest_spread", "pond_edge_food", "pond_edge_width")
        about = {**about, "format": 25, "rules": {**about["rules"], **{name: rule(name).start for name in regions}}}
    if about.get("format") == 25:
        about = {**about, "format": 26,
                 "rules": {**about["rules"], **{name: rule(name).start for name in ("exam_width", "exam_height")}}}
    if about.get("format") != SAVE_FORMAT:
        raise SaveError(f"{path} is save format {about.get('format')}; this code reads format {SAVE_FORMAT}")
    arrays["about"] = about
    return arrays


def _food(saved: Dict[str, Any]) -> Food:
    food = Food(saved["x"], saved["y"], saved["remaining_duration"], saved["energy"], fruit=saved["fruit"], meat=saved["meat"])
    food.radius = saved["radius"]
    return food


def _from_format_1(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 1 save had no owner's hunters switch: the owner had not set one."""
    return {**about, "format": 2, "dial": {**about["dial"], "hunters": None}}


def _from_format_2(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 2 save had no creature's own energy counts: they start at zero from here."""
    return {**about, "format": 3, "creatures": [{**c, "earned": 0.0, "spent": 0.0} for c in about["creatures"]]}


def _from_format_3(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 3 save had no sense genes: its creatures get today's senses, facing where they last headed."""
    width, reach, hearing = TODAY_SENSES

    def facing(heading: List[float]) -> List[float]:
        length = math.hypot(*heading)
        return [heading[0] / length, heading[1] / length] if length > 0 else [1.0, 0.0]

    return {**about, "format": 4, "creatures": [{**c, "vision_width": width, "vision_reach": reach, "hearing": hearing,
                                                 "facing": facing(c["heading"])} for c in about["creatures"]]}


def _from_format_4(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 4 save had no value genes and no pain: its creatures get the starting amounts and feel nothing."""
    fear, love, learns = (HAND_MADE[gene] for gene in ("fear", "love", "learns"))
    return {**about, "format": 5, "creatures": [{**c, "fear_of_death": fear, "love_of_babies": love,
                                                 "learns_from_others": learns, "hurt": None}
                                                for c in about["creatures"]]}


def _from_format_5(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 5 save had no rules of its own: the starting rules as format 6 had them (one price for the senses, the
    square for the sprint), at its size."""
    rules = {**starting_rules(), "world_width": float(about["width"]), "world_height": float(about["height"])}
    rules["sense_cost"] = rules.pop("vision_cost")
    del rules["hearing_cost"], rules["sprint_power"]
    rest = {key: value for key, value in about.items() if key not in ("width", "height")}
    return {**rest, "format": 6, "rules": rules}


def _from_format_6(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 6 save allowed cones to 360° and priced vision and hearing as one: a cone wider than 120° narrows to
    120° and reaches farther, so it sees as much ground as before; both senses keep the one price; the sprint's power
    is the square it was."""
    rules = dict(about["rules"])
    price = rules.pop("sense_cost")
    rules.update({"vision_cost": price, "hearing_cost": price, "sprint_power": 2.0})
    for key in ("gene.vision_width.high", "gene.vision_width.new_high"):
        rules[key] = min(rules[key], 120.0)
    for key in ("gene.vision_width.low", "gene.vision_width.new_low"):
        rules[key] = min(rules[key], 120.0)
    rules["gene.vision_reach.high"] = rules["gene.vision_reach.new_high"] = rule("gene.vision_reach.high").start

    def narrowed(creature: Dict[str, Any]) -> Dict[str, Any]:
        width, reach = creature["vision_width"], creature["vision_reach"]
        if width <= 120.0:
            return creature
        reach = min(reach * math.sqrt(width / 120.0), rules["gene.vision_reach.high"])
        return {**creature, "vision_width": 120.0, "vision_reach": reach}

    return {**about, "format": 7, "rules": rules, "creatures": [narrowed(c) for c in about["creatures"]]}


def _from_format_7(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 7 save had no brain genes: their rules start at their starting values."""
    new = ["neuron_cost"] + [r.name for r in RULES if r.name.startswith(("gene.neurons.", "gene.layers."))]
    return {**about, "format": 8, "rules": {**about["rules"], **{name: rule(name).start for name in new}}}


def _from_format_8(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 8 save had fewer rules: every rule it lacks starts at its starting value; the ones gone (the brain's
    old change and layer-share rules) leave."""
    kept = about["rules"]
    return {**about, "format": 9, "rules": {r.name: kept.get(r.name, r.start) for r in RULES}}


def _from_format_9(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 9 save had a rule for who counts as bigger: it leaves (the creatures learn whom to flee or chase)."""
    return {**about, "format": 10, "rules": {k: v for k, v in about["rules"].items() if k != "threat_ratio"}}


def _from_format_10(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 10 save had an eating reach and an attack reach: one reach, the eating one's value. (An older save
    converted on the way here already has today's rules, its reach among them.)"""
    old = about["rules"]
    rules = {k: v for k, v in old.items() if k not in ("eat_range", "attack_range")}
    return {**about, "format": 11, "rules": {**rules, "reach": old["eat_range"] if "eat_range" in old else old["reach"]}}


def _from_format_11(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 11 save changed one gene a baby by up to a fixed share: every creature gets the starting mutation genes
    (one gene, 5 %), the rules gain their limits and the fixed shares leave."""
    new = [r.name for r in RULES if r.name.startswith(("gene.mutations.", "gene.mutation_size."))]
    rules = {k: v for k, v in about["rules"].items() if not (k.startswith("gene.") and k.endswith(".change"))}
    rules.update({name: rule(name).start for name in new})
    count, size = round(HAND_MADE["mutations"]), HAND_MADE["mutation_size"]
    creatures = [{**c, "mutations": count, "mutation_size": size} for c in about["creatures"]]
    return {**about, "format": 12, "rules": rules, "creatures": creatures}


def _from_format_12(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 12 save gave newcomers fixed values of some genes: those leave, and each gene's newcomer range starts at
    its starting values. Its creatures keep the genes they have."""
    rules = {k: v for k, v in about["rules"].items() if not (k.startswith("gene.") and k.endswith(".start"))}
    rules.update({r.name: r.start for r in RULES if r.name not in rules and r.name.endswith((".new_low", ".new_high"))})
    return {**about, "format": 13, "rules": rules}


def _from_format_13(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 13 save had no terrain: it gets the starting terrain rules and its own terrain, made around where its
    creatures stand; its food is not fruit; its settings (and Auto's) get the starting fruit rate."""
    rules = {**about["rules"], **{r.name: r.start for r in RULES if r.name not in about["rules"]}}
    clear = [(c["x"], c["y"], c["size"] * rules["body_radius"] * 2) for c in about["creatures"]]
    terrain = make_terrain(rules, random.Random(about["step"]), rules["world_width"], rules["world_height"], clear)
    fruit = {"fruit_rate": START.fruit_rate}
    dial = {**about["dial"], "settings": {**about["dial"]["settings"], **fruit},
            "taken": [{**s, **fruit} for s in about["dial"]["taken"]]}
    return {**about, "format": 14, "rules": rules, "terrain": terrain.state(), "settings": {**about["settings"], **fruit},
            "dial": dial, "foods": [{**f, "fruit": False} for f in about["foods"]]}


def _from_format_17(about: Dict[str, Any]) -> Dict[str, Any]:
    """A format 17 save had no diet: its creatures live half on plants, half on meat; a food that runs out (a corpse)
    is meat; the diet rules start at their starting values."""
    rules = {**about["rules"], **{r.name: r.start for r in RULES if r.name.startswith("gene.diet") or r.name == "diet_bite"}}
    return {**about, "format": 18, "rules": rules, "creatures": [{**c, "diet": 0.5} for c in about["creatures"]],
            "foods": [{**f, "meat": f["remaining_duration"] != -1 and not f["fruit"]} for f in about["foods"]]}


def _calls_closed(arrays: Dict[str, np.ndarray]) -> None:
    """A format 18 learner chose among 3 (move, eat, attack) and moved on 2 axes: it gains the 10 calls, each starting
    almost never chosen (bias CALL_BIAS, no weights), and a call's volume (no weights, the starting spread); what it
    did before stays as it was."""
    if f"{LEARNER}params.wc" not in arrays or arrays[f"{LEARNER}params.wc"].shape[-1] != 3:
        return
    from .learner import CALL_BIAS, INITIAL_SPREAD, N_CALLS  # TensorFlow is loaded only for learning animals

    def widen(key: str, added: int, value: float) -> None:
        widths = [(0, 0)] * arrays[key].ndim
        widths[-1] = (0, added)
        arrays[key] = np.pad(arrays[key], widths, constant_values=value)

    for kind in ("params", "m", "v", "inborn"):
        learned = kind in ("params", "inborn")  # weights; m and v are the optimizer's, which start at 0
        widen(f"{LEARNER}{kind}.wc", N_CALLS, 0.0)
        widen(f"{LEARNER}{kind}.bc", N_CALLS, CALL_BIAS if learned else 0.0)
        widen(f"{LEARNER}{kind}.wm", 1, 0.0)
        widen(f"{LEARNER}{kind}.bm", 1, 0.0)
        widen(f"{LEARNER}{kind}.log_std", 1, math.log(INITIAL_SPREAD) if learned else 0.0)


def _room_to_grow(arrays: Dict[str, np.ndarray]) -> None:
    """A format 7 learner's brain was 1 layer of 32 with no room: it keeps those 32, with room for 64 beside them and
    for 2 more layers (all 0, unused), and its size genes say [1, 32, 0, 0]."""
    if f"{LEARNER}size" in arrays or f"{LEARNER}params.w1" not in arrays:
        return
    slots = arrays[f"{LEARNER}params.w1"].shape[0]
    for kind in ("params", "m", "v", "inborn"):
        def pad(name: str, axis: int) -> None:
            key = f"{LEARNER}{kind}.{name}"
            widths = [(0, 0)] * arrays[key].ndim
            widths[axis] = (0, 64 - arrays[key].shape[axis])
            arrays[key] = np.pad(arrays[key], widths)
        pad("w1", -1)
        pad("b1", -1)
        pad("wc", -2)
        pad("wm", -2)
        for name, shape in (("w2", (64, 64)), ("b2", (64,)), ("s2", (1,)), ("w3", (64, 64)), ("b3", (64,)), ("s3", (1,))):
            arrays[f"{LEARNER}{kind}.{name}"] = np.zeros((slots,) + shape)
    arrays[f"{LEARNER}size"] = np.tile([1.0, 32.0, 0.0, 0.0], (slots, 1))
    arrays[f"{LEARNER}inborn_size"] = np.tile([1.0, 32.0, 0.0, 0.0], (slots, 1))


def _more_slots(arrays: Dict[str, np.ndarray]) -> None:
    """A format 15 learner had a slot for each of 80 creatures: the slots up to `ROOM_SLOTS` join empty and free, so
    its brains are kept as they are."""
    if f"{LEARNER}free" not in arrays:
        return
    slots = arrays[f"{LEARNER}free"].shape[0]
    for name in list(arrays):
        if name.startswith(LEARNER) and name != f"{LEARNER}generator":
            widths = [(0, 0)] * arrays[name].ndim
            widths[0] = (0, ROOM_SLOTS - slots)
            arrays[name] = np.pad(arrays[name], widths, constant_values=name == f"{LEARNER}free")


def _drop_input(arrays: Dict[str, np.ndarray], index: int) -> None:
    """A format 6 save's brains heard a size (input `index`): its weights leave every brain; the rest are kept."""
    for name in list(arrays):
        if name == "brain.w1" or (name.startswith(LEARNER) and name.split(".")[-1] in ("w1", "wh")):
            arrays[name] = np.delete(arrays[name], index, axis=-2)


def _widen_inputs(arrays: Dict[str, np.ndarray], added: int) -> None:
    """An older save's brains had fewer inputs: the `added` ones join with zero weights, so nothing they do changes."""
    for name in list(arrays):
        if name == "brain.w1" or (name.startswith(LEARNER) and name.split(".")[-1] in ("w1", "wh")):
            weights = arrays[name]
            pad = [(0, 0)] * weights.ndim
            pad[-2] = (0, added)
            arrays[name] = np.pad(weights, pad)


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
    world = World(rules["world_width"], rules["world_height"], settings_from_state(about["settings"]), about["species"],
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
        creature = Creature(saved["id"], saved["x"], saved["y"], saved["size"], saved["energy"], saved["policy"], brain,
                            parent_id=saved["parent_id"], generation=saved["generation"], slot=saved["slot"],
                            rules=world.rules)
        creature.energy, creature.age, creature.lifespan = saved["energy"], saved["age"], saved["lifespan"]
        creature.food_eaten, creature.last_action, creature.intent = saved["food_eaten"], saved["last_action"], saved["intent"]
        creature.intended_vector = tuple(saved["heading"])
        creature.earned, creature.spent = saved["earned"], saved["spent"]
        creature.vision_width, creature.vision_reach = saved["vision_width"], saved["vision_reach"]
        creature.hearing, creature.facing = saved["hearing"], tuple(saved["facing"])
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
    world.calls = [(living.get(saved["id"]) or SimpleNamespace(**{k: saved[k] for k in ("id", "x", "y", "size", "diet",
                                                                                       "water", "stealth")}),
                    saved["kind"], saved["volume"]) for saved in about.get("calls", [])]  # a caller since gone still heard
    world.count_neurons()
    return world, Dial.from_state(about["dial"]), about["extra"]
