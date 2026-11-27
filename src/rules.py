"""Every rule of the world, once: its words, its table on the Rules page, its starting value and its allowed range.

A world holds its own values (`World.rules`, a dict from rule name to value), changed on the Rules page and kept in its
saves; every module reads its numbers from there. Auto's six knobs (food, its energy and rate, the costs of moving and of
being alive, the lifespan) are the world's settings (`src/settings.py`), turned by Auto or by hand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .settings import SEEING

Rules = Dict[str, float]

# Auto's knobs (src/dial.py): each knob's easy end and hard end today (the start of its two rules, `knob.<name>.easy`
# and `.hard`, the owner's to change: 2026-11-27) and what it is, in plain words. More is easier when the easy end is
# the bigger.
KNOB_ENDS = (
    ("max_food", 800.0, 20.0, "Food pieces at most"),
    ("food_energy", 100.0, 3.0, "Energy per food"),
    ("lifespan", 10000.0, 625.0, "Lifespan"),
    ("food_per_step", 20.0, 0.2, "New food per step"),
    ("fruit_rate", 1.0, 0.002, "Fruit a tree drops per step"),
)

# What everything costs in nature (the owner, 2026-11-27): one anchor (a full size-1 creature standing still lasts about
# 1,000 steps without food) and ratios from animals. Each cost is a rule from 0 to its natural value, never above; a new
# world starts at a tenth; Auto turns it between (src/dial.py) and the owner may set it any time.
COSTS = (
    ("step_cost", 0.1, "Cost of living, each step, × size"),
    ("move_cost", 0.4, "Cost of moving, per unit, × size"),
    ("sprint_power", 1.0, "Cost of sprinting: the move × (speed ÷ cruising) to this power"),  # 1: the owner, 2026-11-27
    ("eat_miss", 0.1, "Cost of eating at nothing"),
    ("bump_cost", 0.2, "Cost of bumping into a rock"),
    ("attack_cost", 1.0, "Cost of an attack, × size"),
    ("attack_damage", 10.0, "Damage of a hit, × the hitter's size"),
    ("vision_cost", SEEING / 2.0, "Cost of seeing, per square unit, each step"),
    ("hearing_cost", SEEING / 2.0, "Cost of hearing, per square unit, each step"),
    ("neuron_cost", 0.0006, "Cost of each neuron, each step"),
    ("stealth_cost", 0.3, "Cost of stealth: living × (1 + stealth × this)"),
    ("armour_cost", 0.3, "Cost of armour: living × (1 + armour × this)"),
)
COST_NAMES = tuple(name for name, _, _ in COSTS)
# What everything cost before 2026-11-27: the exam's world and saves from before keep these.
TODAY_COSTS = {"step_cost": 0.05, "move_cost": 0.5, "sprint_power": 2.0, "eat_miss": 0.2, "bump_cost": 0.5,
               "attack_cost": 1.0, "attack_damage": 5.0, "vision_cost": SEEING, "hearing_cost": SEEING,
               "neuron_cost": 0.0005, "stealth_cost": 0.3, "armour_cost": 0.3}


@dataclass(frozen=True)
class Rule:
    name: str
    group: str  # its group on the Rules page (GROUPS)
    words: str  # what it is, in plain words
    start: float  # a new world's value
    low: float  # the allowed range
    high: float


# A creature made by hand, outside a world (the tests, the yardstick), gets these genes; the world never uses them:
# its newcomers draw every gene from its newcomer range, and babies copy their parents.
HAND_MADE = {"fear": 10.0, "love": 10.0, "learns": 0.5, "learning_speed": 3e-4, "mutations": 1.0, "mutation_size": 0.05,
             "neurons": 32.0, "layers": 1.0, "diet": 0.0, "calls": 0.0,
             "kin": 0.0, "water": 0.0, "stealth": 0.0, "armour": 0.0}  # one made by hand eats plants, never calls, cares for nobody, cannot swim


def _gene(name: str, words: str, low: float, high: float, limits: tuple, new: tuple) -> tuple:
    """A gene's rules: its lowest and highest, and the range a newcomer draws it from (each changeable within
    `limits`). How much a baby's copy changes is the parent's own gene, `mutation_size`."""
    return (Rule(f"gene.{name}.low", "genes", f"{words}: lowest", low, *limits),
            Rule(f"gene.{name}.high", "genes", f"{words}: highest", high, *limits),
            Rule(f"gene.{name}.new_low", "genes", f"{words}: a newcomer gets at least", new[0], *limits),
            Rule(f"gene.{name}.new_high", "genes", f"{words}: a newcomer gets at most", new[1], *limits))


RULES = (
    Rule("baby_share", "costs", "Having a baby: the share of its energy given to the baby", 0.4, 0.05, 0.9),
    Rule("planet_radius", "world", "A new planet's radius", 37.0, 5.0, 2000.0),
    Rule("room", "world", "Room for this many creatures (births pause when full)", 600.0, 10.0, 600.0),
    Rule("soup", "world", "Random newcomers appear while fewer than this many live", 80.0, 0.0, 600.0),
    Rule("newcomer_energy_low", "world", "A newcomer's energy: at least", 50.0, 1.0, 500.0),
    Rule("newcomer_energy_high", "world", "A newcomer's energy: at most", 100.0, 1.0, 500.0),
    Rule("tank", "body", "Energy tank, × size", 100.0, 20.0, 500.0),
    Rule("body_radius", "body", "Body radius, × size", 0.2, 0.05, 1.0),
    Rule("cruise_speed", "body", "Cruising speed, ÷ size", 0.25, 0.05, 1.0),
    Rule("sprint_speed", "body", "Sprint top speed, × √size", 0.8, 0.1, 3.0),
    Rule("tank_power", "body", "Energy tank grows as size to this power", 1.3, 0.5, 2.0),
    Rule("living_power", "body", "Living cost grows as size to this power", 0.75, 0.3, 1.5),
    Rule("speed_power", "body", "Cruising speed falls as size to this power", 0.5, 0.0, 1.5),
    Rule("lifespan_power", "body", "Lifespan grows as size to this power", 0.5, 0.0, 1.5),
    Rule("heard_power", "body", "A mover is heard within the hearing × its size to this power", 1.0, 0.0, 2.0),
    Rule("seen_power", "body", "A body is seen within the reach × its size to this power", 0.5, 0.0, 2.0),
    Rule("birth_at", "birth", "Has a baby at this share of its tank", 0.7, 0.3, 1.0),
    Rule("low_energy", "birth", "Past its lifespan it dies below this share of its tank", 0.3, 0.0, 0.9),
    Rule("reach", "senses", "Reach for eating and attacking, × both radii", 3.0, 1.0, 10.0),
    Rule("call_reach", "senses", "A call at full volume carries this far", 12.0, 0.5, 60.0),
    Rule("diet_bite", "senses", "A meat eater's hit takes more: × (1 + diet × this)", 1.0, 0.0, 5.0),
    Rule("hunter_share", "senses", "A hunter keeps this share of what it takes", 0.8, 0.0, 1.0),
    Rule("corpse_energy", "senses", "A corpse's food, × the prey's size", 8.0, 0.0, 50.0),
    Rule("corpse_steps", "senses", "A corpse lasts this many steps", 5.0, 1.0, 100.0),
    Rule("first_creatures", "world", "A new world starts with this many creatures", 200.0, 1.0, 600.0),
    Rule("obstacles", "world", "Rocks a new world gets", 70.0, 0.0, 300.0),
    Rule("water_share", "world", "Share of a new planet under water", 0.5, 0.0, 0.8),
    Rule("islands", "world", "Islands a new planet gets", 8.0, 0.0, 30.0),
    Rule("island_gap", "world", "An island is at least this far from other land", 6.0, 0.0, 20.0),
    Rule("rivers", "world", "Rivers a new planet gets", 3.0, 0.0, 10.0),
    Rule("river_crossings", "world", "Shallow crossings on each river", 2.0, 0.0, 6.0),
    Rule("river_width", "world", "A river's width", 1.5, 0.5, 4.0),
    Rule("obstacle_size_low", "world", "An obstacle's size: at least", 1.0, 0.3, 10.0),
    Rule("obstacle_size_high", "world", "An obstacle's size: at most", 4.0, 0.3, 10.0),
    Rule("trees", "world", "Trees a new world gets", 92.0, 0.0, 400.0),
    Rule("lone_trees", "world", "Lone trees a new world gets anywhere on its land, besides its forests", 40.0, 0.0, 200.0),
    Rule("armour_strength", "body", "A hit takes × (1 - armour × this) from an armoured body", 0.7, 0.0, 1.0),
    Rule("armour_slow", "body", "Armour slows: speed × (1 - armour × this)", 0.3, 0.0, 1.0),
    Rule("stealth_strength", "body", "A stealthy body is seen and heard from × (1 - stealth × this)", 0.7, 0.0, 1.0),
    Rule("land_penalty", "body", "A swimmer walks slower on land: × (1 - water × this)", 0.7, 0.0, 1.0),
    Rule("swim_saving", "body", "In water a swimmer pays less to move and live: × (1 - water × this)", 0.5, 0.0, 1.0),
    Rule("land_cost", "body", "On land a swimmer pays more to move and live: × (1 + water × this)", 0.5, 0.0, 2.0),
    Rule("pond_food_share", "world", "Share of new food that grows in ponds", 0.2, 0.0, 1.0),
    Rule("year_steps", "world", "A year lasts this many steps (food rises and falls over it)", 20000.0, 1000.0, 200000.0),
    Rule("season_depth", "world", "Seasons: food and fruit × (1 ± this) at the year's best and worst", 0.5, 0.0, 1.0),
    Rule("forests", "world", "Trees of a new world grow in this many forests (0: anywhere)", 4.0, 0.0, 20.0),
    Rule("forest_spread", "world", "A forest's trees stand within this of its centre", 12.0, 1.0, 40.0),
    Rule("pond_edge_food", "world", "Food grows this many times as thick near a pond's edge", 2.0, 1.0, 10.0),
    Rule("pond_edge_width", "world", "How far from a pond its rich edge reaches", 2.0, 0.1, 10.0),
    Rule("tree_canopy", "world", "A tree's crown: its radius", 1.5, 0.3, 5.0),
    Rule("fruit_spread", "world", "Fruit falls this far from its tree at most", 2.0, 0.5, 10.0),
    Rule("hide_reach", "senses", "Under a tree, a creature is seen only from this close", 1.0, 0.0, 10.0),
    Rule("bite", "body", "A bite of food: the most it eats in one step", 1.0, 0.1, 20.0),
    Rule("learn_rollout", "learning", "Steps of its own life between two learning rounds", 64.0, 8.0, 1024.0),
    Rule("learn_passes", "learning", "Passes over those steps in a round", 4.0, 1.0, 20.0),
    Rule("learn_clip", "learning", "How far one round may move its choices (share)", 0.2, 0.01, 1.0),
    Rule("learn_discount", "learning", "How much a reward one step later counts", 0.99, 0.5, 0.999),
    Rule("learn_credit", "learning", "How far back a reward is credited to earlier steps", 0.95, 0.0, 1.0),
    Rule("learn_explore", "learning", "How much it is rewarded for trying different things", 0.01, 0.0, 0.5),
    Rule("learn_guess", "learning", "How much learning to guess what a place is worth counts", 0.5, 0.0, 5.0),
    Rule("learn_spread", "learning", "How spread a newborn's moves are at first", 0.5, 0.01, 2.0),
    Rule("auto_per_creatures", "auto", "One nudge per this many creatures away from its target", 8.0, 1.0, 80.0),
    Rule("auto_per_alive_rate", "auto", "One nudge per this share a 1,000 steps the number alive moves", 0.10, 0.01, 1.0),
    Rule("auto_per_energy_rate", "auto", "One nudge per this share of a full tank a 1,000 steps energy falls", 0.05, 0.01, 1.0),
    Rule("auto_max_nudges", "auto", "The most nudges one look may make", 4.0, 1.0, 20.0),
    Rule("auto_measure", "auto", "It measures the rates over this many last steps", 1000.0, 250.0, 10000.0),
    Rule("auto_nudge", "auto", "One nudge changes a knob by this share (changing it rescales the difficulty number)", 0.05,
         0.01, 0.3),
    Rule("auto_least_nudge", "auto", "One nudge changes a knob by at least this share of its whole range", 0.01, 0.0, 0.2),
    Rule("exam_every", "exam", "The exam every this many steps (changing the exam makes older scores incomparable)", 2000.0,
         250.0, 20000.0),
    Rule("exam_radius", "exam", "The exam planet's radius", 10.9, 3.0, 100.0),
    Rule("exam_brains", "exam", "Creatures taking each exam", 20.0, 2.0, 80.0),
    Rule("exam_steps", "exam", "Steps an exam lasts", 500.0, 50.0, 5000.0),
    Rule("exam_max_food", "exam", "The exam world: food pieces at most", 80.0, 5.0, 400.0),
    Rule("exam_food_energy_low", "exam", "The exam world: a food's energy at least", 10.0, 1.0, 100.0),
    Rule("exam_food_energy_high", "exam", "The exam world: a food's energy at most", 25.0, 1.0, 100.0),
    Rule("exam_food_per_step", "exam", "The exam world: new food a step", 3.0, 0.1, 20.0),
    Rule("exam_move_cost", "exam", "The exam world: cost of moving, per unit", 1.0, 0.0, 5.0),
    Rule("exam_step_cost", "exam", "The exam world: cost of being alive, each step", 0.10, 0.0, 2.0),
    Rule("exam_start_share", "exam", "The exam: creatures start with this share of a full tank", 0.5, 0.05, 1.0),
    Rule("save_every", "runner", "The world saves itself every this many steps", 2000.0, 250.0, 20000.0),
    Rule("row_every", "runner", "A row of numbers (a point on the charts) every this many steps", 250.0, 50.0, 5000.0),
    Rule("control_every", "runner", "It looks at the page's changes every this many steps", 25.0, 1.0, 500.0),
    Rule("now_seconds", "runner", "Its live numbers for the tiles at most every this many seconds", 0.5, 0.1, 10.0),
    Rule("auto_target", "auto", "Auto aims at this many alive", 480.0, 10.0, 600.0),
    Rule("auto_hunters_at", "auto", "Auto switches hunters on at this difficulty", 24.0, 0.0, 200.0),
    Rule("auto_every", "auto", "Auto looks every this many steps", 500.0, 100.0, 5000.0),
) + _gene("size", "Size", 0.5, 2.0, (0.1, 5.0), new=(0.5, 2.0)) \
  + _gene("vision_width", "Vision width (degrees)", 30.0, 120.0, (1.0, 120.0), new=(30.0, 120.0)) \
  + _gene("vision_reach", "Vision reach", 1.0, 10.0, (0.1, 20.0), new=(1.0, 10.0)) \
  + _gene("hearing", "Hearing", 0.0, 4.0, (0.0, 20.0), new=(0.0, 4.0)) \
  + _gene("fear", "Fear of death", 0.1, 100.0, (0.0, 1000.0), new=(0.1, 30.0)) \
  + _gene("love", "Love of babies", 0.1, 100.0, (0.0, 1000.0), new=(0.1, 30.0)) \
  + _gene("learns", "Learning from others", 0.01, 1.0, (0.0, 1.0), new=(0.01, 1.0)) \
  + _gene("learning_speed", "Learning speed", 1e-5, 1e-2, (1e-6, 0.1), new=(1e-4, 1e-3)) \
  + _gene("mutations", "Genes a baby changes", 1.0, 13.0, (1.0, 13.0), new=(1.0, 3.0)) \
  + _gene("mutation_size", "How much a changed gene changes (share)", 0.01, 0.2, (0.0, 1.0), new=(0.01, 0.1)) \
  + _gene("neurons", "Neurons in a layer", 1.0, 64.0, (1.0, 64.0), new=(8.0, 64.0)) \
  + _gene("layers", "Layers", 1.0, 3.0, (1.0, 3.0), new=(1.0, 3.0)) \
  + _gene("diet", "Diet (0 plants, 1 meat)", 0.0, 1.0, (0.0, 1.0), new=(0.0, 1.0)) \
  + _gene("calls", "Calls it can make", 0.0, 10.0, (0.0, 10.0), new=(0.0, 0.0)) \
  + _gene("kin", "Care for kin", 0.0, 1.0, (0.0, 1.0), new=(0.0, 1.0)) \
  + _gene("water", "Water (0 land, 1 swimmer)", 0.0, 1.0, (0.0, 1.0), new=(0.0, 1.0)) \
  + _gene("stealth", "Stealth", 0.0, 1.0, (0.0, 1.0), new=(0.0, 1.0)) \
  + _gene("armour", "Armour", 0.0, 1.0, (0.0, 1.0), new=(0.0, 1.0))

RULES += tuple(Rule(name, "costs", words, value / 10.0, 0.0, value) for name, value, words in COSTS)
RULES += tuple(Rule(f"knob.{name}.{end}", "knobs", f"{words}: its {end} end", start, 0.0, 10.0 * max(abs(easy), abs(hard)))
               for name, easy, hard, words in KNOB_ENDS for end, start in (("easy", easy), ("hard", hard)))

RULES += (
    Rule("gene_min_change", "genes", "A changed gene moves by at least up to this share of its range", 0.01, 0.0, 0.2),
    Rule("gene.weights.share", "genes", "Brain weights: the share that changes when they are a baby's changed gene", 0.02,
         0.0, 1.0),
    Rule("gene.weights.size", "genes", "Brain weights: how much a changed weight moves", 0.02, 0.0, 1.0),
)

GROUPS = (("costs", "Costs"), ("knobs", "Knob ends"), ("world", "World"), ("body", "Body: energy, size and speed"), ("senses", "Eating and hunting"),
          ("birth", "Birth and death"), ("genes", "Genes"), ("learning", "Learning"), ("auto", "Auto"), ("exam", "Exam"),
          ("runner", "Runner"))

_BY_NAME = {rule.name: rule for rule in RULES}
ROOM_SLOTS = int(_BY_NAME["room"].high)  # a learner keeps a slot for each creature there may ever be room for


# What each rule means, in plain words (the "What it means" column of the Rules page).
MEANS = {
    "step_cost": "Energy a creature pays every step just for being alive (× its size to the living power). 0.1: a "
                 "full size-1 creature standing still lasts about 1,000 steps without food.",
    "move_cost": "Energy it pays for each unit it moves, × its size. 0.4: a cruising step costs about a step of living.",
    "sprint_power": "Moving faster than cruising costs the move × (speed ÷ cruising) to this power.",
    "eat_miss": "Energy it pays when it bites at nothing.",
    "bump_cost": "Energy it pays when it runs into a rock.",
    "attack_cost": "Energy an attack costs the attacker, hit or miss, × its size.",
    "attack_damage": "Energy a hit takes from the one hit, × the hitter's size (more for a meat eater, less for armour).",
    "vision_cost": "Energy its eyes cost every step, per square unit of its vision cone.",
    "hearing_cost": "Energy its ears cost every step, per square unit of its hearing circle.",
    "neuron_cost": "Energy each neuron of its brain costs every step. 0.0006: 32 neurons cost about a fifth of living.",
    "stealth_cost": "How much more living costs a stealthy creature: × (1 + stealth × this).",
    "armour_cost": "How much more living costs an armoured creature: × (1 + armour × this).",
    "baby_share": "When it has a baby, this share of its energy goes into the baby (0.4 = 40 %).",
    "tank_power": "How much more energy a bigger body holds: its tank is the tank × size to this power (1.3: twice the "
                  "size holds about 2.5 times the energy).",
    "living_power": "How much more a bigger body pays to stay alive: the living cost × size to this power (0.75: twice "
                    "the size pays about 1.7 times).",
    "speed_power": "How much slower a bigger body cruises: the cruising speed ÷ size to this power (0.5: twice the size "
                   "is about 0.7 times as fast).",
    "lifespan_power": "How much longer a bigger body lives: its lifespan × size to this power (0.5: twice the size lives "
                      "about 1.4 times as long).",
    "heard_power": "How far a moving body is heard: the listener's hearing × the mover's size to this power (1: half "
                   "the size is heard from half as far).",
    "seen_power": "How far a body is seen: the looker's reach × the body's size to this power (0.5: half the size is "
                  "seen from about 0.7 as far).",
    "armour_strength": "How much armour turns a hit: a hit takes × (1 - armour × this). 0.7: full armour takes 30 % "
                       "of a hit.",
    "armour_slow": "How much armour slows: speed × (1 - armour × this). 0.3: full armour moves at 70 %.",
    "stealth_strength": "How much stealth hides: a body of stealth 1 is seen and heard from (1 - this) as far. 0.7: "
                        "from 30 % as far.",
    "forests": "How many forests a new world's trees grow in, with plains between; 0 scatters them anywhere.",
    "forest_spread": "How far from its forest's centre a tree may stand: a small number makes a dense wood.",
    "pond_edge_food": "How much thicker food grows on the land near a pond's edge than on open ground (2 = twice "
                      "as thick).",
    "pond_edge_width": "How far out from a pond's edge the rich ground reaches.",
    "water_share": "How much of a new planet is water: seas, lakes and rivers (0.5: half). Only swimmers go in.",
    "islands": "How many islands a new planet gets, out in open water, cut off from the other land.",
    "island_gap": "How far an island lies from any other land at least: walkers can never reach it.",
    "rivers": "How many rivers a new planet gets, each running from inland down to the sea or a lake.",
    "river_crossings": "How many shallow spots each river has, where walkers can cross it.",
    "river_width": "How wide a river is.",
    "exam_radius": "How big the exam's own planet is (10.9: the ground of the old 50 × 30 exam; a new world's "
                   "size never changes the exam).",
    "year_steps": "How many steps a year lasts: food and fruit grow most a quarter into it and least three "
                  "quarters into it.",
    "season_depth": "How strong the seasons are: food and fruit grow × (1 + this × sin(2π × step ÷ year)). 0.5: "
                    "half again as much in the best season, half as much in the worst; 0: no seasons.",
    "swim_saving": "How much less a swimmer pays in water: its moving and living cost × (1 - its water × this). 0.5: "
                   "a pure swimmer pays half in water.",
    "land_cost": "How much more a swimmer pays on land: its moving and living cost × (1 + its water × this). 0.5: a "
                 "pure swimmer pays 1.5 times on land.",
    "land_penalty": "How much slower a swimmer walks on land: its land speed × (1 - its water × this). 0.7: a "
                    "full swimmer walks at 30 % of its speed.",
    "pond_food_share": "The share of new food that grows inside ponds, where only swimmers reach it.",
    "call_reach": "How far a call carries at full volume; a quieter call carries volume × this. Everyone within it "
                  "hears it, hunters too.",
    "diet_bite": "How much harder a meat eater hits: a hit takes × (1 + its diet × this). 1: a pure meat eater hits "
                 "twice as hard as a plant eater of its size.",
    "planet_radius": "How big a new world's planet is: its radius. Walking all the way round takes 2π × this "
                     "(37: about 232). A running world keeps its own.",
    "room": "The most creatures that can live at once. While it is full, nobody has a baby.",
    "soup": "While fewer than this many are alive, random new creatures appear, so the world never dies out.",
    "newcomer_energy_low": "A random newcomer starts with at least this much energy.",
    "newcomer_energy_high": "A random newcomer starts with at most this much energy.",
    "tank": "The most energy a creature can hold. A bigger creature holds more.",
    "body_radius": "How big its body is on the map. A bigger creature is wider.",
    "cruise_speed": "How far it moves in one step at its normal speed. A smaller creature is faster.",
    "sprint_speed": "The fastest it can go in a burst. A bigger creature sprints faster, but sprinting costs a lot.",
    "birth_at": "It has a baby when its energy reaches this share of its tank (0.7 = 70 % full).",
    "low_energy": "Once it is older than its lifespan, it dies when its energy falls below this share of its tank.",
    "reach": "How close food or another creature must be to eat it or hit it: this × (its radius + the other's). "
             "1 means touching.",
    "hunter_share": "Of the energy a hit takes from the prey, the hunter keeps this share (0.8 = 80 %).",
    "corpse_energy": "When a creature dies, it leaves food worth this × its size.",
    "corpse_steps": "How many steps a dead creature's food stays before it is gone.",
    "first_creatures": "How many creatures a brand-new world starts with.",
    "obstacles": "How many rocks a new world gets. Nothing can enter them, nothing grows on them, nobody sees "
                 "through them.",
    "obstacle_size_low": "The smallest an obstacle can be (about its radius). Each gets a random size and shape.",
    "obstacle_size_high": "The biggest an obstacle can be (about its radius).",
    "lone_trees": "How many single trees a new world gets at random places anywhere on its land (never in water), "
                  "besides the trees of its forests.",
    "trees": "How many trees a new world gets. Each drops fruit now and then; under its crown a creature hides.",
    "tree_canopy": "How wide a tree's crown is (its radius): the area where a creature hides.",
    "fruit_spread": "How far from its trunk a tree's fruit can fall.",
    "hide_reach": "A creature under a tree's crown is seen only by those this close; hunters can wait there.",
    "bite": "The most food energy it can eat in one step.",
    "learn_rollout": "It learns from its own life every this many steps.",
    "learn_passes": "How many times it goes over those steps when it learns.",
    "learn_clip": "The most one learning round may change how it behaves (0.2 = 20 %).",
    "learn_discount": "How much it cares about later rewards: at 0.99 a reward 100 steps away still counts about a "
                      "third.",
    "learn_credit": "How far back a reward is credited to the moves that led to it (1 = all the way back).",
    "learn_explore": "How much it is pushed to try new things instead of repeating what worked.",
    "learn_guess": "How much it practises guessing how good its situation is (that guess helps it learn).",
    "learn_spread": "How random a newborn's moves are before it has learned anything.",
    "auto_target": "Auto makes the world harder or easier to keep about this many creatures alive.",
    "auto_hunters_at": "Hunting is switched on once Auto's difficulty number reaches this.",
    "auto_every": "Auto checks the world every this many steps.",
    "auto_per_creatures": "Auto moves one nudge harder or easier for every this many creatures above or below its "
                          "target.",
    "auto_per_alive_rate": "Auto adds one nudge for every this share the number alive grows or shrinks per 1,000 "
                           "steps.",
    "auto_per_energy_rate": "Auto adds one nudge for every this share of a full tank the average energy falls per "
                            "1,000 steps.",
    "auto_max_nudges": "The most nudges Auto makes in one check.",
    "auto_measure": "Auto judges how things are going over this many last steps.",
    "auto_nudge": "How much one nudge changes the costs and the food (0.05 = 5 %).",
    "auto_least_nudge": "The least one nudge changes a knob, as a share of its range from easiest to hardest (0.01 = 1 %): a knob at 0 still moves.",
    "exam_every": "How often some creatures take a test in a fixed world, so scores compare over time.",
    "exam_brains": "How many creatures take each test.",
    "exam_steps": "How many steps one test lasts.",
    "exam_max_food": "The test world: the most food pieces at once.",
    "exam_food_energy_low": "The test world: a food piece is worth at least this.",
    "exam_food_energy_high": "The test world: a food piece is worth at most this.",
    "exam_food_per_step": "The test world: new food pieces each step.",
    "exam_move_cost": "The test world: energy moving costs, per unit of distance.",
    "exam_step_cost": "The test world: energy being alive costs, each step.",
    "exam_start_share": "In the test, creatures start with this share of a full tank.",
    "save_every": "The world saves itself to disk every this many steps.",
    "row_every": "One row of numbers (one point on the charts) every this many steps.",
    "control_every": "The world checks this page for your changes every this many steps.",
    "now_seconds": "How often the live tiles update, in seconds.",
    "gene.weights.share": "When the gene a baby changes is its brain weights, this share of the weights changes "
                          "(0.02 = 2 %).",
    "gene.weights.size": "How much each changed weight moves, up or down.",
    "gene_min_change": "A baby's changed gene moves by up to its parent's 'how much each changes' share of the gene's "
                       "range (lowest to highest), and never by less than this share of it (0.01 = 1 %).",
}

# What each gene is, in plain words (the gene table's "What it means" column).
GENE_MEANS = {
    "size": "How big its body is. Bigger holds more energy, hits harder and sprints faster, but costs more to live and "
            "move.",
    "vision_width": "How wide its vision cone is, in degrees.",
    "vision_reach": "How far it sees.",
    "hearing": "How far it hears, all around. Sound tells where and how far, never how big.",
    "fear": "How much its own death counts as a loss when it learns.",
    "love": "How much having a baby counts as a reward when it learns.",
    "learns": "How much it learns from what it sees others go through (1 = as much as from its own life).",
    "learning_speed": "How fast its brain changes when it learns.",
    "mutations": "How many of its genes each of its babies changes (the rest are copied exactly). This one moves by one.",
    "mutation_size": "How much each changed gene of its babies changes: up to this share of the gene's range, either "
                     "way (0.05 = 5 %).",
    "neurons": "How many neurons each brain layer has. Each neuron costs energy every step.",
    "layers": "How many layers its brain has.",
    "armour": "How well its body takes a hit: a hit takes × (1 - armour × the armour strength) from it; it moves "
              "slower (× 1 - armour × the armour slowness) and pays more to stay alive (× 1 + armour × the armour "
              "cost).",
    "stealth": "How hard it is to notice: others see and hear it only from (1 - stealth × the stealth strength) as "
               "far; it pays more to stay alive (× 1 + stealth × the stealth cost).",
    "water": "How much it is at home in water: 0 cannot enter a pond; above 0 it can, and swims at this share of "
             "its speed, but walks slower on land (× 1 - water × the land penalty); it pays less to move and live in "
             "water (× 1 - water × the swim saving) and more on land (× 1 + water × the land cost).",
    "kin": "How much what happens to creatures like it counts as its own: when one it sees or hears gains or "
           "loses energy, it feels this share of it, times how alike their genes are (0 = not at all, 1 = twins).",
    "calls": "How many different calls it can make (0 to 10). A new world's creatures make none; a baby may get one "
             "more or one fewer. No call means anything until evolution gives it a meaning.",
    "diet": "What it lives on: 0 = plants and fruit only, 1 = meat only (corpses and hits). In between it gets that "
            "share of each: 0.3 gets 70 % of a plant bite and 30 % of a meat bite. A meat eater also hits harder.",
}

_PART_MEANS = {"low": "No creature's {g} goes below this.", "high": "No creature's {g} goes above this.",
               "new_low": "A random newcomer gets at least this.",
               "new_high": "A random newcomer gets at most this.",
               "easy": "The easiest the {g} knob goes: Auto and its slider stop here.",
               "hard": "The hardest the {g} knob goes: Auto and its slider stop here."}


def means(name: str) -> str:
    """What rule `name` means, in plain words."""
    if name in MEANS:
        return MEANS[name]
    part = name.split(".")[2]
    return _PART_MEANS[part].format(g=_BY_NAME[name].words.split(":")[0].lower())


def rule(name: str) -> Rule:
    return _BY_NAME[name]


def starting_rules() -> Rules:
    """A new world's rules."""
    return {r.name: r.start for r in RULES}


def check(rules: Rules, name: str, value: float) -> Optional[str]:
    """Why `value` cannot be rule `name` in a world with `rules` (outside its range, or a gene's lowest above its
    highest), in plain words; None when it can."""
    found = _BY_NAME.get(name)
    if found is None:
        return f"{name} is an unknown rule"
    if value < found.low:
        return f"{found.words} must be at least {found.low:g}"
    if value > found.high:
        return f"{found.words} can be at most {found.high:g}"
    for low, high in ((".low", ".high"), (".new_low", ".new_high")):
        for mine, other, below in ((low, high, True), (high, low, False)):
            if name.startswith("gene.") and name.endswith(mine):
                partner = rules.get(name[:-len(mine)] + other)
                if partner is not None and (value > partner if below else value < partner):
                    return f"{found.words.split(':')[0]}: the lowest must not be above the highest"
    if name.startswith("knob."):  # an easy end stays on its easy side of the hard end
        base, end = name.rsplit(".", 1)
        partner = rules.get(f"{base}.{'hard' if end == 'easy' else 'easy'}")
        more_is_easier = _BY_NAME[f"{base}.easy"].start > _BY_NAME[f"{base}.hard"].start
        easy, hard = (value, partner) if end == "easy" else (partner, value)
        if partner is not None and (easy <= hard if more_is_easier else easy >= hard):
            side = "above" if more_is_easier else "below"
            return f"{found.words.split(':')[0]}: the easy end must stay {side} the hard end"
    return None


# The genes the owner may change on a picked creature: its attribute and the rules' gene name (its limits).
EDITABLE_GENES = {"size": "size", "vision_width": "vision_width", "vision_reach": "vision_reach", "hearing": "hearing",
                  "fear_of_death": "fear", "love_of_babies": "love", "learns_from_others": "learns",
                  "learning_speed": "learning_speed", "mutations": "mutations", "mutation_size": "mutation_size",
                  "diet": "diet", "calls": "calls", "kin": "kin", "water": "water", "stealth": "stealth", "armour": "armour"}


def check_genes(rules: Rules, genes: object) -> Optional[str]:
    """Why `genes` cannot be given to a creature (an unknown gene, not a number, outside its limits, a brain whose
    neuron counts are not one for each of its layers), in plain words; None when they can."""
    if not isinstance(genes, dict) or not genes:
        return "genes must name at least one gene and its new value"
    for name, value in genes.items():
        if name in ("layers", "neurons"):
            continue
        if name not in EDITABLE_GENES:
            return f"{name} is not a gene that can be changed"
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"{name} must be a number, not {value!r}"
        if name in ("mutations", "calls") and value != int(value):
            return f"{name} must be a whole number"
        low, high = gene_limits(rules, EDITABLE_GENES[name])
        if not low <= value <= high:
            return f"{_BY_NAME[f'gene.{EDITABLE_GENES[name]}.low'].words.split(':')[0]} must be between {low:g} and {high:g}"
    if ("layers" in genes) != ("neurons" in genes):
        return "a brain change names both its layers and its neurons in each layer"
    if "layers" in genes:
        layers, neurons = genes["layers"], genes["neurons"]
        low, high = gene_limits(rules, "layers")
        if isinstance(layers, bool) or not isinstance(layers, int) or not low <= layers <= high:
            return f"layers must be a whole number between {low:g} and {high:g}"
        if not isinstance(neurons, list) or len(neurons) != layers:
            return "neurons must give one neuron count for each layer"
        low, high = gene_limits(rules, "neurons")
        if any(isinstance(n, bool) or not isinstance(n, int) or not low <= n <= high for n in neurons):
            return f"each layer's neurons must be a whole number between {low:g} and {high:g}"
    return None


def gene_limits(rules: Rules, gene: str) -> tuple:
    return rules[f"gene.{gene}.low"], rules[f"gene.{gene}.high"]
