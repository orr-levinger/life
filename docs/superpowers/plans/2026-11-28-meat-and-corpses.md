# Meat and Corpses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Meat gives twice the energy of grass and fruit; a corpse holds 40 × size and lasts 100 × size steps; a corpse
looks like a lying body; food is drawn smaller as it is eaten.

**Architecture:** A new rule `meat_value` in `src/rules.py`; `Creature._eat` and the hit's gain multiply meat by it;
`World.kill` makes the corpse with `corpse_energy × size` and `corpse_steps × size`; the saves converter gives a save
from before the new values; `Food.left` (the share of its energy left, from its radius as made) goes into the view
file; `src/static/planet.js` draws meat as a lying body and every food by `left`.

**Tech Stack:** Python 3.12 (unittest), JavaScript (three.js).

**Spec:** `docs/superpowers/specs/2026-11-28-meat-and-corpses-design.md`.

## Global Constraints

- The world rule changes are the owner's (quoted in the spec): `meat_value` 2, `corpse_energy` 40, `corpse_steps` 100
  × size. Nothing else a creature does or pays changes.
- One place for a food's size by energy: `src/food.py` (`RADIUS_PER_ENERGY`).
- Tests: small hand-made worlds, few steps; the whole suite under 300 s. The check world is half size.

## Review Focus

- A save from before goes on with the new values and with its lying corpses' steps as they were.
- The hit's gain and the bite's gain use the same `meat_value`.

### Task 1: Meat and corpses

**Files:** Modify `src/rules.py` (`Rule("meat_value", "senses", "A bite or hit of meat gives this × what a bite of
plants gives", 2.0, 1.0, 5.0)` and its words; `corpse_energy` 40; `corpse_steps` 100 with words "× the dead one's
size"), `src/creature.py` (`_eat`: `amount × meat_gut × meat_value` for meat; the hit:
`hunter_share × taken × meat_gut × meat_value`), `src/world.py` (`kill`: `remaining_duration=round(corpse_steps ×
size)`), `src/food.py` (`RADIUS_PER_ENERGY = 0.01`; `left`: the energy over the energy it was made with, from its
radius), `src/saves.py` (`_meat_and_corpses(about)`: a save without `meat_value` gets `meat_value` 2, `corpse_energy`
40, `corpse_steps` 100), `src/evolve.py` (each food in the view: `"left"`), `src/static/planet.js` (`FOOD_KINDS.meat`:
a lying body, flattened grey-brown with pale ribs; every food scaled by `Math.sqrt(f.left ?? 1)`). Tests:
`tests/test_world_rules.py`, `tests/test_saves.py`, `tests/test_evolve_runs.py`, `tests/test_viewer.py`.

- [ ] Step 1: tests first:
  - `test_meat_is_worth_twice` (test_world_rules): two creatures of meat gut and grass gut 0.5 each, one eating a
    corpse, one a grass piece, one bite: the meat eater gains twice.
  - `test_a_hit_is_worth_meat_value` (test_world_rules): a hit's gain is `0.8 × taken × meat gut × 2`.
  - `test_a_corpse_lasts_by_its_size` (test_world_rules): a size-1 creature killed leaves 40 energy that is there after
    99 steps and gone after 101; a size-0.5 one leaves 20, gone after 51.
  - `test_a_save_from_before_gets_the_new_meat` (test_saves): `tests/data/before_guts/save.npz` loads with
    `meat_value` 2, `corpse_energy` 40, `corpse_steps` 100.
  - `test_the_view_says_how_much_food_is_left` (test_evolve_runs): a food in the view has `"left"`; half eaten, 0.5.
  - `test_corpses_look_like_bodies_and_food_shrinks` (test_viewer): `planet.js` draws each food scaled by
    `Math.sqrt(f.left`; the meat kind is not the plant's or fruit's shape.
  Run `python -m unittest tests.test_world_rules tests.test_saves tests.test_viewer` and
  `python -m unittest tests.test_evolve_runs` → FAIL.
- [ ] Step 2: the changes. Run the same commands → OK.
- [ ] Step 3: on a half-size check world with hunters on: a corpse seen lying where a creature died, unlike food, still
  there many steps later; a piece being eaten drawn smaller; screenshot; no new console error. Stop the world and its
  viewer; close the browser pane.
- [ ] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "Meat worth more, corpses that stay" (`git show
  --stat` → the files above and this plan); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Meat ×2, corpses by size, saves, view | Step 2's commands | `OK` |
| 2 | The owner sees corpses and shrinking food | Step 3's screenshot | a lying body; a smaller half-eaten piece |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git ls-remote origin evolve-from-scratch` | the last commit's hash |

Anything not done or not proven is reported as such, first.
