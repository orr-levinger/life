# Grass by Patch, Sea Algae, Day and Night Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grass is limited per patch of ground; water has algae beds that grow food like trees; the planet turns once
every 10,000 steps and its lit half grows three quarters of the food.

**Architecture:** `World.spawn_food` (`src/world.py`) keeps its order (grass candidates, then each tree) and gains:
a patch count before a grass piece is kept; each algae bed after the trees; the light, by the sun's direction
(`World.sun()`), weighting grass spots and each tree's and bed's rate. Algae beds are a new kind of the terrain's
(`src/terrain.py`, beside `Tree`), made by `make_terrain` and given to a save from before when loaded. The knob
`max_food` ("Grass pieces at most") becomes `grass_per_patch` ("Grass per patch"); today's yearly season rules go.
The page draws the beds and lights the planet from the sun (`src/static/planet.js`).

**Tech Stack:** Python 3.12 (unittest), JavaScript (three.js).

**Spec:** `docs/superpowers/specs/2026-11-28-food-areas-day-night-design.md`.

## Global Constraints

- The world rule changes are the owner's (quoted in the spec). What a creature does, eats or pays stays.
- One place for "where food grows": `World.spawn_food` and its helpers.
- Tests: small hand-made worlds, few steps; the whole suite under 300 s. The check world is half size.
- One commit per task.

## Review Focus

- Every new food stays seeded by the food seed (`food_rng`), so two worlds with the same seed grow the same food.
- A save from before loads: its grass cap becomes the patch knob's start, it gets algae beds, its yearly season goes.

### Task 1: Grass by the patch

**Files:** Modify `src/settings.py` (`max_food` → `grass_per_patch`, start 3), `src/rules.py` (`KNOB_ENDS`:
`("grass_per_patch", 10.0, 1.0, "Grass per patch")`; `Rule("patch", "world", "A patch of ground: grass is counted within
this distance", 3.0, 0.5, 10.0)`), `src/world.py` (`spawn_food`: a grass spot kept only while fewer than
`grass_per_patch` grass pieces lie within `patch` of it), `src/exam.py` (the exam's settings), `src/saves.py`
(`_grass_by_patch`: a save's `max_food` dropped, `grass_per_patch` 3, in its settings, Auto's and Auto's taken
nudges), `src/scoreboard.py` (the column renamed; `BEFORE_PATCHES` the layout before), `src/viewer.py` and
`src/viewer.html` (the knob's label and group), and every test using `max_food`.

- [ ] Step 1: tests first: `test_grass_stops_at_its_patch` (test_terrain: 3 per patch, a fourth spot within 3 units
  refused, one 4 units away kept); `test_an_eaten_patch_grows_again` (test_terrain); `test_a_save_from_before_gets_patches`
  (test_saves); the knob's ends (test_dial); the scoreboard column (test_scoreboard). Run them → FAIL.
- [ ] Step 2: the changes; every test that used `max_food` moved to `grass_per_patch`. Run `python -m unittest
  tests.test_terrain tests.test_saves tests.test_dial tests.test_scoreboard tests.test_settings tests.test_costs
  tests.test_seasons tests.test_world_rules tests.test_viewer` → OK; `grep -rn "max_food" src tests` → only the
  exam's own `exam_max_food` and the saves converter.
- [ ] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Grass by the patch".

### Task 2: Algae beds

**Files:** Modify `src/terrain.py` (`Bed`, like `Tree`, placed in water; `Terrain.beds` in its state; `make_terrain`
places `algae_beds`), `src/rules.py` (`Rule("algae_beds", "world", "Algae beds a new world gets in its water", 100.0,
0.0, 400.0)`), `src/world.py` (`spawn_food`: each bed grows algae within `fruit_spread` at the fruit's rate, up to
`max_fruit` algae; algae is a plant (`Food.algae`) in the water), `src/food.py` (`algae`), `src/saves.py` (a save
without beds: beds placed in its own water, seeded by its step), `src/evolve.py` (the view's terrain and foods carry
beds and algae), `src/static/planet.js` (beds as dark weed under the water; algae as small green pieces), `src/viewer.html`
(the world's line: "N algae"). Tests: `tests/test_terrain.py`, `tests/test_saves.py`, `tests/test_viewer.py`.

- [ ] Step 1: tests first: `test_beds_are_in_water` (test_terrain: a made planet's beds all `in_pond`);
  `test_a_bed_grows_algae_up_to_its_cap` (test_terrain); `test_algae_feeds_a_grass_gut` (test_terrain);
  `test_a_save_from_before_gets_beds` (test_saves); `test_beds_and_algae_are_drawn` (test_viewer). Run → FAIL.
- [ ] Step 2: the changes. Run the same tests and `tests.test_world_rules` → OK.
- [ ] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Algae beds in the water".

### Task 3: Day and night

**Files:** Modify `src/rules.py` (`Rule("day_steps", "world", "Steps for the planet to turn once", 10000.0, 100.0,
1000000.0)`, `Rule("day_share", "world", "Share of new food that grows on the lit half", 0.75, 0.5, 1.0)`; the yearly
`season_depth` and `year_steps` removed), `src/world.py` (`sun()`: the unit vector to the sun, turning about the
planet's axis by `2π × step / day_steps`; `lit(p)`; `spawn_food`: a grass spot in the dark kept with odds
`(1 − day_share) / day_share`, else the next spot; a tree or bed's rate × `2 × day_share` lit, `2 × (1 − day_share)`
dark; `season()` and its callers removed), `src/saves.py` (the yearly rules dropped from a save), `src/evolve.py` (the
view's `"sun"`), `src/static/planet.js` (the sunlight from the view's sun), `src/viewer.html` (the season tile, if
any, says day and night). Tests: `tests/test_seasons.py` (rewritten for day and night), `tests/test_saves.py`,
`tests/test_viewer.py`.

- [ ] Step 1: tests first: `test_the_sun_turns` (90° after 2,500 steps); `test_the_lit_half_grows_three_quarters`
  (2,000 grass spots with the sun fixed: 0.75 ± 0.03 on the lit half); `test_a_tree_in_the_dark_grows_a_third`
  (fruit over 2,000 steps: dark ÷ lit ≈ 1/3); `test_a_save_from_before_has_no_year` (test_saves);
  `test_the_planet_is_lit_from_the_sun` (test_viewer). Run → FAIL.
- [ ] Step 2: the changes. Run `tests.test_seasons tests.test_saves tests.test_viewer tests.test_terrain
  tests.test_world_rules` → OK; `grep -rn "season_depth\|year_steps" src tests` → nothing.
- [ ] Step 3: on a half-size check world: grass on the plains, algae in a sea, the night side darker and moving;
  screenshots to the owner; stop the world and its viewer; close the browser pane.
- [ ] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "Day and night"; push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Grass by patch | Task 1 Step 2's command | `OK` |
| 2 | Algae beds | Task 2 Step 2's command | `OK` |
| 3 | Day and night | Task 3 Step 2's command | `OK`; no yearly season left |
| 4 | The owner sees it | Task 3 Step 3's screenshots | plains grass, sea algae, a moving night |
| 5 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 6 | Pushed | `git ls-remote origin evolve-from-scratch` | the last commit's hash |

Anything not done or not proven is reported as such, first.
