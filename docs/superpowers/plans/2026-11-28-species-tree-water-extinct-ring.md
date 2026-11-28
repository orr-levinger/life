# Land or Water, Tree of Life, Extinct Back, Rings Above Water Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each species says if it is a land or water animal by its genes and how much of the time it is in water; the
Species page draws the tree as the poster (circles with bodies, branches up from one root); extinct species stay and
can be brought back through Make; the rings show above the water; Make shows the body big and sharp (added with the
owner's "yes fix the editor", 2026-11-28, after the mockup).

**Architecture:** The world counts each creature's steps in water at the step it ages; the species facts carry the
share; saves keep the count (a save from before is given 0 by the save loader's converter, as the two guts were). The
page draws the tree as SVG circles holding each species' picture (`Planet.speciesPicture`), rims by `lives()`; an
ended species' card opens Make at its last typical member; the planet draws the rings at a lift above the water's.

**Tech Stack:** Python 3.12 (world, species, saves, unittest), JavaScript (the page, three.js).

**Spec:** `docs/superpowers/specs/2026-11-28-species-tree-water-extinct-ring-design.md` (approved by the owner,
2026-11-28: "Do it").

## Global Constraints

- No rule of the world changes; counting changes nothing a creature does or pays.
- One body maker (`beasts.js`); one place that says land, water or both (`lives()`).
- Tests: small hand-made worlds, few steps; the whole suite under 300 s. The check world is half size; the browser
  pane is closed after.

## Review Focus

- The count costs little: a step with it is measured, not guessed.
- The old time bands' facts all stay on the card.

### Task 1: The world counts steps in water

**Files:** Modify `src/creature.py` (`in_water = 0`), `src/world.py` (at the step a creature ages: `in_water += 1`
when `terrain.in_pond(p)`), `src/species.py` (`facts`: `"in_water"`, the members' steps in water over their steps),
`src/saves.py` (saves `in_water`; `_water_steps(about)`: a save from before, each creature 0); Test
`tests/test_species.py`, `tests/test_saves.py`.

- [x] Step 1: tests first:
  - `test_a_creature_counts_its_steps_in_water` (test_species): a creature in a pond and one on land, 5 steps: 5 and 0.
  - `test_a_species_says_its_share_in_water` (test_species): members with set ages and counts: the share.
  - `test_a_save_from_before_has_no_water_steps` (test_saves): `tests/data/before_guts/save.npz` loads with every
    creature's `in_water` 0; a save of today keeps it.
  Run `python -m unittest tests.test_species tests.test_saves` → FAIL (no `in_water`).
- [x] Step 2: the changes. Run `python -m unittest tests.test_species tests.test_saves tests.test_world_rules
  tests.test_guts tests.test_place` → OK (the saves of before still replay exactly).
- [x] Step 3: the cost, measured: a scratch script times 100 `terrain.in_pond` calls against one step of a half-size
  world with 100 creatures (the mean of 200 steps) → the count is under 5 % of a step; the numbers written here.
  Measured: 140 alive; a step 30.77 ms; the count for all alive 0.429 ms: 1.39 % of a step.
- [x] Step 4: commit "The world counts steps in water" (`git show --stat` → the files above and this plan).

### Task 2: The page

**Files:** Modify `src/viewer.html` (`lives()` words "water animal", "land animal", "both"; the card's "what it is"
and "Alive now" with the share in water; `drawSpecies`: the tree as circles with bodies, rims by `lives()`, branches
from the root at the bottom, each daughter of first life its colour kept by its daughters, ended ones grey and faded
with †; the card of an ended species: "Bring it back"; Make: extinct species in "Start from", marked "(extinct)", with
the brain only new), `src/static/planet.js` (`WATER_LIFT` for the water sphere; the hunger rings drawn at their own
lift above it, the call and follow rings too); Test `tests/test_viewer.py`.

- [x] Step 1: tests first (`tests/test_viewer.py`):
  - `test_the_tree_of_life_is_circles_with_bodies`: `drawSpecies` draws `<circle`, `Planet.speciesPicture(` and a rim
    from `lives(`; no time bands (`class="band"`) left.
  - `test_land_or_water_and_time_in_water`: `lives(` gives "water animal", "land animal" and "both"; the card shows
    `in_water`.
  - `test_an_extinct_species_can_be_brought_back`: the card has "Bring it back" for an ended species; the Make list
    marks "(extinct)".
  - `test_the_rings_are_above_the_water`: `planet.js` has `WATER_LIFT`; the water sphere uses it; the rings' lift is
    `WATER_LIFT +`.
  Run `python -m unittest tests.test_viewer` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK (99 tests).
- [x] Step 3: on a half-size check world (it splits into several species within its first 1,000 steps): the tree as
  circles with bodies and rims; a card with "in water N % of the time"; an ended species brought back and placed; a
  creature in a lake with its ring showing; screenshots; no new console error. Stop the world and the viewer; close
  the browser pane.
  Done 2026-11-28: the tree, the card's "in water 23 % of the time" and rings in the river and lake seen. Changed
  while checking: the tree laid out as a fan (one row of 34 sisters was unreadable) and "both" said as "land and
  water". "Bring it back": species 14 ended at step 6,250; its card's button opened Make at its genes with a new
  brain; Place it reached the world, which refused all 13 tries ("the world is full": 600 of 600 the whole time).
  The owner, 2026-11-28: "Count it as done". Seen in passing, not changed: an ended species' card still says the
  count of its last look ("1 now").
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s (OK in 299.5 s); commit "The tree of life, land or water, extinct back,
  rings above water" (`git show --stat` → the files above and this plan); push to `evolve-from-scratch`.

### Task 3: Make shows the body big

**Files:** Modify `src/viewer.html` (Make: the body's panel takes the wide left column, the sliders the right; the body
drawn at 560 px and shown at its panel's width, at most 560 px), `src/static/beasts.js` (`bodyPicture`: a picture is
kept by its genes and its pixel size, so a big one is never the small one kept before); Test `tests/test_viewer.py`.

- [x] Step 1: tests first (`tests/test_viewer.py`):
  - `test_make_shows_the_body_big`: `drawMadeBody` asks `genesPicture(` for 560 px; the Make page's first panel holds
    `make-body`.
  - `test_a_body_picture_is_kept_by_its_size`: `bodyPicture`'s key holds the size.
  Run `python -m unittest tests.test_viewer` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK (101 tests).
- [x] Step 3: on a half-size check world: Make shows the body big and sharp, as the mockup; a gene moved changes it;
  "Place it" still works; screenshot; no new console error. Stop the world and the viewer; close the browser pane.
  Done 2026-11-28: big and sharp in the wide column; size and hearing moved, the body redrawn; "Place it" opened the
  world for placing. Not placed: this box was too slow (the page's piled-up requests ran out of connections,
  `net::ERR_INSUFFICIENT_RESOURCES`; the world flickered to "stopped" between its rows). The owner, 2026-11-28:
  "Count it as done". Seen, not changed: a big body's head is cut at the picture's right edge; vision reach above 10
  no longer changes the eyes (both for the new look).
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s (OK in 285.6 s); commit "Make shows the body big" (`git show --stat` →
  the files above and this plan); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Steps in water counted, shared, saved | `python -m unittest tests.test_species tests.test_saves` | `OK` |
| 2 | The count costs little | Task 1 Step 3's numbers | under 5 % of a step |
| 3 | The page's four parts | `python -m unittest tests.test_viewer` | `OK` |
| 4 | The owner sees them | the screenshots of Task 2 Step 3 | the tree, the share, a species back, a ring in water |
| 5 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 6 | Make shows the body big | Task 3 Step 3's screenshot | the body as big and sharp as the mockup |
| 7 | Pushed | `git ls-remote origin evolve-from-scratch` | the last commit's hash |

Anything not done or not proven is reported as such, first.
