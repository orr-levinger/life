# Eyes of Pixels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Creatures see a row of 16 pixels at an evolved sharpness and hear 8 directions.

**Spec:** `docs/superpowers/specs/2026-11-27-eyes-of-pixels-design.md`

## Global Constraints

- `N_SENSES` 84 → 196: eye 16 × 6 at 84, ears 8 × 2 at 180.
- Gene `eye_pixels` 1-16 (newcomers 2-8); cost rule `pixel_cost` (nature 0.0006, start a tenth, Auto turns it).
- Brains of before: zero weights for the new inputs (the padding of 2026-11-27 already does it).
- Checks on this box: half-size worlds, browser closed after. Tests under 300 s.

### Task 1: The gene and its cost

**Files:** Modify `src/rules.py` (`eye_pixels` gene; `pixel_cost` in `COSTS`), `src/world.py` (`GENES`: newcomers draw
it, babies inherit it), `src/creature.py` (`eye_pixels`; living pays `pixel_cost` × pixels; `SEEN_GENES` stays 18);
Test `tests/test_eyes.py` (new).

- [x] Step 1: tests first: `test_sharpness_is_a_gene`, `test_sharpness_costs_energy`. Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_eyes tests.test_costs tests.test_rules` → OK.
- [x] Step 3: commit "Eye sharpness is a gene, and costs energy".

### Task 2: The pixels and the ears

**Files:** Modify `src/brain.py` (`N_SENSES` 196), `src/senses.py` (`_eye_pixels`, `_ears` from the pairs already
measured; water by one ray per real slice); Test `tests/test_eyes.py`.

- [x] Step 1: before: the senses and the brains' choices timed on one half-size world state (the script of
  2026-11-27).
- [x] Step 2: tests first: `test_kin_food_and_a_hunter_land_in_their_pixels`, `test_a_blurry_eye_sees_blocks`,
  `test_a_rock_hides_from_the_pixels`, `test_a_mover_is_heard_from_its_direction`, `test_brains_of_before_act_the_same`.
  Run → FAIL.
- [x] Step 3: the changes. Run `python -m unittest tests.test_eyes tests.test_senses tests.test_learner tests.test_brain`
  → OK.
- [x] Step 4: after: the same timing; report both.
- [x] Step 5: `python scripts/run_tests.py` → OK under 300 s; commit "Eyes of pixels, ears of directions"; push to
  `evolve-from-scratch`.

### Task 3: The Pixels view

**Files:** Modify `src/control.py` (the followed creature's id with the watching), `src/viewer.py` (the page sends it),
`src/evolve.py` (`world_view`: the followed one's pixels, ears and sharpness), `src/viewer.html` (Above | Eyes | Pixels;
the strip and the ear ring); Test `tests/test_viewer.py`, `tests/test_evolve.py`.

- [x] Step 1: tests first: `test_the_view_has_the_followed_ones_pixels`, `test_the_page_has_a_pixels_view`. Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer tests.test_evolve tests.test_control` → OK.
- [x] Step 3: browser pane, half-size world: a screenshot of the Pixels view sent to the owner; the browser closed after.
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "The Pixels view: what it sees, as it sees it"; push.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Sharpness a gene with a cost | `python -m unittest tests.test_eyes` | `OK` |
| 2 | Pixels and ears sense the right things | `python -m unittest tests.test_eyes` | `OK` |
| 3 | Brains of before act the same | `python -m unittest tests.test_eyes tests.test_learner` | `OK` |
| 3b | The Pixels view | screenshot sent | the followed one's pixels and ears drawn |
| 4 | Speed | senses and choices, before and after | numbers reported |
| 5 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 6 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
