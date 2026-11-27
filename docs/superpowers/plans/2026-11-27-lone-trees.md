# Lone Trees Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Besides its forests, a new planet gets `lone_trees` single trees anywhere on land.

**Spec:** `docs/superpowers/specs/2026-11-27-lone-trees-design.md`

## Global Constraints

- New rule `lone_trees` 40 (0-200), group "world". New worlds only.
- A lone tree: a random land point, not in water, not on a rock, not within two crowns of another tree; at most
  `PLACE_TRIES` tries each.
- Tests under 300 s; never a full life simulation as a test.

### Task 1: Lone trees

**Files:** Modify `src/rules.py` (the rule and its text), `src/terrain.py` (`make_terrain` plants the lone trees after
the forests'); Test `tests/test_terrain.py`, `tests/test_rules.py`.

- [x] Step 1: tests first: `test_lone_trees_grow_all_over_the_land` (radius 37, 2 forests, 40 lone trees: at least 30
  trees farther than `forest_spread` from every forest centre; none in water; none on a rock); `test_the_lone_trees_rule`
  (start 40, range 0-200, group world, a text). Run `python -m unittest tests.test_terrain tests.test_rules` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_terrain tests.test_rules tests.test_water tests.test_saves`
  → OK.
- [x] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Lone trees grow all over the land"; push to
  `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Lone trees on land, none in water or on rocks | `python -m unittest tests.test_terrain` | `OK` |
| 2 | The rule on the Rules page | `python -m unittest tests.test_rules` | `OK` |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
