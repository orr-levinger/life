# The Tree of Species Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Creatures belong to species that split as they drift apart; the page shows the tree and a table.

**Spec:** `docs/superpowers/specs/2026-11-27-tree-of-species-design.md`

## Global Constraints

- Rules `species_apart` 0.10 (0.01-0.5) and `species_group` 5 (2-50), group "genes".
- Distance between genes: the mean of |a - b| over the 18 shown genes, each 0-1 by its range (`genes_table`).
- The book is updated at each scoreboard row; kept in saves; `species.json` beside the scoreboard.
- Checks on this box: half-size worlds, browser closed after. Tests under 300 s.

### Task 1: Species in the world

**Files:** Create `src/species.py` (`SpeciesBook`: `look(world, step)`, `state()`, `from_state()`); Modify
`src/creature.py` (`species`), `src/world.py` (babies inherit it; newcomers species 1), `src/evolve.py` (a look at each
row; `species.json`), `src/saves.py` (the book and each creature's species), `src/rules.py` (the two rules); Test
`tests/test_species.py` (new), `tests/test_saves.py`.

- [x] Step 1: tests first: `test_a_baby_is_in_its_parents_species`, `test_a_drifted_group_splits_off`,
  `test_one_odd_creature_does_not_split`, `test_a_species_drifting_together_stays_one`, `test_a_species_with_no_one_left_ends`,
  `test_the_book_survives_a_save`, `test_a_species_knows_its_facts` (how smart, brain, body, ages, births and deaths
  by cause). Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_species tests.test_saves tests.test_evolve` → OK.
- [x] Step 3: commit "Species: born from a parent's, split when a group drifts apart".

### Task 2: The tree on the page

**Files:** Modify `src/viewer.py` (route `/species?run=`), `src/viewer.html` (the Species panel: the tree and the
table; clicking a species opens its card: its numbers, its typical member in words, what changed since it began); Test `tests/test_viewer.py`.

- [x] Step 1: tests first: `test_the_species_book_is_served`, `test_the_page_draws_the_tree`,
  `test_a_species_card_tells_its_story` (the book holds the start and now typical members and the largest count; the page's
  card names each trait). Run → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 3: browser pane, a test server on another port with its own runs folder, a half-size world run long enough
  for a split (or the species of a test world made by hand): a screenshot of the tree sent; closed after.
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "The tree of species"; push to
  `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Species inherit, split, end, are saved | `python -m unittest tests.test_species tests.test_saves` | `OK` |
| 2 | The page serves and draws the tree | `python -m unittest tests.test_viewer` | `OK` |
| 3 | It shows | screenshot sent | the tree and the table |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
