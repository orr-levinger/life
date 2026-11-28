# Random Newcomers While Few Species Live Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** While fewer than 10 species are alive and there is room, a random newcomer arrives every 50 steps; today's
soup (every step while fewer than 80 live) stays.

**Architecture:** Two rules in `src/rules.py` (`soup_species` 10, `soup_species_every` 50). `SpeciesBook` keeps how
many species were alive at its last look (`alive`). `World.step` adds the newcomer after today's soup when the step is
a multiple of `soup_species_every`, the book's `alive` is under `soup_species` and the world is under `room`;
`add_newcomer(counted=False)` leaves it out of `newcomers` (what Auto reads) and counts it in `species_newcomers`.

**Tech Stack:** Python 3.12 (unittest).

**Spec:** `docs/superpowers/specs/2026-11-28-species-soup-design.md`.

## Global Constraints

- The one world rule change is the owner's (quoted in the spec). Today's soup, births and Auto stay as they are.
- Tests: small hand-made worlds, few steps; the whole suite under 300 s. The test helpers' small world turns the new
  soup off (`soup_species` 0), so the tests that count newcomers today stay as they are.

## Review Focus

- Auto never reads these newcomers as "nearly died out".

### Task 1: The species soup

**Files:** Modify `src/rules.py` (`Rule("soup_species", "world", "Random newcomers also arrive while fewer than this many
species live", 10.0, 0.0, 100.0)`, `Rule("soup_species_every", "world", "…one every this many steps, while there is
room", 50.0, 1.0, 1000.0)` and their words), `src/species.py` (`SpeciesBook.alive`: the species alive at the last
look, `None` before the first), `src/world.py` (`step`: the new soup; `add_newcomer(counted=True)`;
`self.species_newcomers`), `tests/helpers.py` (`SMALL_WORLD`: `"soup_species": 0.0`). Test `tests/test_world_rules.py`,
`tests/test_species.py`.

- [ ] Step 1: tests first:
  - `test_few_species_bring_newcomers` (test_world_rules): a small world with `soup_species` 10, the book's `alive` 3,
    room left, food enough: over 100 steps from step 0, exactly 2 such newcomers (at steps 50 and 100), none between;
    `newcomers` unchanged by them, `species_newcomers` 2.
  - `test_enough_species_bring_none` (test_world_rules): the same with `alive` 10: none.
  - `test_a_full_world_gets_none` (test_world_rules): the same at `room`: none.
  - `test_the_book_counts_species_alive` (test_species): after a look with 3 species holding members, `alive` is 3;
    a new book's is `None`.
  Run `python -m unittest tests.test_world_rules tests.test_species` → FAIL.
- [ ] Step 2: the changes. Run `python -m unittest tests.test_world_rules tests.test_species tests.test_evolve
  tests.test_place tests.test_guts` → OK (today's soup and newcomers unchanged).
- [ ] Step 3: `python scripts/run_tests.py` → OK under 300 s; commit "Random newcomers while few species live" (`git show
  --stat` → the files above and this plan); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | The new soup, the count apart | Step 2's command | `OK` |
| 2 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 3 | Pushed | `git ls-remote origin evolve-from-scratch` | the last commit's hash |

Anything not done or not proven is reported as such, first.
