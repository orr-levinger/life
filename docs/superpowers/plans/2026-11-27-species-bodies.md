# Species Bodies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Species page shows each species' typical member as the planet draws it: a big picture on its card, a small
one in each row of "Alive now".

**Architecture:** One new export in `src/static/beasts.js` (`speciesPicture`), built on its own `genesOf` and
`bodyPicture`; `planet.js` passes it on; `viewer.html` puts the pictures in.

**Tech Stack:** JavaScript (three.js), Python 3.12 unittest (the page's files as served).

**Spec:** `docs/superpowers/specs/2026-11-27-species-bodies-design.md` (approved by the owner, 2026-11-27: "Approve").

## Global Constraints

- No change to the world, its rules or `species.json`. One body maker: `genesOf` → `beastModel`.
- Tests: no full life simulation; the whole suite under 300 s.
- The check world is half size (planet radius 18.5, 100 first creatures); the browser pane is closed after.

## Review Focus

- The species' gene names (`fear`, `love`, `learns`) reach `genesOf` as the creature's (`fear_of_death`,
  `love_of_babies`, `learns_from_others`), so a species looks like its members.

### Task 1: The pictures

**Files:** Modify `src/static/beasts.js` (export `speciesPicture(typical, size)`), `src/static/planet.js` (passes it
on), `src/viewer.html` (`speciesCard`: the picture at the top; `drawSpecies`: a picture in each "Alive now" row);
Test `tests/test_viewer.py` (new `test_the_species_page_draws_each_species_body`).

**Interfaces:** `speciesPicture(typical, size = 120) → canvas`: `typical` is a species' `now` (or `start`) from
`species.json`; the body is `genesOf({...typical, fear_of_death: typical.fear, love_of_babies: typical.love,
learns_from_others: typical.learns})` with `size: typical.size`, drawn by `bodyPicture`.

- [x] Step 1: test first: `test_the_species_page_draws_each_species_body`: `beasts.js` has `export function
  speciesPicture`, and in it `genesOf(` and `bodyPicture(`, and the three names `fear_of_death: t.fear`,
  `love_of_babies: t.love`, `learns_from_others: t.learns`; `planet.js` passes `speciesPicture` on from
  `/static/beasts.js`; the page calls `Planet.speciesPicture(` in `speciesCard` and in `drawSpecies`. Run `python -m
  unittest tests.test_viewer` → FAIL (no `speciesPicture`).
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 3: on a half-size check world (`python -m src.viewer --runs <scratch> --port 8771`, a world started on the
  page with planet radius 18.5 and 100 first creatures), open the Species page in the browser pane, click the species:
  a screenshot shows the body on the card and in the "Alive now" row; the console shows no error. Stop the world and
  the viewer; close the browser pane.
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s (OK in 297.5 s); commit "Each species' body on the Species page"
  (`git show --stat` → the four files above, this plan and the spec); push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | One body maker, the species' names mapped | `python -m unittest tests.test_viewer` | `OK` |
| 2 | The owner sees the bodies | the screenshot of Step 3 | a body on the card and in the row; no console error |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git ls-remote origin evolve-from-scratch` | the commit's hash |

Anything not done or not proven is reported as such, first.
