# Legend Matches 3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The legend explains the planet's 3D bodies and marks; the marks are small and sit above the head.

**Spec:** `docs/superpowers/specs/2026-11-27-legend-matches-3d-design.md`

## Global Constraints

- The page only (`src/viewer.html`, `src/static/planet.js`, `src/static/beasts.js`).
- Body pictures rendered once (one small offscreen renderer), reused on every redraw.
- Page under 33 ms a frame, checked on a half-size world; the browser closed after. Tests under 300 s.

### Task 1: 3D bodies in the legend; smaller marks

**Files:** Modify `src/static/beasts.js` (`bodyPicture(genes)`: a small picture of one body), `src/viewer.html` (the
legend's Body rows from `bodyPicture`; the 2D body pictures go; "cone: what it sees"), `src/static/planet.js` (marks
half size, above the head; the eating mark a green open bite); Test `tests/test_viewer.py`.

- [x] Step 1: test first: `test_the_legend_shows_the_3d_bodies` (the page's legend uses `bodyPicture` with each sign's
  words; no `drawCreature(` in the legend), `test_the_marks_sit_above_the_head` (planet.js: the mark size rule and the
  open-bite eating mark). Run `python -m unittest tests.test_viewer` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 3: browser pane, a test server on another port with its own runs folder, a half-size world: screenshots of
  the legend open and of marks; `bench(30)` under 33 ms; server, world and pane closed after.
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "The legend shows the 3D bodies; smaller marks above
  the head"; push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | The legend's bodies are the 3D ones; marks small | `python -m unittest tests.test_viewer` | `OK` |
| 2 | It shows and is fast | screenshots sent; `bench(30)` on the half-size world | legend and marks as described; under 33 ms |
| 3 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
