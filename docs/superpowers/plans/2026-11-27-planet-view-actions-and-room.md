# Planet View: Actions and Room Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The planet shows what each creature is doing; the legend is small and folds away; the planet's height drags
and it goes full screen.

**Spec:** `docs/superpowers/specs/2026-11-27-planet-view-actions-and-room-design.md`

## Global Constraints

- The page only (`src/static/planet.js`, `src/viewer.html`); the world and `/view` unchanged.
- Marks: one instanced batch per kind (one draw each) and one line batch; a mark never smaller than a few pixels.
- Page under 33 ms a frame (`bench(30)`), checked on a half-size world (planet radius 18.5, 100 creatures); the browser
  pane closed after. Tests under 300 s.

### Task 1: Action marks

**Files:** Modify `src/static/planet.js` (`drawActions`: picture marks per kind, strike and chase lines); Test
`tests/test_viewer.py`.

- [x] Step 1: test first: `test_the_planet_shows_what_each_one_does` (planet.js has `drawActions` and each kind:
  `"EAT"`, `"EAT_MISS"`, `"ATTACK"`, `"HIT_BY_ATTACK"`, `"CHASE"`, `"RUN_AWAY"`, `c.baby`). Run `python -m unittest
  tests.test_viewer` → FAIL.
- [x] Step 2: the marks. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 3: commit "The planet shows what each creature is doing".

### Task 2: The legend folds; the planet drags and goes full screen

**Files:** Modify `src/viewer.html` (a Legend button, the legend at half size with the two ring words, a height handle
under the planet kept in `localStorage`, a Full screen button); Test `tests/test_viewer.py`.

- [x] Step 1: test first: `test_the_planet_has_room` (the Legend button; "its hunger"; "a call, as far as it carries";
  the handle; `requestFullscreen`). Run `python -m unittest tests.test_viewer` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 3: browser pane, a test page server on another port with its own runs folder, a half-size world: a
  screenshot with marks, the legend folded and open, the planet dragged taller; `bench(30)` under 33 ms; the server,
  its world and the browser pane closed after.
- [x] Step 4: `python scripts/run_tests.py` → OK under 300 s; commit "A small legend that folds away; the planet drags
  taller and goes full screen"; push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Marks for every action | `python -m unittest tests.test_viewer` | `OK` |
| 2 | Legend folds, height drags, full screen | `python -m unittest tests.test_viewer` | `OK` |
| 3 | It shows and is fast | screenshot sent; `bench(30)` on the half-size world | marks visible; under 33 ms |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
