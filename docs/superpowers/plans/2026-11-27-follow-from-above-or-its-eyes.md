# Follow From Above or Its Eyes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A followed creature is watched from above or from its own eyes.

**Spec:** `docs/superpowers/specs/2026-11-27-follow-from-above-or-its-eyes-design.md`

## Global Constraints

- The page only (`src/viewer.html`, `src/static/planet.js`); no change to the world.
- Eyes: camera at its head (height and front from its body radius), looking along its facing, horizontal field of view
  its vision width; only what it senses drawn; its own body hidden. Above: today's follow.
- Page under 33 ms a frame (`bench(30)`); tests under 300 s.

### Task 1: The two views

**Files:** Modify `src/viewer.html` (the POV button becomes an Above | Eyes switch), `src/static/planet.js`
(`pointOfView(on)` becomes `view("above" | "eyes")`; `steer` puts the camera at the eyes; the followed body hidden in
Eyes; the camera's field of view set from the vision width and put back in Above); Test `tests/test_viewer.py`.

- [x] Step 1: test first: `test_follow_from_above_or_its_eyes` (the page has the Above and Eyes buttons; planet.js
  exports `view`, sets `camera.fov` from `vision_width`, hides the followed body). Run `python -m unittest
  tests.test_viewer` → FAIL.
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK.
- [x] Step 3: in the browser pane (a test page server on another port, its own runs folder): follow a creature;
  screenshots from Above and from Eyes sent to the owner; `bench(30)` in Eyes under 33 ms; the test server and its
  world stopped after.
- [x] Step 4: commit "Follow a creature from above or from its eyes"; push to `evolve-from-scratch`.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | The switch and the eyes camera | `python -m unittest tests.test_viewer` | `OK` |
| 2 | It looks from its eyes | screenshots sent | the owner sees the world from its eyes, and from above |
| 3 | Fast enough | `bench(30)` in Eyes | under 33 ms |
| 4 | Pushed | `git push origin HEAD:evolve-from-scratch` | pushed |

Anything not done or not proven is reported as such, first.
