# Eyes and Make Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eyes grow big with how far a creature sees and show on hunters; species pictures fill their frames; Make takes the world's current gene limits
each time it is shown; long names do not widen Make; the Species pictures appear even when the 3D code loads late.

**Architecture:** `beastModel` in `src/static/beasts.js` sizes and places the eyes from the head; `drawMake` in
`src/viewer.html` reads the rules on every showing and redraws the sliders within them; one CSS rule limits the list's
width; the module that loads the 3D code calls `render()` once it has loaded.

**Tech Stack:** JavaScript (three.js), Python 3.12 unittest (the page's files as served).

**Spec:** `docs/superpowers/specs/2026-11-28-eyes-and-make-fixes-design.md`. The spec and this plan approved by the owner,
2026-11-28: "Approve both".

## Global Constraints

- Nothing in the world changes: no rule, no file it writes. Only `src/static/beasts.js`, `src/static/gallery.html`,
  `src/viewer.html` and `tests/test_viewer.py`.
- Tests: no full life simulation; the whole suite under 300 s. The check world is half size.

## Review Focus

- Make keeps the owner's slider positions when the limits are read again (only values outside the new limits move).

### Task 1: Eyes and Make fixes

**Files:** Modify `src/static/beasts.js` (`beastModel`: `eyeR = headR * (0.15 + 0.45 * g.reach)`; eyes at
`hy + headR * 0.45`, in front of the brow, for every body plan; `bodyPicture(genes, size, close)`: framed close (the
camera fits the body's box, no fixed distance) when `close`, the key holding it; `speciesPicture` passes `close`), `src/static/gallery.html` (an eye row: grazer and
hunter at reach 0, 0.5, 1), `src/viewer.html` (`drawMake`: the rules read on every showing; when Make is already built
for that world, `made.limit` replaced, `keepWithin()` moves only values outside the limits, `drawMakeSliders()`;
`#make-page select { max-width: 100%; text-overflow: ellipsis; }`; the species card's picture 320 px, the table's 56; the module script calls `render()` after
`window.Planet = Planet`), Test `tests/test_viewer.py`.

- [x] Step 1: tests first:
  - `test_eyes_grow_with_reach`: `beastModel` has `headR * (0.15 + 0.45 * g.reach)`.
  - `test_make_reads_the_rules_each_time`: in `drawMake`, the `/rules?run=` fetch comes before the `made.run === run.name`
    check; `function keepWithin(` exists.
  - `test_long_names_do_not_widen_make`: the page's CSS has `#make-page select` with `max-width: 100%`.
  - `test_the_page_draws_again_when_the_planet_loads`: the module script has `render()` after `window.Planet = Planet`.
  - `test_species_pictures_fill_their_frame`: `speciesPicture` passes `true` for `close`; `bodyPicture` has a
    `close` parameter in its key; the card draws `speciesBody(Planet.speciesPicture(s.now, 320), 320)`.
  Run `python -m unittest tests.test_viewer` → FAIL (none yet).
- [x] Step 2: the changes. Run `python -m unittest tests.test_viewer` → OK (112 tests; `test_a_body_picture_is_kept_by_its_size`
  now looks for the key with `close` in it too).
- [x] Step 3: the gallery's eye row: big eyes at reach 1, a hunter's eyes seen; screenshot to the owner.
  Done 2026-11-28: the grazer's eyes grow to half its head; the hunter's sit big on its upper head.
- [x] Step 4: on a half-size check world on this box: in Rules, vision reach's highest changed; Make's "Sees how far"
  slider takes it without a reload; no page wider than the screen; the Species page opened first shows its pictures,
  each filling its frame.
  Screenshot. The world and its viewer left running for the owner to try Make, its address given to the owner.
  Done 2026-11-28 on the 42-species world (`20261128-140314-1-learning`, viewer on port 8771): Make opened with
  "Sees how far" up to 30; the rule set to 20 from outside, Make shown again: up to 20, the slider kept at 15.5, no
  reload; the rule set back to 30. The page no wider than the screen. The Species page opened first: 42 pictures in
  the tree and the table, the card's 320 px with the body filling it.
- [x] Step 5: `python scripts/run_tests.py` → OK under 300 s; commit "Eyes you can see; Make keeps up" (`git show
  --stat` → the four files and this plan). Not pushed: the owner tests here first ("before pulling on the mac").
  Not done yet: the whole suite went over 300 s (2026-11-28, the owner's world and the check world running on this
  box; still running `test_evolve`, `test_evolve_runs`, `test_learner`, `test_viewer`; none failed). Not committed:
  waiting for the owner's test here. The owner, after trying it, 2026-11-28: "Commit and push": committed and pushed
  without the whole suite passing.

## Must be completed to mark this as done

| # | Item | Command or check | What it must show |
|---|------|------------------|-------------------|
| 1 | Eyes, Make limits, list width, late 3D | `python -m unittest tests.test_viewer` | `OK` |
| 2 | The owner sees the eyes | Step 3's screenshot | big eyes at reach 30, a hunter's eyes seen |
| 3 | Make keeps up, species pictures fill their frames | Step 4's check | the new limit on the slider without a reload |
| 4 | Everything else works | `python scripts/run_tests.py` | `OK in … s (limit 300 s)` |
| 5 | Committed for the owner's test | `git log --oneline -1` | "Eyes you can see; Make keeps up" |

Anything not done or not proven is reported as such, first.
