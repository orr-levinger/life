# Back in time from any chart: design

Asked by the owner on 2026-11-27, looking at the charts of a world that died out twice: "i should be able to picl a
point in time in a chart and move bck in time"; then "Ok so add the jump to the past button from any chart". Asked
what happens to the world running now: "It's stops and saved and we resume from the past.. But one thing we have to
make sure is that when we jump to the past and the code is new and all the genes are new and everything is a bit
different, it should still work. With the creatures should continue with the new code from that step without the bugs
or anything that we introduced."

## What the owner sees

- **On any chart of a running world** (Live): clicking a point marks that moment with a thin line and shows a button
  under the charts: "Go back to step N", where N is the world's last save at or before the clicked point (a world
  saves itself every 2,000 steps and keeps every save). A second click elsewhere moves the mark; Esc or "Cancel"
  removes it.
- **Pressing it**: the world running now saves and stops (it stays in History, with all its saves), and a world starts
  from that save, in the same mode (Auto or Manual), with today's code. It is called, as today, "from the HH:MM world
  at N", and its charts show the old world's history up to step N, then its own.
- The button is not shown for a world that has no save at or before the point.

## How it is built

- The page (`src/viewer.html`): a click on a chart turns its x position into a step and finds the save from the
  world's list (`/saves`, already there); the button sends `/control` with `back_to: <save step>`.
- The page server (`src/viewer.py`, `control`): `back_to` is checked (a save of this run), then handled like Update
  code: the world is told to stop (it saves as it ends), and when it has ended, a world starts from that save, as the
  existing "Start a new world from this save" does (`--resume`, a new run folder, the same mode). The two already
  share how a world is started; nothing is copied.
- Nothing new in the world: a resumed world already carries its past history (`_history`) and is converted by the
  save loader (`src/saves.py`) when the save is from older code: the two guts, gills and lungs from diet and water, the
  costs, the senses.

## Today's code on an old save (the owner's "it should still work")

A save from before today's changes is loaded with today's rules, exactly as Update code does. Measured on the owner's
world at its update (step 413,649, from its scoreboard in the bucket): its creatures went on, but two of its costs had
been turned above nature by the older code (eating at nothing 0.30, bumping 0.33) and today's rule holds every cost at
or below nature (the owner's own rule of 2026-11-27, "never above nature"), so Auto's first look set them to 0.10 and
0.20. The difficulty number also jumped (78 to 444) because today's code counts the costs in it; the world itself did
not change at that moment. A jump to the past will do the same to a save from before: this is today's rules, not a
bug.

Proven by tests, not by hope:
- A save written by older code (the two saves the tests already keep, `tests/data/before_guts` and one written now by
  today's code) is loaded, goes back to, and steps on with today's code: no error, every creature's genes in range,
  its costs at or below nature after Auto's first look, its brain choosing as before (the test of 2026-11-27 for the guts).
- The whole jump, on a small hand-made run with two saves: the world is stopped, the new world starts from the chosen
  save, in the same mode, its record says from which run and step.

## Not in this

- No change to the world's rules or to Auto. No jump inside the same run (the past world is a new world; the stopped
  one keeps everything). No jump on History's charts of an ended world (it already has "Start a new world from this
  save").

## How it is proven

- Tests (above), in `tests/test_viewer.py` and `tests/test_saves.py`; the whole suite under 300 s.
- On a half-size check world on this box (planet radius 18.5, 100 first creatures): click a chart, press the button;
  the old world ends in History, the new one runs from that step with its history before it; screenshot; the browser
  pane closed.
- On the owner's Mac: after the owner's first jump, Claude reads the new world's reports from the bucket (its events
  say resumed from that step; no errors) and tells the owner.
