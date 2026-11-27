# The planet view: what each creature is doing, and room to see it: design

Asked by the owner on 2026-11-27: "we lost all the action animations, i have no idea if they are eating fleeing
attacking or what, what does the circle around them even means?"; "still no animations for most actions"; "what are
the big weired circles that show up every few seconds?"; of the legend: "this takes too much of the screen and i want to
be able to drage the edges so i see all the world in the screen"; "add full screen to the world view". Approved as one
plan: "Yes, that plan".

## What changes (the page only)

1. **Action marks on the planet**, from what the world already sends for each creature (`did`, `intent`, `target`,
   `baby`, `call`). A small picture floats just above the creature, always facing the camera and never smaller than a
   few pixels (so it shows from afar too), for the step it happens:
   - **eating**: a green bite; **eating at nothing**: a dotted yellow bite;
   - **attacking**: a red strike line from it to its target and a red star on the target; **hit**: a red flash;
   - **chasing**: an orange line to the one it chases; **fleeing**: a yellow "!";
   - **had a baby**: a pink heart; **a call**: the yellow ring as today (now named in the legend).
2. **The legend** is smaller (half-size pictures in one compact block) and folds away: a **Legend** button shows or
   hides it (hidden at first). It names the two rings: "ring under it: its hunger (green full, yellow half, red hungry)"
   and "big yellow ring: a call, as far as it carries".
3. **Room**: a handle under the planet drags its height (from 240 px to the whole window), kept for next time; a
   **Full screen** button fills the screen with the planet (Esc or the button leaves).

## Not in this

- No change to the world or what it sends; no change to the Above and Eyes views.

## How it is proven

- A test reads the page: planet.js draws each mark (`EAT`, `EAT_MISS`, `ATTACK`, `HIT_BY_ATTACK`, `CHASE`, `RUN_AWAY`,
  `baby`); the page has the Legend button, the two ring words, the height handle, and the Full screen button
  (`requestFullscreen`).
- In the browser pane on a half-size world (the owner's rule for this box): a screenshot with marks showing;
  `bench(30)` under 33 ms; the browser closed after.
