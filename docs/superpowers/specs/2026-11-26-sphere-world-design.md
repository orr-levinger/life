# A sphere world — design

**The owner, 2026-11-26:** "I want the world to be 3d sphere." Creatures live **on the surface** of a ball. Asked why:
"I should be able to easily rotate and zoom in and out." Answers to the design questions, in the owner's picks:
- How real: **a real sphere** — positions on the ball, distances and directions along its surface; no edges, no
  stretched poles, every spot the same.
- Old flat worlds: **the sphere replaces the flat map**; old flat saves stay on disk but cannot be resumed.
- Size: **twice the ground** of today's new world (120 × 72 = 8,640 square units → 17,280: radius 37).
- Creatures: **room for 600**, the same crowding as now; rocks, ponds, trees and food double with the area.
- The view: **a real 3D engine (three.js)** — chosen over drawing the globe on today's canvas.
- Where it runs: "I will just pull the repo from my Mac and run it locally. Let's forget about the hosting for now." —
  the owner runs the simulator and the page on their Mac (its graphics card draws the planet); no cloud hosting. The
  30-frames-a-second check is done on the Mac.

## How it looks (the owner, 2026-11-26, after a 3D speed test: "Make the obstacles look like rocks … extruded out of
## the grass. The grass itself should have some texture to it. The trees look really cool. And the creatures
## themselves, they must also be 3D.")

- **Rocks:** 3D boulders rising out of the grass, each of rough, irregular shape and its own size (from its blob of
  caps); grey-brown, lit, casting a sense of height. Ponds stay flat water, a little sunk into the ground.
- **Grass:** the ground has texture — patches of lighter and darker green, fine noise, so the eye reads distance and
  movement; no flat colour.
- **Trees:** as in the test — round crowns grouped in forests (the owner liked them), with a trunk under each crown.
- **Creatures:** 3D bodies standing on the ground, facing where they go. Their genes still show, as today in 2D:
  size → body size; love of babies → roundness; hunger → body colour (green full, yellow half, red hungry); diet → a
  collar from green (plants) to dark red (meat); fear of death → quills on the back; brain layers → ridges; learning
  from others → antennae; eyes on the front; armour → plates; stealth → paler, less shiny; water → fins. Cones, hearing
  and call rings are drawn on the ground around them.
- **Speed:** the test drew the full scene in 19 ms a frame on this box (52 a second); 3D creatures and rocks must keep a
  frame under 33 ms here (30 a second) with 600 alive, measured by drawing frames back to back. A look test with 3D
  creatures (body, head, eyes, legs, collar, quills as separate parts, ~1,150 triangles each) took 52–59 ms to draw and
  50–80 ms to move: so each creature is one merged, low-detail model (about 150 triangles, its gene parts included),
  and creatures are re-placed only when the world sends a new picture, not every frame.

## Intent

A planet the owner can spin and zoom easily, from the whole ball down to one creature, with no edge anywhere for a
creature to hide against or run into. Nothing about the planet itself (latitude, climate) changes how life works:
every spot is the same.

## Goals

1. The world is the surface of a sphere of radius `planet_radius` (a rule, start 37). Walking straight brings a creature
   back to where it started (after 2π × 37 ≈ 232 units).
2. Everything that today uses x/y (moving, seeing, hearing, calls, reach for eating and attacking, rocks, ponds, trees,
   fruit, food, pond edges, forests, newcomer spots, babies' spots) works on the sphere with distances along the
   surface.
3. A creature's senses are **its own view**: every direction it senses (food, the one it sees, the one it hears, a
   call, its pain, the obstacle in view) is given as (ahead, to the left) relative to where it faces, not as east and
   north — on a ball there is no fixed east everywhere. Its move is chosen the same way: ahead and to the left.
4. Room for 600 (`room` start and highest 600; learner slots 600). A new world: 200 first creatures, soup 80, Auto aims
   at 480, 70 obstacles, 92 trees in 4 forests; food per step and food at most doubled at the start (`START`
   settings, which Auto turns from there).
5. The live page shows the planet in 3D (three.js): drag to spin it any way, wheel or pinch to zoom from the whole
   planet to a close-up; follow a creature (the planet turns under it, it stays in the middle); its point of view (the
   camera on the ground behind it, seeing what it sees). Creatures keep today's look from their genes (shape, hunger
   colour, diet outline, cones, call rings), drawn on the ground.
6. The exam keeps its own fixed world: a small sphere with the area of today's 50 × 30 exam (radius √(1500 / 4π) ≈
   10.9, a rule `exam_radius`), its bodies straight-line in size as now.

## Non-goals

- Latitude, poles, climate or day and night: the planet is the same everywhere.
- Flat worlds: no switch between flat and sphere; the flat code goes.
- Resuming old flat saves (format 26 and before): refused in plain words; their files and history stay.
- Life inside the ball (3D volume).

## Anti-goals (a failure even if it works)

- Anything that behaves differently near some spot (a seam, a pole, a place where directions flip).
- A page that stutters while spinning or zooming with 600 creatures (below 30 frames a second on this machine).
- A step much slower than today's new world for the same crowding: with 600 alive, the time is measured and told to
  the owner before any world switches.

## Constraints

- Every number is a rule with a plain meaning on the Rules page (owner's standing rule).
- Tests first; the whole suite under 5 minutes; no full life simulation as a test (owner's standing rules).
- three.js is a library downloaded once (the owner is asked before it is downloaded: file, source, size) and kept in
  the repository, served by the page server itself — the page never loads anything from the internet.
- No code before an approved plan; never stop the owner's world.

## Approaches

**The world's geometry** — one sensible way: positions are points on the ball (3D unit vectors × the radius);
distance is the arc along the surface; a creature's facing is a direction in the ground at its spot, carried along
when it moves. (Latitude and longitude were set aside: they have poles where directions break, an anti-goal.)

**The view** — the owner chose between:
1. Drawing the globe on today's canvas: least rebuilding, today's creature drawings reused, no library; no lighting.
2. **A real 3D engine (three.js) — chosen**: a lit, shaded planet, the smoothest spin and zoom; a library to add, and
   the drawing of creatures rebuilt for 3D.

## How it is built (decided)

- **Geometry (`src/sphere.py`, new):** a point is a unit vector `p`; the world radius `R`. Arc distance
  `R · atan2(|p × q|, p · q)` (exact also for tiny and near-opposite distances). Moving `d` along a ground direction
  `t` (unit, tangent at `p`): rotate `p` about `p × t` by `d / R`; the facing rotates the same way. Bearing of `q` seen
  from `p` facing `f`: the tangent toward `q`, as (ahead = its part along `f`, left = its part along `p × f`).
  Everything is done for all pairs at once with arrays, as today.
- **Obstacles, trees, ponds, forests:** today's blobs of circles become blobs of caps (a centre point and an arc
  radius); "inside", "overlaps", "edge distance" and "a rock between two points" are measured along the surface. A pond
  still lets swimmers in; a rock still hides what is behind it.
- **Brains:** the inputs keep their layout (47); only the direction pairs change meaning to (ahead, left). Brains of
  old worlds are not converted (their "east" means nothing on a ball).
- **Saves:** a new format holds the sphere (positions and facings as 3D, the planet radius, caps). Older formats are
  refused with "a flat world from before the sphere: it cannot be resumed".
- **The page:** three.js and its orbit controls kept in the repo (`src/static/`), served by `src/viewer.py`. The planet
  a sphere mesh; rocks, ponds and tree crowns drawn as patches on it; each creature a flat drawing lying on the ground
  at its spot, facing its way, painted from today's 2D drawing (so its look from genes stays); cones and call rings as
  shapes on the ground. Live tiles, charts and the Rules page stay as they are.

## Calls made in writing this up (the owner may correct any)

1. **Flat worlds at the switch:** no flat world is running (the owner's last one stopped 2026-11-25 18:46). Once the
   sphere lands, "Update code", "start from a save" and the runner refuse a flat save up front, in plain words, before
   anything stops.
2. **Facing after a move:** as today — a creature faces the way it last moved (a move of (0, 1) turns it to its left).
3. **Point of view:** as today — the camera at the creature, showing only what it sees and hears (the page already gets
   each watched creature's senses).
4. **Speed baseline:** today's 120 × 72 world with 300 alive (53–93 ms a step). With 600 alive the step time is told
   to the owner; above 200 ms a neighbour grid on the sphere (only nearby pairs are compared) is added.
5. **Yardsticks and measures:** the random and hand-written yardsticks and their proximity sensor work on the sphere;
   the scoreboard's "toward food", "flees" and "chases" use bearings on the ground; "direction concentration" becomes
   how much the facings of all creatures agree in 3D (1 = all one way). The proof thresholds stay. Old flat runs stay
   listed and charted as they are.
6. **Planet radius:** a rule for new worlds (like a new world's size today); a running world keeps its own.
7. **Rule limits:** raised so the new starts fit — `room` and `auto_target` highest 600, `first_creatures` and `soup`
   highest 600, `obstacles` 300, `trees` 400; `forests` start 4, `forest_spread` stays 12.
8. **The flat matplotlib window** (`src/visualization.py`, `--window`) goes with the flat code.

## Left to the builder

- Mesh detail, how creature drawings are cached as textures, how far the camera may zoom in and out.
- How the follow and point-of-view cameras move smoothly.
- Where each new rule sits on the Rules page.

## How it is checked

- Walking straight for 2π R comes back to the start (within 0.001); arc distances match a known case (a quarter turn
  = π R / 2); a bearing straight ahead is (1, 0), to the left (0, 1), from anywhere on the ball, including two spots on
  opposite sides.
- Every existing behaviour test runs on the sphere (eating, attacking, calls, kin, water, stealth, armour, seasons,
  forests, pond edges), on a small test planet.
- The same small world run from two far-apart starting spots behaves alike (no special place).
- Speed: a new planet with 600 alive, timed; the ms per step told to the owner.
- The page: a screenshot of the whole planet and one zoomed in, and spinning and zooming tried by hand; frames a second
  measured with 600 creatures.

## Done means

A new world on the page is a planet the owner can spin and zoom, with 600 room, every trait working on it, the suite
passing under 5 minutes, and the step time and frame rate measured and told.
