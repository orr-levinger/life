# Eyes of pixels, ears of directions: design

Asked by the owner on 2026-11-27: "How can they then form packs or group together if they only see one thing at a
time? we need to really change what they see and what they hear to some kind of an image with pixels." Then: "Perhaps
how many slices a creature has is also evolutionary ... some animals have very high resolution sight, some have very
low resolution." And "Build it".

## The senses change, in one line

A creature sees a row of 16 pixels across its field of vision (each the nearest creature, food and rock or water that
way) at a sharpness its genes set (1 to 16 real pixels, each costing energy), and hears 8 directions around it.

## The eye

- **16 pixels** from its left edge of vision to its right (all around for 360°). Each pixel has 6 numbers, 0 when
  nothing is there: a creature's closeness (1 at it, 0 at the vision reach), its size against its own (0-1), how much
  of a hunter it is (diet), how alike to it (kin); food's closeness; a rock's or water's closeness.
- **Sharpness is a gene, `eye_pixels` (1-16)**: the field is cut into that many real slices; each fills its share of
  the 16 pixels (a sharp eye: 16 pixels of their own; an eye of 4: 4 blocks of 4 alike pixels). A baby's brain reads
  the same 16 pixels whatever its sharpness. Newcomers draw it from 2 to 8; babies inherit and mutate it as any gene.
- **Sharpness costs energy**: a new cost `pixel_cost` per real pixel per step, natural value 0.0006 (16 pixels about a
  tenth of living), from 0 to nature, a new world at a tenth, turned by Auto as every cost.
- What hides a creature today (rocks, a tree's crown, stealth, its cone, its reach) hides it in the pixels.

## The ears

- **8 directions** around it (ahead, ahead-left, left, behind-left, behind, behind-right, right, ahead-right), each 2
  numbers: how loud the movers are that way (the loudest, 1 at it, 0 where it stops being heard) and how loud the calls
  are that way.

## Kept

Everything sensed today stays (the nearest one's genes, pain and who hit it, the nearest food, calls with their kind,
water, shelter). The brain grows by 112 inputs (84 to 196); every brain of before loads with zero weights for them and
acts as before until it learns to use them.

## Seeing what it sees (the page)

The owner: "One is just simple camera, and one is exactly how he sees it with the pixels." Following a creature, the
switch has three views: **Above**, **Eyes** (the camera in its head, as today) and **Pixels**: its 16 pixels drawn as it
gets them, a strip across the screen, each pixel a block whose colour says what is there (a creature: red for a hunter,
tan for a grazer, blue-tinted for kin; food: green; a rock or water: grey or blue), bright when close, dark when far;
blocks as wide as its sharpness makes them; and its 8 ear directions as a ring, brighter where it is loud. The world
sends these numbers for the followed creature only (the page tells it which one, with its watching).

## The one-minute arithmetic

- The pairs (who sees whom, and where in its view) are measured already; sorting them into 16 pixels is one more pass
  over them. Water per pixel: one ray per real slice (the distance map makes a ray a few lookups).
- The brain's first layer grows from 84 to 196 inputs: for 600 creatures a few milliseconds a step at most.
- Packs: "kin to my left, a hunter ahead" is now in one step's senses.

## How it is proven

- Tests: kin left, food ahead, a hunter right land in the right pixels; sharpness 4 gives 4 blocks of 4 alike pixels,
  16 gives 16 of their own; something behind a rock is not in the pixels; a mover behind-left is heard behind-left; the
  gene and its cost; brains of before act the same.
- Speed: the senses and the brains' choices timed on one world state, before and after.
- The whole suite under 300 s.

## What could prove it wrong

After a long run, creatures still do not keep closer to their kin than to strangers.
