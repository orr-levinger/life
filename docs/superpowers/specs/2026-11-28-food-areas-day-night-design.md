# Grass by patch, sea algae, day and night: design

Asked by the owner on 2026-11-28, after empty plains and empty seas in their world: "do the grass patch of grouped and
food by area"; "the [seas] were getting completely empty of food. So we need to fix that and create some kind of [sea]
forests, not of trees but of some kind of algae, maybe some random patches of it, which is like the trees"; and:
"I want to create some kind of seasons ... let's imagine the earth is spinning. Half of it is facing the sun, half of it
is night ... we have the same amount of food, but half of it is double and half of it is half. So we have a quarter
and 3 quarters ... I want a full rotation every 10,000 steps." Earlier (point 4): "auto should update the food
qunataty and enery per unit average basd on the area (water, forest, planes)".

**The world rule changes (the owner's, quoted above):** grass is limited per patch of ground instead of over the whole
planet; water gets algae beds that grow food as trees drop fruit; the planet turns once every 10,000 steps, and the
side facing the sun grows three quarters of the food. The yearly season of today is replaced by the turning.

## 1. Grass by the patch

- Today one cap holds all the grass on the planet ("Grass pieces at most", 252 in the owner's world): grass nobody
  reaches fills it, and then no grass grows anywhere.
- New: grass grows on a spot only while fewer than "Grass per patch" pieces lie within 3 units of it (a patch, about
  28 square units). A patch nobody eats stays full; a patch that is eaten grows again. The plains fill up, and wherever
  creatures graze, grass keeps coming.
- "Grass per patch" is the knob (it replaces "Grass pieces at most"): from 10 (easiest) to 1 (hardest), a new world
  starting at 3. Auto turns it as it turned the old cap.
- The arithmetic (a seed-1 planet of today's rules, 20,000 points sampled, 2026-11-28): of 17,203 square units, 8,184
  are land (4,539 open, 3,645 within 4 of one of its 132 trees), 8,650 water, 369 rock. Land holds about 290 patches:
  at 3 a patch up to about 870 grass pieces, against 252 today; new grass a step stays "New food per step" (8.2 now).

## 2. Algae beds in the water

- A new world gets "Algae beds" (100) placed in its seas, lakes and rivers (half the planet is water), as it places trees on land. Each bed grows
  algae near it, within 2 units, as a tree drops fruit: at the fruit's rate ("Fruit or algae a tree or bed grows per
  step") and up to the same cap counted on its own ("Fruit pieces at most" for fruit, the same number for algae).
- Algae is a plant: a grass gut eats it, in the water, where swimmers reach it.
- A world saved before gets its beds when it is loaded, placed in its own water the same way.
- The arithmetic: 100 beds × 0.27 (the owner's fruit rate now) = about 27 algae a step, up to 300 in the water.
- Drawn as clumps of dark green-brown weed under the water; algae as small green pieces.

## 3. Day and night

- The planet turns: the side facing the sun moves all the way round once every 10,000 steps (0.036° a step). At the
  equator the edge of night moves about 0.023 units a step; a creature walks about 0.3 a step, so it can follow the
  light; a creature living 4,500 steps sees almost half a turn.
- The same amount of food grows as today, but the lit half gets three quarters of it and the dark half one quarter:
  a grass spot in the dark is kept one time in three (the next spot tried instead, so the amount stays); a tree or bed
  in the light grows at 1.5 × the rate, in the dark at 0.5 ×.
- Today's yearly season (food rising and falling over 20,000 steps) is replaced by this.
- Drawn: the planet lit from the sun's side, its night side darker, turning as the world turns.

## Food by area (the owner's point 4)

Each area now grows its own food by its own knob: grass on open land by the patch, fruit in forests by the tree, algae
in water by the bed; and the light decides where most of it grows. One "Energy per food" stays for all plants (meat ×
2). Different energy per area is not in this; it can be added after this is seen.

## How it is proven

- Tests (small hand-made worlds, few steps): grass stops at 3 within a patch and grows again beside an eaten patch;
  beds grow algae up to their cap, in the water; on a world with the sun fixed, about three quarters of 2,000 grass
  spots land on the lit half; a tree on the dark side grows a third as much as one in the light; the sun's angle after
  2,500 steps is 90°; a save from before loads with beds, the patch knob and no yearly season.
- On a half-size check world: grass on the plains, algae in a sea, the night side darker and moving.
- The whole suite under 300 s.
