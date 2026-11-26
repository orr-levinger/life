# Every cost a knob: design

Approved by the owner on 2026-11-26 ("Ok").

## The goal, in the owner's words

"all the cost of everything is supposed to be part of the auto mechanism to increase and reduce the cost, to make it
easier for when they have a very simple brains. Nothing should be hard-coded. Just increase the max and reduce the
lows, 0 to whatever ... cost of sprinting should also be something auto control and is draggable and in the knobs.
Anything that costs must be part of the auto mechanism that auto can control."

And on where a new world starts: "If the cost of everything at the beginning is 0, then they are not learning. It has
to be some non-zero value" - agreed: a new world starts each cost at a tenth of today's value.

## Why (measured 2026-11-26)

A new planet loses most of its creatures to hunger in its first 150 steps. Per creature-step: moving -1.25, all else
-0.28. A newborn's random moves are sprints; a random move costs 0.26 / 2.03 / 12.07 (size 0.5 / 1 / 2) against 0.13
for a cruising step of size 1. At a tenth of today's costs a size-1 random move costs about 0.03.

## What changes

1. **Every cost is a knob** (a Settings field, a slider on the page, turned by Auto): moving, living (step), and
   newly sprinting (`sprint_power`), eating at nothing (`eat_miss`), an attack (`attack_cost`), being hit
   (`attack_damage`), bumping into a rock (`bump_cost`), seeing (`vision_cost`), hearing (`hearing_cost`), each neuron
   (`neuron_cost`), stealth's cost (`stealth_cost`), armour's cost (`armour_cost`).
2. **Each cost's easy end is 0**; its hard end is at least today's value (5-10 times it, see the plan).
3. **A new world starts each cost at a tenth of today's value.** Food, lifespan and fruit start as they do now.
4. **Auto turns every knob** as it does now (one knob at a time), and every nudge moves a knob by **5 % of its value or
   1 % of its whole range (hard end - easy end), whichever is bigger** (the owner: "some min value change, like 1% of the
   total range should be the min change"). So a knob at 0 or near it never gets stuck.
5. The ten new knobs leave the Rules page (one place for each number).
6. **Old worlds keep their costs**: a save or a scoreboard from before has today's values for the new knobs and
   resumes as it was.
7. **The exam world does not change**: its costs are today's values.

## Not in this

- No change to what a cost does, only to who sets it and where it starts.
- Seas, islands and rivers continue after this (their plan, from its Task 3).

## How it is proven

- Tests: every cost is a knob with easy end 0; a new world starts at a tenth; Auto lifts a knob from 0; an old save and
  an old scoreboard resume with today's costs; the exam's costs are today's.
- Live: on a new planet, hunger deaths in the first 1,000 steps fall far below today's (about 320 of 600), and
  `toward_food` climbs above 0 within about 20,000 steps (watched on the graphs).

## What could prove it wrong

`toward_food` stays at 0 while the world is easy: the brains are not learning from the small costs.
