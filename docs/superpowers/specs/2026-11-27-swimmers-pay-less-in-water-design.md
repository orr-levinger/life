# Swimmers pay less in water, more on land: design

Asked by the owner on 2026-11-27: "water should affect the enery animal pays per step in water and lamd respectivly" and
"being aquatic should be more energy efficient as far as moving and and just being the land creature as water is less
energy demanding". The rule, approved: "Yes, add it".

## The rule

A creature's water gene (0 land, 1 swimmer) changes what it pays to move and to live, by where it is:

- **in water:** × (1 − water × `swim_saving`)
- **on land:** × (1 + water × `land_cost`)

`swim_saving` 0.5 (0-1) and `land_cost` 0.5 (0-2) are new body rules on the Rules page, each with its plain text. A
pure swimmer pays half in water and 1.5 times on land; a land creature (water 0) pays as today everywhere.

"What it pays to move" is a move's cost (sprints included). "What it pays to live" is the whole cost of a step alive
(living, stealth, armour, senses, neurons). A move is priced where it starts; a step alive where the creature is after
its action. Its speed rules (`land_penalty`, swimming at its water's share of its speed) stay as they are.

## Where it applies

Everywhere a creature lives: new worlds, resumed worlds (a world saved before gets the two rules at their start values),
the exam world (it has no water, so a swimmer there pays the land price; the exam's world itself does not change).

## Not in this

- No change to bumping, eating, attacking or calls.
- No change to the page beyond the Rules page listing the two rules.

## How it is proven

- A test on a pond world: a swimmer (water 1) moving and living in the pond pays × 0.5, on land × 1.5; half a swimmer
  (water 0.5) × 0.75 and × 1.25; a land creature (water 0) on land × 1.
- A test: both rules exist with their start, range and text.
- The whole test suite under 300 s.
