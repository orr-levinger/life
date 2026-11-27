# Costs fixed by nature, food turned by Auto: design

Agreed by the owner on 2026-11-27 in a side session, relayed to this one: "it's not all about optimizing the number of
creatures... the end goal is to actually have them the most intelligence, the most number of diversity species, and
have them behave as natural as nature can be." Then option A (costs out of Auto, fixed rules; Auto turns only food and
lifespan), "do it" to the numbers below, and "do it" to about 1,000 steps without food as the anchor.

This undoes part of "every cost a knob" (2026-11-26): costs stop being Auto's knobs.

Approved by the owner on 2026-11-27 ("Yes, I approve"), with: costs "changeable from the rules page under the costs
section ... and I should also be able to control the costs in the game itself, in the knobs. Just make it not part of
auto mode. We need reasonable max for each ... between 0 and the max value of the cost. The max value should be what we
expect when they are as smart as they can be. Not over that it will get them extinct in few steps."

## The world-rule change, in one line

Costs stop being Auto's knobs: each is a fixed rule (below), a new world grows into them from a tenth over its first
generations, and Auto turns only food and lifespan.

## What changes

1. **Costs are the owner's rules, never Auto's**: `step_cost`, `move_cost`, `sprint_power`, `eat_miss`, `bump_cost`,
   `attack_cost`, `attack_damage`, `vision_cost`, `hearing_cost`, `neuron_cost`, `stealth_cost`, `armour_cost`, each
   from 0 to its max, in the Rules page's Costs section and as sliders on the world's card (in Auto and in Manual
   alike); both edit the same rule and the running world takes it within a second. Their knob-end rules go (the range
   is the rule's own). **The max is 3 × the natural value** (the owner's "what we expect when they are as smart as
   they can be"): living 0.3 still leaves a full size-1 creature about 330 steps without food; sprint's max is 4.
2. **Auto turns only food and lifespan**: `max_food`, `food_per_step`, `food_energy`, `fruit_rate`, `lifespan` (with
   their knob ends as today).
3. **Growing up**: a new world pays a tenth of each cost at first and grows into the full value as its deepest living
   generation goes from 0 to `grow_up_generations` (10): the share is 0.1 + 0.9 × generation ÷ 10, never going back
   down (the highest share reached is kept, also in saves). After that the costs are the rules' values.
4. **The values**, from one anchor (a full size-1 creature standing still lasts about 1,000 steps without food: tank
   100) and ratios from animals:

   | Rule | Value | Why |
   |------|-------|-----|
   | step_cost (living) | 0.1 a step | the anchor: 100 ÷ 1,000 |
   | neuron_cost | 0.0006 a neuron | 32 neurons ≈ 20 % of living |
   | vision_cost, hearing_cost | today's ÷ 2 (0.025 ÷ (π·9) each) | typical eyes and ears ≈ 5 % of living each |
   | move_cost | 0.4 a unit | a cruising step ≈ one step of living |
   | sprint_power | 2 | as today |
   | eat_miss | 0.1 | |
   | bump_cost | 0.2 | |
   | attack_cost | 1.0 × size | |
   | attack_damage | 10 × size | |
   | stealth_cost, armour_cost | 0.3 | full stealth or armour: +30 % of living |

5. **Checked in the world**: on a half-size test world (radius 18.5, 100 creatures; the owner's rule for this box), a
   few thousand steps of energy spending read by kind. Targets: about half on living and brain, about a third on
   moving, and creatures that steer to food outlive random ones. A share far off: that one ratio changes, the owner is
   told why, and it is checked again.
6. **Old saves** keep the costs they had: a save from before gets each cost rule from its saved settings, grown up
   (share 1). **The exam** keeps today's costs (its own fixed world).
7. **The difficulty number** counts only food and lifespan from now on; a resumed world's difficulty line steps once
   at the update (said on the page's next point, not hidden).

## Not in this

- No smarter Auto, no new measures of intelligence or species (proposed for later in the side session, not approved).

## How it is proven

- Tests: the twelve cost rules with these values; Auto turns only the five; the card has no cost sliders; a cost
  changed on the Rules page reaches the running world; growing up: share 0.1 at generation 0, 0.55 at 5, 1 at 10, never
  lower after; an old save keeps its costs; the exam pays today's costs.
- The half-size world's spending shares reported (numbers), and the steer-to-food creatures outliving random ones.
- The whole suite under 300 s.
