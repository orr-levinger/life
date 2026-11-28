# Meat worth more, corpses that stay: design

Asked by the owner on 2026-11-28: "predetors dont evolove its seems not profitalbe ... lets invetigate this"; "i didnt
see corpes of dead animals (tey should look different then regualr food)"; "they size should srhink as its being
eating". Then, on the numbers: "from meat should give double the energy as grass and fruits, and should be riskier and
rarer and harder to get. That's the trade-off. And corpse should remain much longer, not 5 steps, maybe 100 steps";
"Corpse should stay depending on its size. A big creature should last longer."

## What was found (the code, read 2026-11-28)

- A bite of meat gives `bite × meat gut`, the same as a bite of grass gives `bite × grass gut` (`src/creature.py`
  `_eat`). A hit gives the hunter `0.8 × damage × meat gut` and costs `size × attack_cost`.
- With the owner's world's costs (attack cost 1.16, attack damage 10), a hit loses energy below a meat gut of about
  0.13, and a whole kill below about 0.12; below about 0.24 a step hunting earns less than a step grazing.
- A death leaves a corpse of `8 × size` energy that is gone after 5 steps (`World.kill`): 4.8 for a size-0.6 creature,
  on screen half a second at the watched speed, drawn small and dark.
- Food is drawn at the size it was made, however much of it is eaten.

So the first small steps toward meat always lose: hunters cannot begin to evolve.

## The world rule changes (the owner's yes, quoted above)

| What | Today | New |
|---|---|---|
| Energy from a bite of meat | bite × meat gut | **2 ×** bite × meat gut (new rule `meat_value`, 2) |
| Energy from a hit | 0.8 × damage × meat gut | 0.8 × damage × meat gut × `meat_value` |
| Food in a corpse | 8 × size (`corpse_energy` 8) | **40 × size** (`corpse_energy` 40) |
| How long a corpse lasts | 5 steps (`corpse_steps` 5) | **100 × size** steps (`corpse_steps` 100, now × size) |

Unchanged: grass and fruit energy, the bite, attack cost and damage, speeds. Meat stays rare (only where something
died), it rots, and fresh meat means a hit and its cost: the owner's trade-off.

The arithmetic, with the owner's world's costs: a hit at meat gut 0.1 goes from −0.17 to +0.36 energy; a size-1 corpse
feeds a full meat eater 80 (about three grass pieces) and lasts 100 steps; a size-0.5 one 40 for 50 steps.

A save from before takes the new values (`meat_value` 2, `corpse_energy` 40, `corpse_steps` 100), so the owner's
running world gets them when it is updated; its corpses lying then keep the steps they had left.

## What is drawn

- **A corpse looks like a lying body:** a flattened, dull grey-brown body shape with pale ribs, as big as its food,
  not like grass (tan), fruit (red) or a creature.
- **Food shrinks as it is eaten:** grass, fruit and corpses are drawn by what is left of them (the world sends each
  food's share left); a half-eaten piece is drawn at about 70 % of its size (the size by the square root of the share).

## Not in this

- Food per patch of ground and per area, the fruit cap knob, shorter names: their own plans.

## How it is proven

- Tests (small hand-made worlds, few steps): a bite of meat gives twice a bite of grass to the same gut; a hit gives
  `meat_value` times; a size-1 corpse holds 40 and is gone after 100 steps, a size-0.5 one after 50; a save from before
  loads with the new values; the view file sends each food's share left.
- On a half-size check world: a corpse seen lying where a creature died, looking unlike food; a piece being eaten
  drawn smaller.
- The whole suite under 300 s.
