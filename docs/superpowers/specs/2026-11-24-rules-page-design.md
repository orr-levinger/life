# The Rules page, and the rules that were missing

**Date:** 2026-11-24 · **Status:** for the owner's approval · **Builds on:** `2026-11-23-living-world-design.md`

## The owner's words

- "table to me all the rules of the game! it seems many things I wanted are missing!"
- "actually create a configurable page in the webapp that shows everything and I can control it" · changes apply "Yes,
  right away" · "all the genes also" · on the mockup: "no toggles in the body and movement and put only configurable
  stuff!" · "change the table to costs, values — it's confusing that costs is merged with allowed values" · "where is the
  cost for sprinting?!" · "show everything that has a cost".
- "the whole point is that hearing is 360 shorter range and vision is max 120 longer range" · "hearing can't tell size!"
- "better vision cost more and better hearing cost more" · "each neuron should have a cost to it" · "additional hidden
  layers and more or less neurons are also genes" · "if you are bigger every step should cost more!"
- "fix the UI" · "we should run 1 world at a time" · "charts are always to the single world we are now testing" · chart
  lines only for the owner's own changes: "Yes, add it".

## What is in scope

1. **A Rules page** showing every rule and every gene of the world, each changeable, applied to the running world at
   once and kept with it.
2. **The rules that were missing or wrong:** vision at most 120°, hearing without size, size costs energy every step and
   in moving, the brain's size in the genes with a cost per neuron.
3. **One world at a time:** Live shows the one running world, its charts alone, lines only for the owner's changes, a
   visible edge on each vision cone.

Out of scope: the reward genes, learning from others and the pain sensor themselves (built; running in the owner's world
since step 158,000); the exam's design; Auto's way of steering.

## 1. The Rules page

- A third tab, **Rules**, beside Live and History, for the running world, in three tables:
  - **Costs** — every energy cost a creature pays, each a row: being alive (each step, × size) and moving (per unit of
    distance, × size), marked "Auto" because Auto moves them while on Auto (the owner's change holds on Manual);
    sprinting (the move × (speed ÷ cruising speed) to this power: 2 today, range 1–4); vision (per square unit seen,
    each step); hearing (per square unit heard, each step); each neuron (each step); eating at nothing; an attack, hit or
    miss (× size); being hit (energy lost, × the attacker's size); having a baby (the share of its energy it gives).
  - **Values** — everything else that is a number: the world's size, the room for 80, the soup of 10, food's lifetime,
    speeds, the energy tank, birth at 70 %, the baby's 40 %, low energy at 30 %, hunting's damage, range and the hunter's
    80 %, corpses, the eat range, newcomers' energy, the brain's growth chances, Auto's target of 64, its nudge size and
    how often it looks.
  - **Genes** — below.
- **Only what the owner can change is on the page:** each row is the rule in plain words, a number box with its value
  now, its allowed range, its starting value and Reset. No switches, no read-only rows, no live statistics.
- **Every rule is there.** Nothing about the world stays a constant hidden in code: each value the code uses is a rule —
  the world's size, the room for 80, the soup of 10, birth at 70 %, the baby's 40 %, low energy at 30 %, speeds and the
  sprint, hunting's damage, cost, range and the hunter's 80 %, corpses, the eat range, eating at nothing, newcomers'
  energy, sense costs, the cost of size, the neuron cost, Auto's target of 64, its nudge size and how often it looks.
  Learning's own machinery (PPO's rates, rollout, clip) is not a rule of the world and is not shown.
- **Every gene is there**, in the Genes table, one row each: size, vision width, vision reach, hearing, fear of death,
  love of babies, learning from others, learning speed, layers, neurons. Each row has number boxes for its limits (lowest
  and highest), a newcomer's amount (a range for the genes newcomers draw at random, one number for the ones they start
  with) and how much a baby's copy may change.
- **Auto's six knobs** stay on the world's screen too; the two that are costs (being alive, moving) are also rows of
  Costs, marked "Auto" (the owner: "show everything that has a cost").
- **Applied at once:** a change reaches the running world within 25 steps, as the knobs do today. A rule about how a
  creature is born (a gene's limits, start or mutation) applies to the next creatures born; a rule about each step (a
  cost, a speed) applies to everyone at once. A creature's own genes never change in its life; a gene's new limits never
  reach back into creatures already alive.
- **Kept with the world:** its rules are written into every save, so resuming or continuing keeps them; a new world
  starts from the starting values. Saves become format 6 (a format 5 save gets the starting values).
- **Checked on the way in:** a value outside its range is refused with a plain message beside its box.
- The world screen gets a "Rules" link to the page.

## 2. The rules that were missing or wrong

| Rule | Now | Becomes |
|---|---|---|
| Vision cone width | gene 30–360° | gene **30–120°** |
| Vision reach | gene 1–8 | gene **1–10** (farther than hearing) |
| Hearing | gene 0–4; where, how far **and size** | gene 0–4; **where and how far only** (4 inputs) |
| Cost of being alive | the same for every size | **× size** (a size-2 body pays twice a size-1's) |
| Cost of moving | the same for every size | **× size** (always) |
| Brain size | fixed: 1 layer of 32 | **genes:** 1–3 layers of 1–64 neurons, as the approved "brain's size evolves" |
| Cost of a brain | none | **0.0005 energy a step per neuron** (32 neurons: 0.016) |

- **The owner's world on update:** cones narrow to 120° and reach 5 (area 120/360 × π × 5² = 26.2, against today's
  full circle of 3, π × 3² = 28.3, so its sense cost hardly changes); the heard-size input leaves every brain; each brain
  becomes 1 layer of 32 with room to grow. Brains read 20 inputs: 12 seen, 4 heard, 4 pain.
- **The brain's size evolves** as already approved (spec section "The brain's size evolves"): room for 3 layers of 64;
  at birth a neuron may switch on or off (5 %) or a layer (1 %); a new neuron or layer starts silent, so a baby acts like
  its parent at birth. Its rates and the neuron cost are rules on the page.
- The new numbers (reach 10, × size, 0.0005) are Claude's starting values, not measured; the page lets the owner change
  them.

## 3. One world at a time

- **At most one running world.** Start is refused while one runs ("Stop the running world first").
- **Live shows that one world:** its controls, tiles and charts; nothing of any other world.
- **Charts mark only the owner's own changes:** a line where a knob or rule was changed by hand; Auto's nudges are words
  in the hover text only.
- **A vision cone has a visible edge** in Jump in: a thin outline at the cone's end, so how far each creature sees is
  clear; hearing stays the faint dotted circle.
- This replaces the approved "table of live worlds" (Task 4x), which one world at a time makes pointless.

## How it is checked

- Tests for: every rule and gene on the page with its range, and nothing on it that cannot be changed; a change refused
  outside its range; a change reaching the running world; rules kept in a save and restored; the cone limit of 120°; hearing
  giving no size; the costs × size; the brain's size genes (silent growth, bounds, the neuron cost); the owner's world's
  conversion; one world at a time; lines only for the owner's changes.
- **The fair check** before the owner's world is updated: the real runner, with Auto, from a copy of the owner's latest
  save, 8,000 steps, three seeds with today's code and three with this; a run ending below 40 alive is a fail; shown live
  and its numbers given to the owner as they come out.
- In the app's browser: the Rules page, a change reaching the running world, Live with the one world, the cone's edge,
  no console errors; screenshots to the owner.
