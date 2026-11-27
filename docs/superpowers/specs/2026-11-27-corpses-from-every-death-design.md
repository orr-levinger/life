# Every death leaves a corpse

**The owner, 2026-11-27:** "is this only when creatures died from hits? it should be the same if they die of hunger or
old age". Asked "Every creature that dies leaves a corpse worth 8 × its size, whatever killed it?": "Yes for the
corpse".

## Today

Only a creature killed by a hit leaves a corpse (`src/creature.py:307-312`, in `_attack`): meat food worth
`corpse_energy` (8) × its size, lasting `corpse_steps` (5) steps, where it died. A creature that dies of hunger or of
old age (`src/world.py:277-280`) leaves nothing.

## The change

The corpse is made in one place, `World.kill`, so every death leaves one: hunger, old age and a hit alike. Same food,
same size rule (8 × size), same 5 steps, same spot, meat. `_attack` no longer makes its own. Nobody dies in an exam
(`World.evolving` false), so exams are unchanged.

The owner, approving: "whoever eats it gets energy best if it's carnivore or not ... It's not becoming a food that grass eaters can eat. It's on the range of the genes." Already so: a corpse is meat, and an eater gets bite × its diet (`src/creature.py:250`; `tests/test_diet.py` checks it).

## The arithmetic (measured on the local half-size world `runs/20261126-203658-1-learning`, its last 20 rows of
`scoreboard.csv`, steps 59,500-64,250, 600 alive)

- Deaths not from hits: 23.2 per 250-step row = 0.093 per step. Each leaves 4-6 food: about 0.5 food energy per step.
- New plant food there: 20 pieces × 100 energy = 2,000 energy per step. The corpses add about 0.02 % to it there.
- On a hard world (the easy-to-hard ends: food per step 0.2, energy 3 → 0.6 per step) the same deaths would add about as
  much as the plants: meat becomes a real food for meat eaters when plants are scarce. Not measured on the Mac.
- No free energy loop: a creature exists only from its parent's energy or the soup; dying gives back less than it took
  to make (a baby's share and its living).

## Not in this

- No change to the corpse's worth, its length or the hunter's share. No new rule.

## How it is proven

- A hand-made small world: a creature at 0 energy dies of hunger → one meat food of 8 × its size at its spot, 5 steps.
- The same for old age.
- The existing hit test (`test_kill_leaves_corpse_and_counts_attack_death`) still passes: exactly one corpse.
- `python scripts/run_tests.py` → OK under 300 s.
