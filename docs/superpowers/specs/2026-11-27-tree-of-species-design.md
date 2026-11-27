# The tree of species: design

Asked by the owner on 2026-11-27: "how many species we have at a given moment ... How can we group them into species as
time moves on, so we can see like how evolution takes place"; on splitting: "how do you decide when to split ... the
parent species could have been extinct already"; then: "let's start building the species table like Darwin, like you
see a tree of creatures and how they split into different creatures."

## What a species is (agreed in the conversation)

1. **A baby belongs to its parent's species.** The first creatures and the soup's newcomers belong to species 1, "first
   life".
2. **Each species has a typical member**: the average of its living members' genes (the 18 genes a creature shows,
   each 0-1 by its range). It moves as the species evolves; no one has to be alive from its start.
3. **Every chart point (`row_every`, 250 steps) the world looks at every species**: if its members fall into separate
   groups (two members are in one group when a chain of members, each within `species_apart` (0.10: a mean difference
   of 10 % of each gene's range) of the next, links them), every group of at least `species_group` (5) but the biggest
   becomes a new species whose parent is that species; the biggest keeps the name; a smaller group stays (mutants, not a
   species). (Changed while building, 2026-11-27: "far from the typical member" failed when a drifted group was large
   enough to pull the typical member toward itself; groups apart is the agreed point 4 itself.)
4. **Slow drift is not a split**: a whole species changing together keeps its name.
5. **A species with no living member has ended** (its place in the tree stays).
6. `species_apart` and `species_group` are rules on the Rules page (for the world's next looks).

## What is kept

The species book of a world: each species' number, its parent, the step it began and ended, its typical member's
genes, and how many were alive at each chart point. Kept in saves and written beside the scoreboard (`species.json`) for
the page. A save of before: every creature in species 1.

## The page: the tree

A **Species** panel under a world's charts (in Live and History):
- **The tree**: time from left to right; each species a horizontal band from the step it began to the step it ended
  (or now), as thick as the number alive, in its own colour; a thin line from the parent's band at the step it split
  off. Like Darwin's "I think" sketch, lying on its side.
- **The table** beside it: each living species, how many, and its typical member in words (grazer or hunter, size,
  swimmer or walker, armour, eye sharpness, brain size), and its parent.
- **Clicking a species** (its band or its row) opens its card (the owner: "once I click a node in the tree, I should see
  their stats ... all the interesting information about the species"): how many are alive now and at most, when it
  began and ended, its parent and its daughter species; and its typical member in plain words with the number beside
  each: a hunter or a grazer (diet), aquatic or a walker (water), size, armour, stealth, eyesight (sharpness, how wide,
  how far), hearing, calls, care for kin, fear of death, love of babies, learning from others and learning speed, brain
  (layers and neurons), how much its babies change; and how its typical member changed since it began (the genes that
  moved most). The owner, then: "How smart a species is, how big of a brain ... all the cool, interesting biological
  facts. Size also interesting." So the card also shows, at the last look:
  - **how smart**: its members' energy earned ÷ energy spent (their own intelligence, as the exam counts it) and the
    share of their moves that steer to food they see;
  - **its brain**: layers and neurons, smallest to largest;
  - **its body**: size (smallest, typical, largest), its energy tank, how fast it runs;
  - **its lives**: how old they are (typical, oldest), how many generations deep, how many babies born, and how its
    members died (hunger, hunted, old age).
  The book keeps, for each species, its typical member at its start and now, its largest count, and these facts at the
  last look; births and deaths are counted as they happen.

## Not in this

- Colouring creatures on the planet by species (a later step); any change to how creatures live, breed or decide.

## The one-minute arithmetic

A look every 250 steps: for each species, its members' genes (a 600 × 18 table, already made for the senses) against
its typical member, then the far ones against each other (a few dozen at most): well under a millisecond a step on
average.

## How it is proven

- Tests: a baby is in its parent's species; 5 drifted creatures close together split off with the right parent; 1
  drifted creature does not; a species drifting together does not split; a species with no one left ends; the book
  survives a save; `/species?run=` serves it; the page draws the tree and the table.
- In the browser pane on a half-size world: a screenshot of the tree.
- The whole suite under 300 s.

## What could prove it wrong

Species split off and fold back every few looks (the threshold too tight), or never split in a long run (too loose).
