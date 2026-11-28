# Random newcomers while few species live: design

Asked by the owner on 2026-11-28: "new random creatures should spawn randomly when number of species is below 10 and
the total count is under the max"; told that today newcomers come every step while fewer than 80 live, and offered one
every 50 steps while fewer than 10 species live: "if fewwer then 80 or less then 10 unique species".

**The world rule change (the owner's, quoted above):** random newcomers also arrive while fewer than 10 species are
alive, one every 50 steps, as long as there is room. Today's soup stays: one every step while fewer than 80 live.

## The rule

- **Today (kept):** while fewer than `soup` (80) creatures live, a random newcomer arrives every step.
- **New:** while fewer than `soup_species` (10) species are alive and fewer than `room` (600) creatures live, a random
  newcomer arrives every `soup_species_every` (50) steps.
- The species alive are counted at each chart point (the tree of species looks every chart point, about every 250
  steps); between two looks the last count holds. Before the first look a world has no count yet: it counts as fewer.
- A newcomer is made as the soup makes one: random genes, a new brain, in "first life"; its own offspring can split
  off as a new species when a group of them forms (`species_group`, 5).
- **Not "nearly died out":** Auto reads the soup's newcomers as a sign the world is dying out and makes it easier.
  These newcomers come because species are few, not because creatures are; they are counted apart (`species_newcomers`)
  and Auto does not read them.

## The one-minute arithmetic

- A full world (600) gets none: there is no room. A world at 550 with 3 species gets 1 every 50 steps: 20 per 1,000
  steps, against the hundreds of babies born in that time (the owner's world: about 240 births a 1,000 steps), so
  babies keep almost every free place and the evolved families are not swamped.
- A group of 5 kin is needed for a species: a lone newcomer becomes one only through its babies. So new species come
  slowly, from newcomers that live well enough to have young.

## Not in this

- How many creatures a new world starts with (asked separately: set in Rules, or a new starting value).

## How it is proven

- Tests (small hand-made worlds, few steps): with 3 species alive and room, a newcomer every 50 steps and none between;
  with 10 species, none; at the room, none; the soup's newcomers still come below 80 as today; these newcomers not in
  `newcomers` (what Auto reads) but in `species_newcomers`.
- The whole suite under 300 s.
