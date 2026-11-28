# Make a creature and place it in the world: design

Asked by the owner on 2026-11-27: "Add a page where I can play with genes and see how the creature will look like and
spawn it if I wish in map where ever I click". Asked what brain a made creature gets (a new empty one, the smartest
alive's, or a copy of one the owner picks): "All 3 options".

## What the owner sees

- **A new tab, "Make"**: a slider for every gene a creature has (the genes the page can already change on a followed
  creature: size, the two guts, gills and lungs, eyes, hearing, feelings, calls, kin, stealth, armour, brain layers and
  neurons, learning speed, mutations), each between the world's own limits, with its plain meaning on hover as on the
  Rules page. Beside them, the body those genes make, drawn exactly as the planet draws a creature, changing as a
  slider moves.
- **Start from**: the sliders start at the typical member of the running world's biggest species; a "Start from"
  list sets them to any species' typical member.
- **Brain**, three choices:
  1. a new, empty brain (learns from zero, like a newcomer of the soup);
  2. a copy of the brain of the smartest creature alive now (the one that earns the most for each energy it spends);
  3. a copy of the brain of the creature the owner follows in the live view.
  With 2 or 3 the brain's size (layers, neurons) is that brain's: those sliders show it and are locked, because a brain
  only fits a body of its own size.
- **Place it**: switches to Live, into the planet view; the next click on the ground places the creature there (on a
  rock or in a pond it cannot live in, it is refused, said in plain words). It can be placed again with more clicks
  until Esc or "Done".

## What it is in the world

- A made creature is like a newcomer of the soup: the energy a newcomer gets, a lifespan by its size, generation 0, in
  the species "first life" with a new brain, or in the species of the creature whose brain it copies. From then on it
  lives by the world's rules: it eats, has babies and dies like any other. Nothing else in the world changes.
- It is not counted in the world's newcomer count: Auto reads that count as "the world nearly died out" and eases the
  world, and a made creature is not that. An event in the world's report says it was placed, where, with which genes
  and brain.

## How it is built

- The page (`src/viewer.html`): the "Make" tab; its sliders built from the gene limits the page already reads
  (`/rules`), the picture by `speciesPicture`'s body maker (`src/static/beasts.js`, one body maker); "Place it" uses
  the planet's own click on the ground (`src/static/planet.js`, which already turns a click into a creature to follow;
  it now also gives the spot on the ground).
- The page server (`src/viewer.py`, `/control`): `place: {spot, genes, brain}` checked by `check_genes` (as an edit
  of a creature is) and written for the world, each once (as an edit is).
- The world (`src/world.py`): `place(spot, genes, brain)`: a creature as `add_newcomer` makes one, at that spot, with
  those genes; its brain new (`learner.newcomer`) or a copy (`learner.baby` of that creature's slot with nothing
  changed); refused when the spot is blocked for it or the room is full.

## Not in this

- No change to any rule of the world, to births, to Auto, or to how a newcomer of the soup is made.
- No placing in History's ended worlds.

## How it is proven

- Tests (small hand-made worlds, few steps): a placed creature is where it was placed with exactly the given genes;
  each of the three brains (new; the smartest's, equal weights; the followed one's, equal weights); a blocked spot and a
  full room refused; a gene outside its limits refused by the server; the newcomer count unchanged; the event written.
- On a half-size check world on this box: move sliders, the body changes; place one by a click; it is there in the
  view; screenshot; the browser pane closed.
- The whole suite under 300 s.
