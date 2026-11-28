# Land or water, the tree of life, extinct species back, the ring above water: design

Asked by the owner on 2026-11-28, four things, then "Do it" for all of them:
1. Looking at the species card: "no info on land vs land animal"; then, on measuring time in water: "Land or water
   is genes based not activity based", then "Actually also activity".
2. With a picture of a poster of the evolutionary tree of life (a trunk from one root at the bottom, branches splitting
   upward, each kind a circle with its picture, each big branch its own colour): "the tree of life should look like
   this".
3. "also i want to see instinct cretures and able to bring a instikt creature back if i want" (extinct species).
4. With a picture of a creature in a lake: "the circle in water invosible we need to bring it a bit higher" (the hunger
   ring under each creature sinks below the water's surface).

None of the four changes a rule of the world.

## 1. Land or water animal: by its genes, and where it really lives

The owner, 2026-11-28: "Land or water is genes based not activity based". It comes from the gills and lungs, as the
card's "aquatic" already does (the page's `lives()`: water animal, land animal or both).
- The tree shows it on every circle: a blue rim for a water animal, a brown rim for a land animal, half and half for
  both; its words under the circle with the species' name ("water", "land", "both").
- The card says it first, in plain words: "a water animal (gills 0.76, lungs 0.24)" instead of only "aquatic"; "Alive
  now" the same words.
- And where it really lives (the owner, 2026-11-28: "Actually also activity"): each creature counts the steps it spends
  in water (a pond, lake, sea or river: where the world already makes its gills count). The card adds it after the
  genes, for example: "a water animal (gills 0.76, lungs 0.24); in water 38 % of the time" (its members' steps in water over all
  their steps, at the last look), and "Alive now" the same share. A save from before has no count: its creatures start
  counting from 0 when loaded. Counting changes nothing a creature does or pays.

## 2. The tree of life

- The Species page's tree is drawn as the poster: one root at the bottom ("first life"), each species a circle with
  its body inside (the same picture as its card: its typical member, as the planet draws it), and a branch from its
  parent's circle to it. Daughters fan out above their parent; each daughter of the first life gives its branch a colour
  that its own daughters keep.
- A circle's size shows how many are alive (an ended species keeps a small grey circle, its picture faded, marked †).
- Clicking a circle opens its card, as a band does today. The card keeps every number it shows now (when it began and
  ended, how many at most, ...), so nothing the time bands told is lost.

## 3. Extinct species, and bringing one back

- Extinct species stay in the tree (grey, †) and on their card ("lived from step A to step B (ended)", their last
  typical member's picture).
- An extinct species' card has a button, "Bring it back": it opens the Make tab with the sliders at that species' last
  typical member. Its brain can only be new (the brains died with them), said in plain words. Then "Place it" as for any
  made creature. The Make tab's "Start from" list also lists the extinct species, marked "(extinct)".
- A creature brought back starts a new line in the species "first life" (the old species stays ended: it is history, as
  it happened).

## 4. The ring above the water

- The hunger ring under each creature, the call rings and the follow ring are drawn just above the water's surface
  (today, by `src/static/planet.js` and `beasts.js`: the water is drawn 0.03 above the ground, the
  hunger ring 0.024 × the body's radius above it, about 0.005 for a size-1 creature, so under the water), so they
  show on land and in water alike.

## Not in this

- No rule of the world changes; births, deaths, species splitting and Auto stay as they are.

## How it is proven

- Tests (small hand-made worlds, few steps): a creature in a pond counts its steps in water, one on land does not; the
  species facts give the share; a save from before loads with the count at 0.
- Tests (the page's files as served): the tree draws circles with `Planet.speciesPicture(` and a rim by `lives(`; the
  card says "water animal", "land animal" or "both"; an ended species has "Bring it back"; the Make tab lists extinct
  species; the rings' lift is above the water's.
- On a half-size check world on this box: the tree as circles with pictures (several species: the small world splits
  them early); an ended species brought back and placed; a creature in a lake shows its ring; screenshots; the browser
  pane closed.
- The whole suite under 300 s.
