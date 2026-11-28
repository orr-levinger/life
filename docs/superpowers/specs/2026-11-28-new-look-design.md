# The new look: every gene shows, colour from the genes: design

Asked by the owner on 2026-11-28, looking at the creatures and the Make tab:
1. "care for keen has no affect on look"; "the changes of genes should be more drastic changes to look"; "all animal
   have same color thats borring". Choosing where colour comes from: "From its genes".
2. Then, testing on the Mac: "lungs has 0 affect on look"; "gills 1 should have 0 legs and more aquatic face";
   "neurons in layer 2 and neurons layer 2 same cahnges too look"; "grass gus 0 affect on look"; "armore 1 should be
   full body armore"; "care for kin 0 affect"; "hears how far at max ears still to small and un notecable".
3. "Approve" to this design (2026-11-28).

The mockups (round 1 and round 2, now on the left and proposed on the right, one gene per row) were sent with this
design; this document says what they show.

Nothing in the world changes: no rule, nothing a creature does or pays, no file it writes. Only the body maker
(`src/static/beasts.js`) and the pages that call it change. Each creature's brain layers' neuron counts are already in
the world's view file (`brain`, written by `src/evolve.py`; found while planning, 2026-11-28: this design first said
the world would have to add them). A species' picture has only its typical neurons in all and its typical layers, so
it shows them shared evenly over its layers.

## Colour: from the genes

- **The coat's colour** comes from what it eats and where it lives: green-gold for a plant eater, red for a meat
  eater, blue for a swimmer, and the shades in between.
- **Fear of death** makes the coat bright (warning colours, as poison frogs); **stealth** makes it dull (camouflage);
  **armour** makes it darker.
- **Markings** (the belly, the kin frill, the antennae tips, a saddle on the back) take their own colour: pink by love
  of babies, gold by care for kin, violet by learning from others, mixed by how much of each; pale when all three are
  low.
- Far away, a creature keeps its coat's colour, as today.

## Every gene shows, and clearly

| Gene | Today | New |
|---|---|---|
| Gills | a fin, shorter legs | legs shrink with the gills and are gone at 1: a fish; flippers grow in their place; no neck; a long, flat fish head with a mouth line and lips; big gill slits; ears shrink |
| Lungs | nothing | a deep, round chest (from narrow at 0 to barrel-chested at 1) and nostrils as big as the lungs |
| Grass gut | nothing | a big belly in the markings' colour, from none at 0 to a round, low belly at 1 |
| Meat gut | hunter's jaws, fangs, crouched legs | the same, and the red coat |
| Brain layer 1's neurons | head size, from all layers' sum | head size, from layer 1 only |
| Brain layers 2 and 3 | one small horn tine or crest plate each, all alike | each layer's own horn tine (a grazer) or crest plate (a hunter), as big as that layer's neurons |
| Armour | small plates down the back | bands of bone that cover more with more armour; at 1 a shell over the whole body and a helmet on the head |
| Care for kin | a thin white stripe on the face | a frill around the neck, as wide as the care, in the markings' colour |
| Hearing | small ears | ears up to as big as the head at full hearing (small on a fish) |
| Vision reach | eye size, only up to reach 10 | eye size over the whole range, 1 to 30 |
| Fear of death | quills | longer and more quills, and a bright coat |
| Love of babies | a rounder body | rounder, and pink markings |
| Learns from others | antennae | longer antennae with violet tips |
| Calls | a throat sac | a bigger throat sac |

**Finer steps.** Each gene now shows in tenths of its range instead of quarters, so a small change of a gene
changes the body.

## The picture

The body's picture (the species card, "Alive now", the tree, the Make tab) frames the whole body, so a big
creature's head is no longer cut at the edge.

## What it costs

Finer steps make more different bodies to build. Bodies with the same rounded genes still share one model, and far
away a body is still drawn in batches by its plan. The time to draw a frame on this box is measured before and after,
on a half-size world; the plan says the limit.

## Not in this

- No rule of the world changes; the genes, their ranges and their costs stay.
- The species' Latin names (their own plan).

## How it is proven

- Tests (the page's files as served): the body maker reads lungs, grass gut and each brain layer; the coat's colour
  comes from the genes (`coatOf` and `markOf`); legs are gone at gills 1; the steps are tenths; the picture frames the
  body.
- A test that the body maker reads each layer from a creature's `brain` (as the view file has it), and a species' even share.
- A gallery page beside the body maker (`/static/gallery.html`, as today) shows each gene low, middle and high: the
  owner's check.
- On a half-size check world: the herd in colours; a swimmer without legs; a frame's drawing time against before.
- The whole suite under 300 s.
