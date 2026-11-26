# Seas, islands and rivers: design

Approved by the owner on 2026-11-26 ("About half", then "build it").

## The goal, in the owner's words

"from the images the water bodies are too small! no creature will enter water like that.. we need big sees and ilands
areas where it will get kind of hard to get to so uniqune speciens have chance to ivolve.. rivers, see, islands"

Land creatures split by water, so groups apart from each other can evolve into their own kinds.

## The one-minute arithmetic

- A creature walks about 0.25 a step and lives about 5,000 steps: about 1,250 in a life. The planet (radius 37) is about
  232 around. On land anyone can reach anywhere; only water can keep groups apart.
- A creature with the water gene at 0 can never enter water. An island at least 6 from the mainland is cut off from
  walkers for good; a swimmer crosses a strait of 30 in a few hundred steps.
- Land is half the planet: about 8,600 square units for at most 600 creatures (about 14 each).

## What changes

1. **Water covers about half the planet** (rule `water_share`, 0.5). Where water lies comes from a smooth height on the
   ball (a sum of a dozen gentle waves, from the world's seed); everything below sea level is water. Sea level is set so
   the share of water is `water_share`. This gives one big ocean, inland seas and lakes, and 2-3 continents.
2. **Islands** (rule `islands`, 8): round raised spots of mixed size (radius 2-6) placed in open water, each at least
   `island_gap` (6) from any other land.
3. **Rivers** (rule `rivers`, 3): each starts inland and runs downhill to the sea as a chain of points, `river_width`
   (1.5) wide, with `river_crossings` (2) shallow spots where it is land: walkers cross there; swimmers use rivers as
   roads.
4. **Rocks stay** as today (`obstacles`, now rocks only). **The old ponds go** for new worlds.
5. **Water is water everywhere it was a pond before**: only swimmers enter it, it hides nothing, the same share of new
   food grows in it (`pond_food_share`), shores are rich (`pond_edge_food`, `pond_edge_width`), a swimmer is slower on
   land (`land_penalty`), and the senses "in water" and "water seen ahead" read it.
6. **Newcomers land anywhere on land, islands included** (as today: a newcomer never starts in water).
7. **Fast**: "is this point water?" is a few dozen sums for one point and one array pass for many; no step gets slower
   than 5%.
8. **Saves**: a new save keeps the water (its waves, sea level, islands, rivers). A sphere save from today (ponds as
   caps) still resumes with its ponds as its water.
9. **The exam planet does not change.**
10. **The page**: seas, lakes and rivers drawn as crisp blue water (a water picture wrapped on the ball, so thin rivers
    show), islands and shores visible; still under 33 ms a frame on this box.

New rules the owner can change (Rules page, for the next new world): `water_share` 0.5 (0-0.8), `islands` 8 (0-30),
`island_gap` 6 (0-20), `rivers` 3 (0-10), `river_crossings` 2 (0-6), `river_width` 1.5 (0.5-4).

## Not in this

- No tides, currents, depth or drowning; water is just water.
- No change to how creatures sense, decide or learn, nor to any gene.
- The starving seen on new planets is a separate question.

## How each part is proven

- Water share of a new planet is 0.5 ± 0.03 (100,000 random points).
- Each island is water-locked: every point within `island_gap` of an island's edge, outside it, is water.
- Each river reaches the sea and has `river_crossings` land spots.
- A walker cannot enter water anywhere; a swimmer can.
- A sphere save of today resumes; a new save resumes with the same water.
- Step time on a new planet with 600: within 5% of before.
- Page: screenshots of the whole planet and an island close up; bench under 33 ms.

## What could prove it wrong

An island group that, over time, never ends up with genes different from the mainland.
