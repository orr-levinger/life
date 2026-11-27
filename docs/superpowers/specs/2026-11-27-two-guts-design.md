# Two guts, gills and lungs: specialists win, the road between stays open

## The goal, in the owner's words (2026-11-27)

- "We need to try to pull the genes to edges. ... If you eat only meat, you should get relatively more energy from
  only meat. ... Same for water. If you are only land or only water, you should be more adapted to this environment
  than if it's linearly half-half."
- The trap they saw in a curve: "if they're all [grass eaters] right now, they will never become predators because it
  doesn't worth it. They will not do the slow transitions."
- Nature: "if you have gills, you can['t] breathe out of water. You cannot have both. Like it's a choice". Answer
  agreed: nature ends this-or-that through competition, and the road there goes through creatures with both (lungfish,
  mudskippers, omnivores).
- The running world must go on: "how will we make the world that's already running still work after this change?"
  Agreed: each creature's new genes come from its old ones; the new costs start at 0 and Auto turns them up.
- Approved to write this: "Go".

## Today

- One gene `diet` (0 plants, 1 meat): a plant bite gives × (1 - diet), a meat bite × diet; a hit takes × (1 + diet ×
  `diet_bite`) and feeds the hitter × diet. Straight lines: a 50/50 creature gets exactly half of each, and nothing
  makes the middle worse than the ends.
- One gene `water` (0 land, 1 swimmer): in a pond it moves at × water and pays × (1 - water × `swim_saving`); on land
  it moves at × (1 - water × `land_penalty`) and pays × (1 + water × `land_cost`); water 0 cannot enter a pond.

## The change (each one a rule of the world)

**`diet` becomes two genes, each 0-1: `grass_gut` and `meat_gut`.**
- A plant or fruit bite gives × `grass_gut`; a meat bite (corpse) × `meat_gut`.
- A hit takes × (1 + `meat_gut` × `diet_bite`) and feeds the hitter × `meat_gut` (`hunter_share` as today).

**`water` becomes two genes, each 0-1: `gills` and `lungs`.**
- In a pond: moves at × `gills`, pays × (1 - `gills` × `swim_saving`); `gills` 0 cannot enter a pond.
- On land: moves at × (1 - (1 - `lungs`) × `land_penalty`), pays × (1 + (1 - `lungs`) × `land_cost`).

**Each organ costs upkeep, two new costs** (in the COSTS table: 0 up to their natural value; Auto turns them, the owner
may set them on the Rules page):
- `gut_cost`: living × (1 + (`grass_gut` + `meat_gut`) × this). Natural value 0.15.
- `breath_cost`: living × (1 + (`gills` + `lungs`) × this). Natural value 0.15.
- Why 0.15: a mammal's gut uses a large share of its resting energy (often quoted at 10-20 %; not measured here) and the
  same price is given to breathing organs; stealth and armour cost 0.3 each, for comparison.

**Why this pulls to the edges without a trap.** Each organ's worth is a straight line: a little more meat gut costs a
little more upkeep and gives a little more of every meat bite. So an organ whose food pays more than its upkeep grows to
1, and one whose food pays less shrinks to 0: the ends win. A grazer becomes a hunter in small steps: its meat gut
grows while it is still a full grazer (it pays a little more, and eats more), and once meat feeds it well enough, its
grass gut shrinks to save its price. No step on the way is a loss. Where both foods are plenty (a mixed place, a
shore), having both can win: omnivores and mudskippers.

## The one-minute arithmetic (measured on the local half-size world `runs/20261126-203658-1-learning`, last 20 rows
of `scoreboard.csv`: `bites_per_creature_step` 0.178, non-hit deaths 0.093 per step, 600 alive)

- Living for a size-1 creature: `step_cost` 0.1 per step. One full gut at 0.15: 0.015 per step.
- Grass gut: about 0.18 bites per step → ~0.18 energy per step; its 0.015 is ~8 % of that: it pays easily.
- Meat gut, from corpses alone: 0.093 deaths × ~5 food = ~0.5 energy per step for 600 creatures = ~0.0008 each:
  far less than 0.015. Corpses alone will not grow a meat gut there.
- Meat gut, from hunting: a size-1 hunter with meat gut 1 keeps 0.8 × 10 = 8 energy per hit; one hit in ~530 steps
  pays its gut. So a meat gut grows where hunting works, and shrinks where it does not.
- On a hard world (scarce plants) grass pays less and meat relatively more. Not measured on the Mac.

## The running world (generation 30K on the Mac) goes on

- **Genes**: a save from before is converted as it is loaded, and saved in the new shape: `meat_gut` = diet,
  `grass_gut` = 1 - diet, `gills` = water, `lungs` = 1 - water. Each creature then eats, moves and pays exactly as
  before.
- **Costs**: an old save has no `gut_cost` or `breath_cost`: both start at 0 (as `pixel_cost` did), so nobody pays
  more on the first step; Auto turns them up toward 0.15 like every cost.
- **Brains**: creatures see the 18 genes of the one they see and of the one that hit them. `meat_gut` and `gills` take
  the places of diet and water (the same values: what a brain learned still means the same); `grass_gut` and `lungs`
  of both are 4 new senses at the end (196 → 200), whose weights start at 0 in every brain of before, as with the
  pixels and ears. Calls and the eye's pixels show `meat_gut` where they showed diet.
- **The page**: the coat colour, the gene tiles, the species card and the Rules page show the four genes (rust for
  meat gut, teal for gills, as diet and water today).

## Not in this

- No change to bites, the corpse, the hunter's share or Auto's way of choosing. Size, stealth, armour, eyes and hearing
  stay amounts with their own price (no curve).

## How it is proven (small hand-made worlds, a step or two)

1. A grass gut of 0.5 gets 0.5 of a plant bite; a meat gut of 0.5 gets 0.5 of a corpse bite; both 1 get the full bite
   of each.
2. A hit takes and feeds by `meat_gut`.
3. In a pond: speed × gills and cost × (1 - gills × saving); gills 0 cannot enter. On land: speed and cost by (1 -
   lungs).
4. Upkeep: living × (1 + guts × `gut_cost`) × (1 + gills and lungs × `breath_cost`).
5. **The step after the update:** a save of before (diet 0.3, water 0.6) loads as meat 0.3, grass 0.7, gills 0.6,
   lungs 0.4, both new costs 0; its energy after one step equals what today's code gives it.
6. A brain of 196 senses loads with 200, the 4 new rows at 0; its choices on the same senses are unchanged.
7. A baby inherits each of the four genes with its own mutation.
8. `python scripts/run_tests.py` → OK under 300 s.
