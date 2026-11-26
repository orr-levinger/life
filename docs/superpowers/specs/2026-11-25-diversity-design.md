# Diversity — design

**The owner, 2026-11-25:** "I want more variety… we need to change the costs formulas and we need to add benefits to
each… how better can we simulate nature? what can we add to the game to increase diversity?" Of the ideas discussed:
"I LIKE ALL THE IDEAS"; of care for kin: "amazing idea". And: "the world need to be much bigger with much more max
creatures" — "add this to the plan".

## Why

Every cost and benefit of size grows in a straight line with size, so no size wins: in the owner's world at step
241,125 every creature's size was 0.500–0.532, the lowest the rules allow (measured from its `now.json`, genes.size).
Nothing else separates ways of life: all eat the same food, and nothing pays a creature for living in a group.

## What is decided

The world gives costs, benefits and senses; it never says who attacks or helps whom. Every new trait is a gene,
every number a rule on the Rules page.

0. **A bigger world with more creatures**: a new world is 120 × 72 (today 50 × 30, about 6× the area) with room for
   300 creatures (today 80, the most the rules allow); Auto aims at 240 alive (today 64); a new world starts with 100
   creatures, the soup keeps 40, and gets 35 obstacles and 46 trees (today 6 and 8, grown with the area). The owner's
   world keeps its own size unless the owner changes it or starts a new world.
1. **Size by curves** (each power a rule): energy held = tank × size^1.3; living cost = step cost × size^0.75; speed
   = cruising speed ÷ size^0.5; lifespan × size^0.5; a moving creature is heard only within the hearer's hearing × its
   size, seen only within the looker's reach × √its size.
2. **Diet gene** 0 (plants) – 1 (meat): energy from food and fruit × (1 − diet); from corpses and hits × diet; a bite
   hurts × (1 + diet). Senses: the nearest seen food is meat; its own diet.
3. **Fighting back** needs no new rule: anyone may attack anyone, and attackers' hits add up.
4. **Calls** (the owner, 2026-11-26: "several types of call… 1 call could mean danger 1 call could mean food";
   "number of calls is also a gene between 1 and 10"; "a call should have a volume that control the distance";
   "cost price is u are being noticed"; "lets just build it and fine tune it later"): a gene `calls` 0–10, how many
   different calls a creature can make; newcomers start at 0 and evolution turns calls on (a baby's mutation adds
   one). Each call type is its own action (those above its gene are closed); with each call the brain also chooses a
   volume 0–1, and the call carries volume × `call_reach`. Everyone within that distance hears which call it was, how
   loud it arrived, its direction and the caller's voice — the caller's diet, size, water and stealth genes, each
   scaled 0–1 (similar genes sound alike; never a hash). No energy cost: its price is being noticed (predators hear
   it too) and the step spent calling. No call has a meaning; meanings are left to evolution.
5. **Care for kin** gene 0–1: a share of what happens to the creatures it perceives (their change in energy) counts in
   its own reward, scaled by how similar their genes are. The fixed rules "family can never be attacked" and "family
   is not seen as another creature" go. Sense: how similar the nearest seen creature is.
6. **Water gene** 0–1: a pond can be entered with water above 0; speed in water × water, on land × (1 − water × the
   land penalty); some new food grows in ponds; ponds stop blocking sight (rocks still do). Senses: in water; the
   nearest obstacle in view is water.
7. **Stealth gene** 0–1: seen and heard from (1 − stealth × 0.7) as far; living cost × (1 + stealth × 0.3).
8. **Armour gene** 0–1: a hit takes × (1 − armour × 0.7); speed × (1 − armour × 0.3); living cost × (1 + armour × 0.3).
9. **Seasons**: food and fruit per step rise and fall over a year of steps (length and depth are rules).
10. **Regions**: trees grow together in forests with plains between; pond edges grow more food.

## How it is checked

Each trait is built on its own: tests first, then the code, the whole suite, the owner's world updated, and the
trait's gene tile read after 20,000 steps (lowest, average, highest).

## Out of scope

The exam stays a fixed world (its scores compare). Nothing not listed above.
