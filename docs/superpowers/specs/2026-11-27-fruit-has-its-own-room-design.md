# Fruit has its own room: design

Asked by the owner on 2026-11-27: "also why fruits are so rare?", then "trees should have there own seperate limit and
not depend on plants".

## Why (from the code)

Fruit and plant food share one limit, "Food pieces at most" (`max_food`). Each step plant food is placed first and
fills the room; a tree finds none left and drops nothing.

## The world-rule change, in one line

Fruit has its own limit, "Fruit pieces at most" (rule `max_fruit`, 300, 0-3000, on the Rules page), and no longer
counts against "Food pieces at most", which counts plants and meat only.

## What changes

1. A new world rule `max_fruit` (300, 0 to 3000) with its plain text; the owner changes it on the Rules page, the
   running world takes it within a second.
2. A tree drops fruit while the fruit on the planet is fewer than `max_fruit`; plant food grows while plants and meat
   are fewer than `max_food`. A corpse (meat) is placed as today.
3. Auto still turns the fruit rate (`fruit_rate`) as today; it does not turn `max_fruit`.
4. A world saved before gets `max_fruit` at 300, as every newer rule does.

## How it is proven

- Tests: with `max_food` full of plants, trees still drop fruit (up to `max_fruit`); with `max_fruit` full, no more
  fruit but plants still grow; the rule's start, range and text.
- The whole suite under 300 s.
