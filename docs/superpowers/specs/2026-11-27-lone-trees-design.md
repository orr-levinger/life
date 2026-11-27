# Lone trees all over the land: design

Asked by the owner on 2026-11-27: "huge areas with no trees there should be some trees all around just not in water".
The rule, approved: "Yes, add it" (to "besides the forests, 40 lone trees grow anywhere on land, never in water").

## The rule

A new world rule `lone_trees`, 40 (0-200), on the Rules page with its plain text: besides the trees of its forests, a new
planet gets this many single trees at random places anywhere on its land. Like every tree: never in water, never on a
rock, never within two crowns of another tree (a place that does not fit is tried again, at most `PLACE_TRIES` times;
then that tree is not planted). Each drops fruit as every tree does.

Only new worlds get them (trees are made with the planet); a resumed world keeps its trees.

## How it is proven

- A test: a planet made with 2 forests and `lone_trees` 40 has at least 30 trees away from every forest, none in water,
  none on a rock.
- A test: the rule's start, range and text.
- The whole suite under 300 s.
