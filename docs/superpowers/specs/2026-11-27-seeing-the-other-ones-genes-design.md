# Seeing another creature's genes, and knowing who hit it: design

Asked by the owner on 2026-11-27: "when creatures see other creatures, especially even when getting hit, they should
know exactly what they are based on their genes. Like, he should know this is a dangerous creature because he hit me,
and then he should learn that this is the genetic makeups that is scary ... They should see all the DNA of the other
creature, basically. And when they are hit, they should know exactly what hit them and where and from, like, not just
left, right, up, down."

## Today

Of the nearest creature it sees, a creature senses: seen, distance, where (ahead, left), its size against its own, and
how alike their genes are (one number). When hit, for one step: hit, the share of its tank taken, where the hitter is
(ahead, left). It never senses who the hitter is or how far.

## The senses change, in one line

A creature sees all 18 genes of the nearest creature it sees; when hit, for the step after, it senses the hitter's 18
genes and its distance as well as its direction.

## What changes

1. **The one it sees**: 18 more inputs, each of its genes scaled 0-1 by the gene's range (as the voice of a call is
   today): size, diet, water, stealth, armour, vision width, vision reach, hearing, calls, kin, fear of death, love of
   babies, learns from others, learning speed (on a log scale), genes a baby changes, how much each changes, neurons,
   layers. Zeros when it sees no one.
2. **The one that hit it**, for the step after the hit (as pain is today): its distance (÷ its vision reach) and its
   18 genes, beside today's hit, share taken and direction. Zeros when not hit.
3. **The brain grows by 37 inputs** (47 to 84). Every brain alive, in every world and save, keeps what it learned: its
   new inputs start with zero weights, so it acts exactly as before until it learns to use them (by learning in life
   and by its babies' mutations).
4. Nothing else: no change to what is seen (stealth, cone, rocks, trees hide as today), to costs or to learning.

## The one-minute arithmetic

A learner already learns from 47 inputs; 37 more is less than double. A creature hit by a hunter loses energy (a loss
it learns from) on the step it then senses the hitter's genes: the very pairing "these genes, this loss" it needs.

## How it is proven

- Tests: the seen one's 18 genes are in the senses, scaled 0-1, zeros when no one is seen; after a hit, the hitter's
  genes and distance are sensed for one step and gone after; a brain of before (47 inputs) and a learner's saved state
  of before load into 84 inputs and give exactly the same choices on the same senses (new weights zero).
- The whole suite under 300 s.

## What could prove it wrong

Over a long run, prey still do not move away from hunters more than from plant eaters (the "Flees bigger" and hunting
charts): the genes are sensed but not learned from.
