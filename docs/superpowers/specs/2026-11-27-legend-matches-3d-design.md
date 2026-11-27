# The legend matches the 3D planet: design

Asked by the owner on 2026-11-27: "add index as to what each animation means and what creture looks means like we had
before just match it to 3d.." and "the green apple for eating is a bit weired and too big it hides all the creature".

## What changes (the page only)

1. **The legend's Body rows show the 3D bodies** the planet draws (the gallery's look, the owner's approved one), one
   small picture per sign with its words: plant eater · meat eater · swimmer; size; a rounder body (loves babies);
   quills (fears death); horn tines or a crest (brain layers); head size (neurons); antennae (learns from others);
   bands on the coat (learns fast); eyes to the sides (wide vision) and big eyes (sees far); ears (hearing); bone plates
   (armour); a duller coat (stealth); a throat sac (calls); a pale blaze (cares for kin). The 2D pictures of before go.
2. **The Rings and Doing rows stay** (the planet's own marks and rings), and gain "cone: what it sees" as the planet
   draws it.
3. **The marks are smaller and sit above the head**, never over the body: about half today's size (still never under
   a few pixels), and the eating mark becomes a small green bite mark (an open "C", not a filled disc).

## Not in this

- No change to the world, the bodies themselves, or the Above and Eyes views.

## How it is proven

- A test reads the page: the legend's Body row draws `beastModel` bodies (an exported `bodyPicture`), names each sign,
  and has no 2D creature pictures left; the marks' size rule and the new eating mark are in planet.js.
- In the browser pane on a half-size world: a screenshot of the legend open and of marks over creatures; `bench(30)`
  under 33 ms; the browser closed after.
