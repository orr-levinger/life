# Eyes you can see, and Make that keeps up: design

Asked by the owner on 2026-11-28, with a picture of the Make tab (a hunter at "Sees how far" 10, its highest): "THE
EYES eveer grow enogu hwhen see how far is at the exreams! what.. predetor eyes are even none seen at all"; "i made
eyes range to max 30!" (the Make slider still stopped at 10); then "fix it all locally so i can test the "Make" here
before pulling on the mac"; then "the animal image in the species too small!!".

Nothing in the world changes: no rule, no file the world writes. Only the page and the body maker.

## 1. Eyes that grow with how far it sees

- Today an eye's radius is 0.035 + 0.12 × reach (reach 0..1 over vision reach 1 to 30), small beside the head at any
  reach. New: the eye grows with the head and with reach, from a small eye to about half the head's radius at reach
  30 (`headR × (0.15 + 0.45 × reach)`), so a far-seeing creature has big eyes.
- A hunter's eyes sit today low and forward, inside its jaws and brow ridge, so they cannot be seen. New: every
  creature's eyes sit on the upper half of the head, in front of the brow, so a hunter's show; vision width still
  turns them from forward (a hunter's) to the sides (a grazer's).

## 2. Make keeps up with the world's rules

Today Make reads the world's gene limits once, when it is first opened; a rule changed afterwards (for example vision
reach's highest raised from 10 to 30) does not reach its sliders until the page is reloaded. New: each time Make is
shown, it reads the limits again; the sliders take the new limits, and the owner's slider positions stay (moved inside
the new limits if they fall outside).

## 3. Species pictures that fill their frame

The owner, 2026-11-28, with a picture of the Species page: "the animal image in the species too small!!". Today a
species' picture is drawn with the camera at a fixed distance (so that size shows), and a small creature (size 0.56)
fills only a corner of it: on the card, in the tree's circles and in "Alive now". New:
- A species' picture is framed close: the camera fits the body's bounding box, so every body fills its picture (its
  size is said in words on the card and in the table).
- The card's picture twice as big (320 px instead of 160), the table's 56 px instead of 40.
- Make keeps today's framing, so moving the size slider still shows the body growing.

## 4. Two things seen while checking

- A long Latin name in Make's "Start from" list widens the page. New: the list is as wide as its panel at most; a long
  name is cut with "…" inside the list.
- When the Species page is the first page opened and the 3D code loads after the page has drawn, the pictures are
  missing until another tab is opened. New: when the 3D code has loaded, the page draws itself again.

## Not in this

- No rule of the world changes (the owner: "actauly 30": vision reach's limits stay as they are).
- The alien look (the owner's pictures to come).

## How it is proven

- Tests (the page's files as served): the eye's radius grows with `headR` and `g.reach`; Make reads the rules each
  time it is shown; the list has a width limit; the page draws again when the 3D code has loaded; a species'
  picture is framed close and the card's is 320 px.
- The gallery's eye row (grazer and hunter, reach low, middle, high): big eyes at reach 30, a hunter's eyes seen.
- On a half-size check world on this box, left running for the owner to try Make here: a rule changed in Rules
  reaches Make's slider without a reload; no page wider than the screen; species pictures fill their frames.
- The whole suite under 300 s.
