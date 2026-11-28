# Latin names for species: design

Asked by the owner on 2026-11-28: "Species 16 is a boring name. The system should generate cool names based on the
genes and use some common way to name animals… Just not species number." Then, choosing between Latin and English
names: "Latin". Then: "Names should be generated only when a new species is created." Then "Approve" to this design.

Nothing in the world changes: no rule, no save, no file the world writes. Only the page changes.

## When a species gets its name

- A species is named from its genes at its birth: its typical member at its first look (`start` in `species.json`,
  written once when the species is born and never changed). Its name therefore never changes, however its genes
  drift later.
- The name is worked out by the page from the tree of species the world already writes, so every world has names at
  once: the running one, the ones in History, and the Mac's world after a page reload. Nothing has to be copied or
  converted.
- A name is never given twice in one world. Species are named in the order they were born (by their number); a later
  species never changes an earlier one's name.

## How the name is made: two Latin words

The first word (the genus) says its body plan; the second (the species name) says what sets it apart.

**The first word: what it eats + where it lives and how big it is.**

| What it eats (as the card says it: grass gut against meat gut) | Start |
|---|---|
| plants | Phyto |
| meat | Sarco |
| both | Panto |

| Where it lives (as the card says it: gills against lungs) | Ending |
|---|---|
| land, size under 0.9 | mys (a mouse) |
| land, size 0.9 to 1.4 | therium (a beast) |
| land, size over 1.4 | saurus (a lizard) |
| water, size under 1.2 | ichthys (a fish) |
| water, size 1.2 or more | cetus (a whale) |
| land and water | batrachus (a frog) |

For example: *Phytotherium*, *Sarcoichthys*, *Pantobatrachus*.

**The second word: the gene it differs most in from its parent species at its parent's birth.** Each gene is measured
as a share of its whole range (learning speed on a log scale, as the Make slider). "First life" has no parent: its
genes are measured from the middle of each range. A gene that went up gives its high word; one that went down gives
its low word.

| Gene | Up | Down |
|---|---|---|
| size | giganteus | pumilus |
| stealth | occultus | conspicuus |
| armour | loricatus | nudus |
| vision width | circumspectus | angustus |
| vision reach | oculatus | myops |
| hearing | auritus | surdus |
| calls | vocalis | tacitus |
| care for kin | collaris | solitarius |
| fear of death | spinosus | audax |
| love of babies | pius | incuriosus |
| learns from others | docilis | indocilis |
| learning speed | velox | lentus |
| mutations | mutabilis | constans |
| mutation size | varius | fidelis |
| neurons | cerebrosus | simplex |
| layers | profundus | planus |

For example: *Sarcosaurus spinosus* (a big meat eater whose fear of death rose most), *Phytomys auritus* (a small plant
eater whose hearing rose most).

**Never twice.** When the name is already taken in this world, the gene it differs second most in gives the second
word, then the third, and so on; if every one is taken, the name gets a number: *Phytotherium auritus II*.

The words are Latin roots in the way animals are named; the grammar of the two words is not matched (a Latin endings
rule), which the owner did not ask for.

## Where the name shows

Everywhere the page now says "species 16" or "first life": the tree of life (under each circle), the species card's
title and its "from" and "daughters", the "Alive now" table and its "from" column, and the Make tab's "Start from"
list. The name is written in italics, as Latin names are. The card says once, in small grey words, "number 16 in this
world", so a species can still be matched to the world's events.

## One place

The names are made in one function on the page (`latinName`), next to `lives()` and `eats()`, which already say land
or water and what it eats, so the name and the card always agree. One table on the page holds, for each of the 16
genes, its two words and the range it is measured by. The ranges are fixed: the world's starting ranges of today
(for example size 0.5 to 2, hearing 0 to 4, fear of death 0.1 to 100, from `starting_rules()`). A world's own ranges
move as it gets harder, so measuring by them would change old names; fixed ranges keep every name as it was given.

## Not in this

- No rule of the world, no save and no file the world writes changes.
- English names (the owner chose Latin).

## How it is proven

- Tests (the page's files as served): `latinName` exists, every species name on the page comes from it (no
  "species ${" and no "first life" words left), each of the 16 genes has both words, the six endings and three starts
  are there.
- A test that runs the naming on a small hand-made tree of species (a first life, two daughters with set genes, one
  daughter that would get a name already taken) and checks the four names, including the next gene's word for the
  taken one.
- On a half-size check world on this box: the tree, the card, "Alive now" and Make show Latin names; a screenshot;
  the browser pane closed.
- The whole suite under 300 s.
