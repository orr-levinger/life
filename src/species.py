"""The tree of species (the owner, 2026-11-27: "a tree of creatures and how they split into different creatures").

A baby belongs to its parent's species; the first creatures and the soup's newcomers to species 1, "first life". At
every chart point the world looks at each species: if its members fall into separate groups (members are in one group
when a chain of members, each within `species_apart` of the next in their genes, links them), every group of at least
`species_group` but the biggest becomes a new species, the old one its parent; the biggest keeps the name. A species
with no one left has ended. The book keeps each species' parent, when it began and ended, its typical member at its
start and now, how many lived at each look, and its facts at the last look (how smart, its brain, its body, its lives).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .creature import SEEN_GENES, Creature, genes_table

FIRST = 1  # "first life": the first creatures and the soup's newcomers
MIN_SPENT = 5.0  # a creature's earned ÷ spent counts once it has spent this much (as the page shows it)
CAUSES = ("hunger", "attack", "age")
_ATTRS = {"fear": "fear_of_death", "love": "love_of_babies", "learns": "learns_from_others"}


def typical(members: Sequence[Creature], world: Any) -> Dict[str, float]:
    """Its members' genes, gene by gene: the mean (learning speed as the learner holds it)."""
    out = {}
    for gene in SEEN_GENES:
        if gene == "learning_speed":
            learner = getattr(world, "learner", None)
            values = [learner.lr(c.slot) for c in members if learner is not None and c.policy == "learner"
                      and c.slot is not None]
        else:
            values = [float(getattr(c, _ATTRS.get(gene, gene), 0.0) or 0.0) for c in members]
        out[gene] = round(float(np.mean(values)), 6) if values else 0.0
    return out


def _spread(values: List[float]) -> List[float]:
    """Smallest, typical (mean) and largest."""
    return [round(min(values), 4), round(float(np.mean(values)), 4), round(max(values), 4)] if values else [0.0, 0.0, 0.0]


def facts(members: Sequence[Creature], world: Any) -> Dict[str, Any]:
    """The interesting numbers of a species now (the owner: "How smart a species is, how big of a brain ... Size")."""
    smart = [c.earned / c.spent for c in members if c.spent >= MIN_SPENT]
    moving = [c for c in members if c.intent != "REST"]
    return {"alive": len(members),
            "smart": round(float(np.mean(smart)), 4) if smart else None,  # energy earned ÷ spent, as the exam counts
            "to_food": round(sum(c.intent == "GO_TO_FOOD" for c in moving) / len(moving), 4) if moving else None,
            "layers": _spread([c.layers for c in members if c.layers]),
            "neurons": _spread([c.neurons for c in members if c.layers]),
            "size": _spread([c.size for c in members]), "tank": round(float(np.mean([c.max_energy for c in members])), 2),
            "speed": round(float(np.mean([c.top_speed for c in members])), 4),
            "age": round(float(np.mean([c.age for c in members])), 1), "oldest": int(max(c.age for c in members)),
            "generations": int(max(c.generation for c in members))}


class SpeciesBook:
    def __init__(self) -> None:
        self.species: Dict[int, Dict[str, Any]] = {FIRST: self.new_entry(FIRST, parent=None, began=0)}
        self.next_id = FIRST + 1

    @staticmethod
    def new_entry(number: int, parent: Optional[int], began: int) -> Dict[str, Any]:
        return {"id": number, "parent": parent, "began": began, "ended": None, "start": None, "now": None,
                "largest": 0, "counts": [], "facts": None, "births": 0, "deaths": {cause: 0 for cause in CAUSES}}

    def _entry(self, number: int) -> Dict[str, Any]:
        if number not in self.species:  # a species the book has not met (a creature made by hand): it begins now
            self.species[number] = self.new_entry(number, parent=None, began=0)
            self.next_id = max(self.next_id, number + 1)
        return self.species[number]

    def born(self, number: int) -> None:
        self._entry(number)["births"] += 1

    def died(self, number: int, cause: str) -> None:
        deaths = self._entry(number)["deaths"]
        deaths[cause] = deaths.get(cause, 0) + 1

    def look(self, world: Any, step: int) -> None:
        """At a chart point: splits each species whose members have fallen into separate groups, then writes down each
        one's count, typical member and facts; a species with no one left has ended."""
        creatures = list(world.creatures)
        apart, group = world.rules["species_apart"], max(1, round(world.rules["species_group"]))
        if creatures:
            genes = genes_table(creatures, world)
            numbers = np.array([c.species for c in creatures])
            for number in sorted(set(numbers.tolist())):
                self._entry(number)
                rows = np.nonzero(numbers == number)[0]
                if len(rows) < 2 * group:
                    continue  # too few to hold two groups of a species' size
                close = np.mean(np.abs(genes[rows][:, None, :] - genes[rows][None, :, :]), axis=2) <= apart
                count, label = connected_components(csr_matrix(close), directed=False)
                sizes = np.bincount(label, minlength=count)
                big = [k for k in np.argsort(-sizes) if sizes[k] >= group]
                for k in big[1:]:  # the biggest keeps the name; every other group of a species' size is a new one
                    new = self.next_id
                    self.next_id += 1
                    self.species[new] = self.new_entry(new, parent=number, began=step)
                    for row in rows[label == k]:
                        creatures[row].species = new
        members: Dict[int, List[Creature]] = {}
        for c in creatures:
            members.setdefault(c.species, []).append(c)
        for number, entry in self.species.items():
            mine = members.get(number, [])
            if not mine:
                if entry["ended"] is None and (entry["counts"] or entry["began"] < step or number == FIRST):
                    entry["ended"] = step
                continue
            entry["now"] = typical(mine, world)
            if entry["start"] is None:
                entry["start"] = entry["now"]
            entry["counts"].append([step, len(mine)])
            entry["largest"] = max(entry["largest"], len(mine))
            entry["facts"] = facts(mine, world)

    def state(self) -> Dict[str, Any]:
        return {"next_id": self.next_id, "species": [self.species[k] for k in sorted(self.species)]}

    @staticmethod
    def from_state(state: Optional[Dict[str, Any]]) -> "SpeciesBook":
        book = SpeciesBook()
        if state:
            book.species = {entry["id"]: entry for entry in state["species"]}
            book.next_id = state["next_id"]
        return book

