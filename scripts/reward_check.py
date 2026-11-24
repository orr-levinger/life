"""The fair check of a learning change: the real runner, with Auto, resumed from a copy of a world's save (never the
world's own folder), with today's learning or the new one, and its own learner seed.

    python scripts/reward_check.py runs/<run> --save 156000 --until 164000 --learning own|shared --seed 11 --out <folder>

Afterwards it prints alive every 1,000 steps, fleeing and chasing over the last 4,000 steps (the scoreboard's shares), and
the average fear of death, love of babies and learning from others at the end. Watch it live with
`python -m src.viewer --runs <the folder's parent> --port 8766`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.evolve as evolve  # noqa: E402
import src.world as world_module  # noqa: E402
from src.saves import save_path  # noqa: E402
from src.scoreboard import numbers, read_rows  # noqa: E402

LAST = 4000


def own_learning() -> None:
    """Today's learning: a step's reward is its change in energy alone, and nobody learns from others."""
    world_module.World._valued = staticmethod(lambda valuer, change, died, baby: change)
    world_module.World._show = lambda self, acting, events: None


def copy_of(run: Path, step: int, out: Path) -> Path:
    """The save and what a resume reads beside it, copied into a folder of the check's own; returns the copied save."""
    source = out.with_name(out.name + "-source")
    (source / "saves").mkdir(parents=True)
    for name in ("run.json", "scoreboard.csv"):
        shutil.copy2(run / name, source / name)
    return Path(shutil.copy2(save_path(run, step), save_path(source, step)))


def seeded(seed: int):
    """The loader, with the learner's dice reseeded: a save restores them, so every seed would otherwise run alike."""
    load = evolve.load_world

    def load_world(path, learner_seed):
        world, dial, extra = load(path, learner_seed=seed)
        world.learner._gen.reset_from_seed(seed)
        return world, dial, extra

    return load_world


def report(out: Path, after: int) -> None:
    rows = [row for row in read_rows(out / "scoreboard.csv") if int(float(row["step"])) > after]
    alive = [f"{int(float(r['step'])) // 1000}k:{r['population']}" for r in rows if int(float(r["step"])) % 1000 == 0]
    print("== alive " + " ".join(alive))
    last = [r for r in rows if int(float(r["step"])) > int(float(rows[-1]["step"])) - LAST]
    mean = lambda key: sum(numbers(last, key)) / max(1, len(numbers(last, key)))
    print(f"== fleeing {mean('flee'):.0%} · chasing {mean('chase'):.0%} over the last {LAST:,} steps")
    end = rows[-1]
    print(f"== averages: fear of death {float(end['fear_of_death']):.2f} · love of babies {float(end['love_of_babies']):.2f}"
          f" · learns from others {float(end['learns_from_others']):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--save", type=int, required=True)
    parser.add_argument("--until", type=int, required=True)
    parser.add_argument("--learning", choices=("own", "shared"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.learning == "own":
        own_learning()
    save = copy_of(args.run, args.save, args.out)
    evolve.load_world = seeded(args.seed)
    evolve.run(evolve.parse_args(["--resume", str(save), "--steps", str(args.until), "--out", str(args.out)]))
    report(args.out, args.save)


if __name__ == "__main__":
    main()
