"""Watch the simulation in a window: `python src/main.py [evolve arguments]` (seed 0, instinct animals, when none are given)."""
import os
import sys
from typing import Optional, Sequence

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evolve import main as evolve_main


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv) or ["--seed", "0", "--species", "instinct"]
    if "--window" not in args:
        args.append("--window")
    evolve_main(args)


if __name__ == "__main__":
    main()
