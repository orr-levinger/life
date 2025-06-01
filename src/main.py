import sys
import os
import random
import time
import matplotlib.pyplot as plt

# Handle both being imported as a module and being run directly
if __name__ == "__main__":
    # When run directly, use absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.world import World
    from src.creature import Creature
    from src.visualization import Visualizer
else:
    # When imported as a module, use relative imports
    from .world import World
    from .creature import Creature
    from .visualization import Visualizer

def main() -> None:
    # Configuration for Stage 1.5
    WIDTH, HEIGHT = 40, 40
    FOOD_RATE = 0.005          # Reduced food spawn rate to prevent screen filling
    INITIAL_CREATURES = 400     # Start with 10 creatures
    NUM_STEPS = 1000            # Run 100 steps for demonstration
    PAUSE_TIME = 0.001          # seconds to pause between frames
    USE_BRAIN = True  # Whether to use brains for creatures (default is True)

    # 1) Create the world
    world = World(WIDTH, HEIGHT, FOOD_RATE)

    # 2) Spawn initial creatures at random positions
    for _ in range(INITIAL_CREATURES):
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
        size = random.uniform(0.5, 2.0)
        energy = random.uniform(50.0, 100.0)
        # IMPLEMENT: no need to compute velocity here; Creature.__init__ will do it
        # velocity = 1.0 / size
        creature = Creature(x, y, size, energy, create_brain=USE_BRAIN)
        world.add_creature(creature)

    # 3) Create the Visualizer
    viz = Visualizer(WIDTH, HEIGHT)

    # 4) Show initial state
    plt.ion()                 # Turn on interactive mode
    viz.render(world)
    plt.pause(PAUSE_TIME)

    # 5) Main loop: step and redraw
    for step in range(1, NUM_STEPS + 1):
        world.step()
        viz.render(world)
        plt.pause(PAUSE_TIME)

        alive = len(world.creatures)
        print(f"Step {step}: {alive} creatures alive")

        # If all creatures died, exit early
        if alive == 0:
            print("All creatures have died prematurely.")
            break

    # 6) Keep window open until closed by user
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
