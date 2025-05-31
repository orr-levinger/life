from world import World
from creature import Creature
import random

def main() -> None:
    # Configuration for Stage 1
    WIDTH, HEIGHT = 50, 50
    FOOD_RATE = 0.0            # No food spawns in Stage 1
    INITIAL_CREATURES = 5      # Start with 5 creatures
    NUM_STEPS = 10             # Run 10 steps for demonstration

    # Initialize world
    world = World(WIDTH, HEIGHT, FOOD_RATE)

    # Create initial creatures at random positions
    for _ in range(INITIAL_CREATURES):
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
        size = random.uniform(0.5, 2.0)
        energy = random.uniform(10.0, 20.0)
        velocity = 1.0 / size   # Example: bigger creatures move slower
        creature = Creature(x, y, size, energy, velocity)
        world.add_creature(creature)

    # Main simulation loop
    for step in range(NUM_STEPS):
        world.step()
        alive = len(world.creatures)
        print(f"Step {step}: {alive} creatures alive")
        if alive == 0:
            print("All creatures have died prematurely.")
            break

if __name__ == "__main__":
    main()