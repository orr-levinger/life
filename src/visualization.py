# src/visualization.py

import matplotlib.pyplot as plt
from typing import Tuple

# Import at the top to avoid circular imports
import sys
import os

# Handle both being imported as a module and being run directly
if __name__ == "__main__":
    # When run directly, use absolute imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.world import World
else:
    # When imported as a module, use relative imports
    from .world import World

class Visualizer:
    """
    A simple Matplotlib-based visualizer for our World + Creature system.

    Usage:
        viz = Visualizer(world_width, world_height)
        viz.render(world)       # to draw the current state
        plt.pause(0.1)           # pause briefly so the figure updates
    """

    def __init__(self, width: int, height: int, cell_size: float = 1.0, debug: bool = False):
        """
        width, height: The dimensions of the world grid.
        cell_size: The size of each "cell" in plotting units. Defaults to 1.0.
        debug: Whether to show debug information like remaining energy.

        Initializes a Matplotlib figure & axes, sets aspect ratio, and
        turns off the axis ticks.
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.debug = debug

        # Create figure & axis:
        self.fig, self.ax = plt.subplots(figsize=(width * 0.5, height * 0.5))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor('black')  # black background to contrast creatures/food

        # We'll keep scatter-plot "handles" so we can update them in place if desired.
        self.creature_scat = None
        self.food_scat = None

    def render(self, world: World) -> None:
        """
        Draw the current state of `world`:
         - Creatures as green circles, radius ∝ creature.size.
         - Food as red squares/dots at each (x,y) in world.food_positions.
        After calling render(), call plt.pause() to let the figure update.
        """
        # Clear existing artists:
        self.ax.cla()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor('black')

        # 1) Draw food (if any). Each food point is a red square with size proportional to remaining_energy / size:
        if len(world.foods) > 0:
            food_xs = [f.x + 0.5 for f in world.foods]
            food_ys = [f.y + 0.5 for f in world.foods]
            # Scale the food size based on remaining energy
            # Matplotlib's 's' argument in scatter is area in points^2
            food_sizes = [(f.size * 25 * (f.remaining_energy / f.energy_value)) for f in world.foods]  # Scale by remaining energy

            # Use red squares with size proportional to remaining energy
            self.ax.scatter(
                food_xs,
                food_ys,
                marker='s',
                color='red',
                s=food_sizes,
                alpha=0.8,
                label='Food'
            )

            # Draw remaining duration and energy information
            for f in world.foods:
                # Draw remaining duration for non-infinite food
                if f.remaining_duration > 0:
                    self.ax.text(
                        f.x + 0.5, 
                        f.y + 0.3, 
                        f"{f.remaining_duration}", 
                        color='white',
                        fontsize=8,
                        ha='center',
                        va='center'
                    )

                # In debug mode, show remaining energy
                if self.debug:
                    self.ax.text(
                        f.x + 0.5,
                        f.y + 0.7,
                        f"{f.remaining_energy:.1f}",
                        color='white',
                        fontsize=6,
                        ha='center',
                        va='center'
                    )

        # 2) Draw creatures as green circles, radius = creature.size:
        if len(world.creatures) > 0:
            creature_xs = [c.x + 0.5 for c in world.creatures]
            creature_ys = [c.y + 0.5 for c in world.creatures]
            # Scale the reported size (which might be any float). We want a visible marker size.
            # Matplotlib's 's' argument in scatter is area in points^2, so we take size*size*50 as a baseline.
            creature_sizes = [(c.size * 50) for c in world.creatures]
            self.ax.scatter(
                creature_xs,
                creature_ys,
                marker='o',
                color='green',
                s=creature_sizes,
                edgecolors='white',
                linewidths=0.5,
                alpha=0.9,
                label='Creatures'
            )

        # Optional: Add a legend or title
        # self.ax.legend(loc='upper right', fontsize='small', facecolor='black', labelcolor='white')
        self.ax.set_title(
            f"World: {len(world.creatures)} creatures, {len(world.foods)} food",
            color='white',
            pad=10
        )

        # Finally, draw the canvas immediately:
        self.fig.canvas.draw()
