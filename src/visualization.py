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
         - Creatures as green circles with radius = creature.radius.
         - Food as red circles with radius proportional to its energy.
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

        # 1) Draw food (if any) as red circles with radius proportional to energy
        if len(world.foods) > 0:
            for f in world.foods:
                # Calculate the scaled radius based on energy
                scaled_radius = f.radius

                # Draw the food as a circle
                food_circle = plt.Circle(
                    (f.x, f.y),
                    radius=scaled_radius,
                    color='red',
                    alpha=0.8
                )
                self.ax.add_patch(food_circle)

                # Draw remaining duration for non-infinite food
                if f.remaining_duration > 0:
                    self.ax.text(
                        f.x, 
                        f.y - scaled_radius * 0.5, 
                        f"{f.remaining_duration}", 
                        color='white',
                        fontsize=8,
                        ha='center',
                        va='center'
                    )

                # In debug mode, show remaining energy
                if self.debug:
                    self.ax.text(
                        f.x,
                        f.y + scaled_radius * 0.5,
                        f"{f.energy:.1f}",
                        color='white',
                        fontsize=6,
                        ha='center',
                        va='center'
                    )

        # 2) Draw creatures as green circles with radius = creature.radius
        if len(world.creatures) > 0:
            for c in world.creatures:
                # Draw the creature as a circle
                creature_circle = plt.Circle(
                    (c.x, c.y),
                    radius=c.radius,
                    color=c.get_color(),  # Use the creature's color method
                    alpha=0.9,
                    ec='white',
                    lw=0.5
                )
                self.ax.add_patch(creature_circle)

                # Always show energy above the creature
                self.ax.text(
                    c.x,
                    c.y + c.radius + 0.2,  # Position above the creature
                    f"{c.energy:.1f}",
                    color='white',
                    fontsize=8,  # Slightly larger font for better visibility
                    ha='center',
                    va='bottom'
                )

                self.ax.text(
                    c.x,
                    c.y,
                    f"{c.get_nn_score():.0f}",
                    color='red',
                    fontsize=6,
                    ha='center',
                    va='center'
                )

                self.ax.text(
                    c.x,
                    c.y - c.radius - 0.2,   # Position text slightly below the bottom of the circle
                    f"{c.steps_without_reward}",  # Display as integer
                    color='white',
                    fontsize=6,            # Smaller font so it doesn't clutter
                    ha='center',
                    va='top'               # Top-aligned so the text anchors from above
                )

            # Draw arrows indicating movement direction for each creature
            for c in world.creatures:
                # Get the creature's position (center of the circle)
                cx, cy = c.x, c.y

                # Get the intended vector
                dx, dy = c.intended_vector

                # Skip if no movement or resting
                if (dx == 0 and dy == 0) or c.intent == "REST":
                    continue

                # Determine arrow color based on intent
                arrow_color = 'gray'  # Default color for wandering

                if c.intent == "ATTACK":
                    arrow_color = 'red'  # Red for attack
                elif c.intent == "GO_TO_FOOD":
                    arrow_color = 'green'  # Green for going to food
                elif c.intent == "RUN_AWAY":
                    arrow_color = 'blue'  # Blue for fleeing
                elif c.intent == "WANDER":
                    arrow_color = 'gray'  # Gray for wandering

                # Draw the arrow
                # Scale the arrow length to be visible but not too large
                # The arrow length is proportional to the creature's current speed
                arrow_scale = 2.0  # Adjust this value to make arrows more visible
                self.ax.arrow(
                    cx, cy,  # Start at creature center
                    dx * arrow_scale, dy * arrow_scale,  # End at scaled intended vector
                    head_width=0.2,
                    head_length=0.3,
                    fc=arrow_color,
                    ec=arrow_color,
                    alpha=0.8
                )

        # Optional: Add a legend or title
        # self.ax.legend(loc='upper right', fontsize='small', facecolor='black', labelcolor='white')
        self.ax.set_title(
            f"World: {len(world.creatures)} creatures, {len(world.foods)} food",
            color='white',
            pad=0
        )

        # Finally, draw the canvas immediately:
        self.fig.canvas.draw()
