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
         - Creatures as circles colored by their brain usage (purple/green).
         - Food as red circles with radius proportional to its energy.
         - Overlay text: energy, NN score, steps without reward.
         - Also display total creatures, max NN score, and current simulation step.

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

        # 2) Draw creatures as colored circles with annotations
        if len(world.creatures) > 0:
            for c in world.creatures:
                # Draw the creature as a circle
                creature_circle = plt.Circle(
                    (c.x, c.y),
                    radius=c.radius,
                    color=c.get_color(),  # "purple" if NN used, else "green"
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

                # Show NN score in the middle of the creature
                self.ax.text(
                    c.x,
                    c.y,
                    f"{c.get_nn_score():.0f}",  # Red number indicating NN score (0–100)
                    color='red',
                    fontsize=6,
                    ha='center',
                    va='center'
                )

                # Show steps without reward below the creature
                self.ax.text(
                    c.x,
                    c.y - c.radius - 0.2,   # Position text slightly below the bottom of the circle
                    f"{c.generation}",  # Display as integer
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
                    arrow_color = 'red'       # Red for attack
                elif c.intent == "GO_TO_FOOD":
                    arrow_color = 'green'     # Green for going to food
                elif c.intent == "RUN_AWAY":
                    arrow_color = 'blue'      # Blue for fleeing

                # Draw the arrow
                arrow_scale = 2.0  # Adjust to make arrows visible but not too large
                self.ax.arrow(
                    cx, cy,                     # Start at creature center
                    dx * arrow_scale, dy * arrow_scale,  # Scaled intended vector
                    head_width=0.2,
                    head_length=0.3,
                    fc=arrow_color,
                    ec=arrow_color,
                    alpha=0.8
                )

        # 3) Compute and display global statistics: total creatures, max NN score, current step
        total_creatures = len(world.creatures)
        # Determine maximum NN score among all creatures (or 0 if no creatures)
        max_brain_score = 0.0
        if total_creatures > 0:
            # Use get_nn_score() to retrieve each creature's score and take the maximum
            max_brain_score = max(c.get_nn_score() for c in world.creatures)

        # Attempt to get the current simulation step from the world
        # We expect the World class to maintain a counter like world.step_count
        # If it doesn't exist, default to 0 to avoid an AttributeError
        step_number = getattr(world, 'step_count', None)
        if step_number is None:
            step_number = getattr(world, 'current_step', 0)

        # Draw the text in the top-left corner (inside the plotting area)
        self.ax.text(
            0.5,                            # x-coordinate (slightly right of left edge)
            self.height - 0.5,              # y-coordinate (slightly below top edge)
            f"Step: {step_number}   Creatures: {total_creatures}   MaxScore: {max_brain_score:.0f}",
            color='white',
            fontsize=10,
            ha='left',
            va='top'
        )

        # 4) Set the overall title (optional) or keep it minimal
        # Here, we choose not to duplicate info in the title, since we put stats in corner.
        # But you could also set something like:
        # self.ax.set_title(f"Step {step_number} — Creatures {total_creatures} — Max NN {max_brain_score:.0f}",
        #                   color='white', pad=0)

        # Finally, draw the canvas immediately:
        self.fig.canvas.draw()
