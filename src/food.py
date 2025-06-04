# src/food.py

from typing import Tuple
import math

class Food:
    """
    Represents a food item in the world that creatures can eat.

    Food can be either:
    1. Spawned by the environment (with infinite duration)
    2. Created from a creature's corpse (with duration proportional to the creature's size)

    Food has energy that decreases as creatures eat it. Each "bite" reduces
    energy by 1 and increases the creature's energy by 1.

    Food has a physical radius for collision detection in continuous space.
    Size is based on energy (size = energy).
    """

    def __init__(self, x: float, y: float, remaining_duration: int, energy: float = None):
        """
        Initialize a new Food object.

        Args:
            x, y: Continuous coordinates where the food is located
            remaining_duration: Number of world steps this food persists before disappearing
                               -1 means it never expires (infinite lifespan)
            energy: How much energy this food contains (also determines size)
                   If None, a random value between 10-25 is used
        """
        import random
        self.x = float(x)
        self.y = float(y)

        # If energy is not provided, use a random value between 10-25
        if energy is None:
            self.energy = random.uniform(10.0, 25.0)
        else:
            self.energy = energy

        # Radius is 0.01 of the energy
        self.radius = 0.01 * self.energy

        self.remaining_duration = remaining_duration

    def decay(self) -> None:
        """
        Reduce the remaining duration by 1 if it's positive.
        """
        if self.remaining_duration > 0:
            self.remaining_duration -= 1

    def is_expired(self) -> bool:
        """
        Check if this food has expired and should be removed.

        Returns:
            True if remaining_duration is 0, False otherwise
        """
        return self.remaining_duration == 0

    def is_overlapping(self, other_x: float, other_y: float, other_radius: float) -> bool:
        """
        Check if this food overlaps with another object (like a creature).

        Args:
            other_x: x-coordinate of the other object
            other_y: y-coordinate of the other object
            other_radius: radius of the other object

        Returns:
            True if the distance between centers is less than or equal to the sum of radii
        """
        # Calculate distance between centers
        dx = self.x - other_x
        dy = self.y - other_y
        distance = math.sqrt(dx*dx + dy*dy)

        # Check if distance is less than or equal to sum of radii
        return distance <= (self.radius + other_radius)

    def distance_to(self, other_x: float, other_y: float) -> float:
        """
        Calculate the distance from this food to another point.

        Args:
            other_x: x-coordinate of the other point
            other_y: y-coordinate of the other point

        Returns:
            Euclidean distance between this food and the point
        """
        dx = self.x - other_x
        dy = self.y - other_y
        return math.sqrt(dx*dx + dy*dy)

    def __repr__(self) -> str:
        """
        String representation of the Food object.
        """
        duration_str = "∞" if self.remaining_duration == -1 else str(self.remaining_duration)
        return f"<Food x={self.x:.2f} y={self.y:.2f} energy={self.energy:.2f} radius={self.radius:.2f} duration={duration_str}>"
