# src/food.py

from typing import Tuple
import math

class Food:
    """
    Represents a food item in the world that creatures can eat.

    Food can be either:
    1. Spawned by the environment (with infinite duration)
    2. Created from a creature's corpse (with duration proportional to the creature's size)

    Food has a remaining_energy that decreases as creatures eat it. Each "bite" reduces
    remaining_energy by 1 and increases the creature's energy by 1.

    Food has a physical radius for collision detection in continuous space.
    """

    def __init__(self, x: float, y: float, size: float, energy_value: float, remaining_duration: int, radius: float = None):
        """
        Initialize a new Food object.

        Args:
            x, y: Continuous coordinates where the food is located
            size: How big this food is (affects visualization and total energy)
            energy_value: How much total energy this food contains
            remaining_duration: Number of world steps this food persists before disappearing
                               -1 means it never expires (infinite lifespan)
            radius: Physical radius for collision detection. If None, computed as size * 0.2
        """
        self.x = float(x)
        self.y = float(y)
        self.size = size
        self.energy_value = energy_value
        self.remaining_energy = energy_value  # Initialize remaining_energy to energy_value
        self.initial_energy = energy_value    # Store initial energy for visualization scaling
        self.remaining_duration = remaining_duration

        # Set radius based on size if not provided
        if radius is None:
            self.radius = size * 0.2  # Default radius factor
        else:
            self.radius = radius

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
        return f"<Food x={self.x:.2f} y={self.y:.2f} size={self.size:.2f} energy={self.energy_value:.2f} remaining={self.remaining_energy:.2f} radius={self.radius:.2f} duration={duration_str}>"
