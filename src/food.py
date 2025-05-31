# src/food.py

from typing import Tuple

class Food:
    """
    Represents a food item in the world that creatures can eat.
    
    Food can be either:
    1. Spawned by the environment (with infinite duration)
    2. Created from a creature's corpse (with duration proportional to the creature's size)
    """
    
    def __init__(self, x: int, y: int, size: float, energy_value: float, remaining_duration: int):
        """
        Initialize a new Food object.
        
        Args:
            x, y: Grid cell coordinates where the food is located
            size: How big this food is (affects visualization)
            energy_value: How much energy a creature gains by eating this food
            remaining_duration: Number of world steps this food persists before disappearing
                               -1 means it never expires (infinite lifespan)
        """
        self.x = x
        self.y = y
        self.size = size
        self.energy_value = energy_value
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
        
    def __repr__(self) -> str:
        """
        String representation of the Food object.
        """
        duration_str = "∞" if self.remaining_duration == -1 else str(self.remaining_duration)
        return f"<Food x={self.x} y={self.y} size={self.size:.2f} energy={self.energy_value:.2f} duration={duration_str}>"