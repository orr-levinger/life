import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class NeuralNetwork:
    """
    A simple neural network for creature decision making.
    
    This network takes sensory inputs and outputs action probabilities.
    It supports mutation for evolutionary learning.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, weights: Optional[List[np.ndarray]] = None):
        """
        Initialize a neural network with the given architecture.
        
        Args:
            input_size: Number of input neurons (sensory inputs)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons (action probabilities)
            weights: Optional pre-defined weights for the network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Define the network architecture
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights randomly if not provided
        if weights is None:
            self.weights = []
            for i in range(len(self.layer_sizes) - 1):
                # Initialize with small random values
                w = np.random.randn(self.layer_sizes[i] + 1, self.layer_sizes[i + 1]) * 0.1
                self.weights.append(w)
        else:
            self.weights = weights
            
        # For tracking learning
        self.last_inputs = None
        self.last_hidden_states = None
        self.last_output = None
        self.last_action = None
        self.reward_history = []
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            inputs: Input vector of shape (input_size,)
            
        Returns:
            Output vector of shape (output_size,)
        """
        # Store inputs for learning
        self.last_inputs = inputs
        self.last_hidden_states = []
        
        # Forward pass through each layer
        x = inputs
        for i, w in enumerate(self.weights):
            # Add bias term
            x = np.append(x, 1.0)
            
            # Linear combination
            z = np.dot(x, w)
            
            # Apply activation function (ReLU for hidden layers, softmax for output)
            if i < len(self.weights) - 1:
                # ReLU activation for hidden layers
                x = np.maximum(0, z)
                self.last_hidden_states.append(x)
            else:
                # Softmax activation for output layer
                exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
                x = exp_z / np.sum(exp_z)
        
        # Store output for learning
        self.last_output = x
        return x
    
    def decide(self, sensory_inputs: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Make a decision based on sensory inputs.
        
        Args:
            sensory_inputs: Dictionary of sensory inputs
            
        Returns:
            Tuple of (action_type, action_params)
        """
        # Convert sensory inputs to a flat vector
        input_vector = self._process_sensory_inputs(sensory_inputs)
        
        # Forward pass through the network
        output_probs = self.forward(input_vector)
        
        # Map output probabilities to actions
        action_type, action_params = self._map_output_to_action(output_probs, sensory_inputs)
        
        # Store the action for learning
        self.last_action = (action_type, action_params)
        
        return action_type, action_params
    
    def _process_sensory_inputs(self, sensory_inputs: Dict[str, Any]) -> np.ndarray:
        """
        Convert sensory inputs to a flat vector for the neural network.
        
        Args:
            sensory_inputs: Dictionary containing:
                - 'vision': List of (type, obj, dist, angle) tuples
                - 'on_food': Boolean indicating if creature is on food
                - 'creature_state': Dictionary with creature's state (energy, size, etc.)
                
        Returns:
            Flat numpy array of inputs
        """
        # Initialize input vector with creature's own state
        creature_state = sensory_inputs.get('creature_state', {})
        inputs = [
            creature_state.get('energy', 0) / 100.0,  # Normalize energy
            creature_state.get('size', 1.0) / 5.0,    # Normalize size
            creature_state.get('velocity', 1.0),      # Velocity
            1.0 if sensory_inputs.get('on_food', False) else 0.0  # On food flag
        ]
        
        # Process vision inputs
        vision = sensory_inputs.get('vision', [])
        
        # Count nearby creatures and food
        num_creatures = 0
        num_food = 0
        nearest_creature_dist = 999.0
        nearest_creature_angle = 0.0
        nearest_food_dist = 999.0
        nearest_food_angle = 0.0
        
        for type_tag, obj, dist, angle in vision:
            if type_tag == 'creature':
                num_creatures += 1
                if dist < nearest_creature_dist:
                    nearest_creature_dist = dist
                    nearest_creature_angle = angle
            elif type_tag == 'food':
                num_food += 1
                if dist < nearest_food_dist:
                    nearest_food_dist = dist
                    nearest_food_angle = angle
        
        # Normalize distances
        nearest_creature_dist = min(nearest_creature_dist, 10.0) / 10.0
        nearest_food_dist = min(nearest_food_dist, 10.0) / 10.0
        
        # Add vision-derived inputs
        inputs.extend([
            num_creatures / 10.0,  # Normalize count
            num_food / 10.0,       # Normalize count
            nearest_creature_dist,
            np.cos(nearest_creature_angle),  # Direction components
            np.sin(nearest_creature_angle),
            nearest_food_dist,
            np.cos(nearest_food_angle),      # Direction components
            np.sin(nearest_food_angle)
        ])
        
        return np.array(inputs)
    
    def _map_output_to_action(self, output_probs: np.ndarray, sensory_inputs: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Map neural network outputs to creature actions.
        
        Args:
            output_probs: Output probabilities from the neural network
            sensory_inputs: Original sensory inputs for reference
            
        Returns:
            Tuple of (action_type, action_params)
        """
        # Define action types
        action_types = ["MOVE", "REST", "EAT", "EAT_AT_CURRENT", "ATTACK", "FLEE"]
        
        # Choose action type based on highest probability
        action_idx = np.argmax(output_probs[:len(action_types)])
        action_type = action_types[action_idx]
        
        # Get creature state
        creature_state = sensory_inputs.get('creature_state', {})
        velocity = creature_state.get('velocity', 1.0)
        
        # Get vision data
        vision = sensory_inputs.get('vision', [])
        on_food = sensory_inputs.get('on_food', False)
        
        # Process action parameters based on action type
        if action_type == "MOVE":
            # Use remaining outputs to determine movement direction and speed
            angle = 2 * np.pi * output_probs[len(action_types)]
            speed_factor = output_probs[len(action_types) + 1]  # Between 0 and 1
            
            # Calculate movement vector
            dx = np.cos(angle) * velocity * speed_factor
            dy = np.sin(angle) * velocity * speed_factor
            
            return "MOVE", (dx, dy)
            
        elif action_type == "REST":
            return "REST", None
            
        elif action_type == "EAT":
            # Find nearest food
            nearest_food = None
            nearest_dist = float('inf')
            
            for type_tag, obj, dist, angle in vision:
                if type_tag == "food" and dist < nearest_dist:
                    nearest_food = obj
                    nearest_dist = dist
            
            if nearest_food:
                return "EAT", nearest_food
            else:
                # Fallback to REST if no food found
                return "REST", None
                
        elif action_type == "EAT_AT_CURRENT":
            if on_food:
                return "EAT_AT_CURRENT", None
            else:
                # Fallback to REST if not on food
                return "REST", None
                
        elif action_type == "ATTACK":
            # Find nearest creature
            nearest_creature = None
            nearest_dist = float('inf')
            
            for type_tag, obj, dist, angle in vision:
                if type_tag == "creature" and dist < nearest_dist:
                    nearest_creature = obj
                    nearest_dist = dist
            
            if nearest_creature:
                return "ATTACK", nearest_creature
            else:
                # Fallback to REST if no creature found
                return "REST", None
                
        elif action_type == "FLEE":
            # Find nearest creature to flee from
            nearest_creature = None
            nearest_dist = float('inf')
            nearest_angle = 0
            
            for type_tag, obj, dist, angle in vision:
                if type_tag == "creature" and dist < nearest_dist:
                    nearest_creature = obj
                    nearest_dist = dist
                    nearest_angle = angle
            
            if nearest_creature:
                # Flee in opposite direction
                flee_angle = nearest_angle + np.pi
                dx = np.cos(flee_angle) * velocity
                dy = np.sin(flee_angle) * velocity
                
                return "FLEE", (dx, dy)
            else:
                # Fallback to REST if no creature found
                return "REST", None
        
        # Default fallback
        return "REST", None
    
    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.2) -> 'NeuralNetwork':
        """
        Create a mutated copy of this neural network.
        
        Args:
            mutation_rate: Probability of each weight being mutated
            mutation_scale: Scale of mutations when they occur
            
        Returns:
            A new NeuralNetwork with mutated weights
        """
        new_weights = []
        
        for layer_weights in self.weights:
            # Create a copy of the weights
            w_copy = layer_weights.copy()
            
            # Create a mask of weights to mutate
            mutation_mask = np.random.random(w_copy.shape) < mutation_rate
            
            # Generate random mutations
            mutations = np.random.randn(*w_copy.shape) * mutation_scale
            
            # Apply mutations only where mask is True
            w_copy[mutation_mask] += mutations[mutation_mask]
            
            new_weights.append(w_copy)
        
        # Create a new network with the mutated weights
        return NeuralNetwork(
            self.input_size,
            self.hidden_sizes,
            self.output_size,
            new_weights
        )
    
    def apply_reward(self, reward: float) -> None:
        """
        Apply a reward to the network for reinforcement learning.
        
        Args:
            reward: Positive value for good actions, negative for bad actions
        """
        # Store the reward for this action
        self.reward_history.append((self.last_action, reward))
        
        # In a more sophisticated implementation, we would use these rewards
        # to update the weights using reinforcement learning algorithms
        # For now, we'll just store them for potential future use
    
    def clone(self) -> 'NeuralNetwork':
        """
        Create an exact copy of this neural network.
        
        Returns:
            A new NeuralNetwork with the same weights
        """
        new_weights = [w.copy() for w in self.weights]
        
        return NeuralNetwork(
            self.input_size,
            self.hidden_sizes,
            self.output_size,
            new_weights
        )