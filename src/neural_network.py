# src/neural_network.py

import tensorflow as tf     # TensorFlow library for building and training neural networks
import numpy as np          # NumPy for numerical operations, especially array manipulations
from typing import Dict, Any, List, Tuple, Optional

class NeuralNetwork:
    """
    A small policy-network implemented in tf.keras for creature decision-making.
    The network architecture is:
      inputs → Dense(hidden1, ReLU) → Dense(hidden2, ReLU) → Dense(output_size, logits)
    We apply a simple REINFORCE (policy-gradient) update rule:
      loss = −log(π(a|s)) * reward
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, learning_rate: float = 0.01):
        """
        Initialize the neural network with:
          - input_size: number of float inputs (sensory + state features)
          - hidden_sizes: list of integers specifying the size of each hidden layer
          - output_size: number of outputs (action logits; here 8: 6 discrete actions + 2 continuous movement params)
          - learning_rate: learning rate for the Adam optimizer

        This constructor builds a tf.keras.Sequential model:
          1) Dense(hidden_sizes[0], activation='relu', input_shape=(input_size,))
          2) For each additional hidden layer in hidden_sizes: Dense(size, activation='relu')
          3) Final Dense(output_size, activation=None) to produce raw logits

        We store:
          - self.model: the tf.keras.Sequential model
          - self.optimizer: an Adam optimizer with the specified learning rate
          - placeholders for the last forward pass:
              - self._last_input: saved input vector (shape (1, input_size)) to recompute logits during training
              - self._last_logits: saved raw output from the network (shape (1, output_size))
              - self._last_action_idx: index of the action chosen (0–5) during that forward pass
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.lr = learning_rate

        # Build a sequential model container
        self.model = tf.keras.Sequential()

        # Add the first hidden layer and specify input_shape so Keras can build weights
        # - hidden_sizes[0] neurons
        # - ReLU activation introduces nonlinearity
        # - input_shape=(input_size,) expects a 1D array of length input_size
        self.model.add(tf.keras.layers.Dense(hidden_sizes[0], activation='relu', input_shape=(input_size,)))

        # Add any additional hidden layers (if hidden_sizes has length > 1)
        for h in hidden_sizes[1:]:
            # Each subsequent hidden layer has 'h' neurons and ReLU activation
            self.model.add(tf.keras.layers.Dense(h, activation='relu'))

        # Add the output layer:
        # - output_size neurons, one logit per possible action or movement parameter
        # - activation=None means raw linear outputs, which we will convert to probabilities via softmax later
        self.model.add(tf.keras.layers.Dense(output_size, activation=None))

        # Create an Adam optimizer to update weights during training
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # Placeholders to store data from the last forward pass for computing gradients:
        self._last_input: Optional[np.ndarray] = None       # numpy array of shape (1, input_size)
        self._last_logits: Optional[tf.Tensor] = None       # TensorFlow tensor of shape (1, output_size)
        self._last_action_idx: Optional[int] = None         # Index (0–5) of the chosen discrete action

    def decide(self, sensory_inputs: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Use the neural network to choose an action based on sensory inputs.

        Steps:
          1. Convert sensory_inputs (dict) into a flat numpy array of shape (input_size,)
          2. Wrap it in a batch of size 1 (shape (1, input_size)) for the model
          3. Perform a forward pass: logits = model(input_batch)
          4. Apply softmax to logits to obtain action probabilities (shape (1, output_size))
          5. Extract numpy array of probabilities: probs_np (shape (output_size,))
          6. Choose action_idx = argmax(probs_np[:6]) among the first 6 outputs (discrete action logits)
          7. Store input_batch and logits for later use in apply_reward()
          8. Call _map_output_to_action(probs_np, sensory_inputs) to get (action_type, action_params)
          9. Return (action_type, action_params)

        Args:
            sensory_inputs: a dictionary containing keys:
              - 'vision': list of (type_tag, obj, distance, angle)
              - 'on_food': boolean
              - 'creature_state': dict with { 'energy', 'size', 'velocity', 'max_energy' }

        Returns:
            action_type (str), action_params (Any)
        """
        # 1) Convert sensory inputs to a flat numpy vector
        input_vec = self._process_sensory_inputs(sensory_inputs)   # shape: (input_size,)
        # 2) Create a batch of size 1 (model expects 2D inputs)
        input_batch = np.expand_dims(input_vec, axis=0).astype(np.float32)  # shape: (1, input_size)
        # Store this batch for potential gradient computation later
        self._last_input = input_batch

        # 3) Run the forward pass through the model
        logits = self.model(input_batch)                     # Tensor shape: (1, output_size)
        # Save raw logits in case we need to recompute them during training
        self._last_logits = logits

        # 4) Compute softmax probabilities from logits for action selection
        probs = tf.nn.softmax(logits, axis=-1)               # Tensor shape: (1, output_size)
        # Convert to numpy array and flatten to 1D for easy indexing
        probs_np = probs.numpy().flatten()                   # numpy array shape: (output_size,)

        # 6) Among the first 6 outputs, pick the index of the largest probability
        # These indices correspond to discrete actions: ["MOVE", "REST", "EAT", ... , "FLEE"]
        action_idx = int(np.argmax(probs_np[:6]))
        # Store the chosen action index to compute policy-gradient later
        self._last_action_idx = action_idx

        # 8) Map the entire probability vector and sensory inputs to a concrete action & parameters
        action_type, action_params = self._map_output_to_action(probs_np, sensory_inputs)

        return action_type, action_params

    def apply_reward(self, reward: float) -> None:
        """
        After the environment has been updated and energy change (reward) is known, perform a single
        REINFORCE policy-gradient update.

        Specifically:
          1. If we have no stored forward-pass data, do nothing (no training).
          2. Otherwise, recompute logits = model(self._last_input).
          3. Compute softmax to get new probabilities.
          4. Extract the probability of the previously chosen action: prob = probs[0, self._last_action_idx].
          5. Compute loss = −log(prob + epsilon) * reward. (Add small epsilon for numerical stability.)
          6. Use tf.GradientTape() to compute gradients of loss w.r.t. model.trainable_variables.
          7. Apply gradients using the Adam optimizer.
          8. Clear stored forward-pass data to avoid accidental reuse.

        Args:
            reward (float): Can be positive (energy gained) or negative (energy lost).
        """
        # If any of the needed data from the last forward pass is missing, skip training
        if self._last_input is None or self._last_logits is None or self._last_action_idx is None:
            return

        # Use TensorFlow's automatic differentiation
        with tf.GradientTape() as tape:
            # Recompute logits for the stored input (fresh forward pass)
            logits = self.model(self._last_input)               # shape: (1, output_size)
            # Convert to probabilities
            probs = tf.nn.softmax(logits, axis=-1)               # shape: (1, output_size)
            # Extract the probability for the chosen action
            chosen_prob = probs[0, self._last_action_idx]        # scalar tensor

            # Compute REINFORCE loss: −log(π(a|s)) * reward
            # Adding a tiny epsilon (1e-8) prevents log(0) if chosen_prob is extremely small
            loss = -tf.math.log(chosen_prob + 1e-8) * reward     # scalar tensor

        # Compute gradients of loss w.r.t. each model parameter
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients via optimizer step
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Clear stored data to avoid using stale forward-pass information
        self._last_input = None
        self._last_logits = None
        self._last_action_idx = None

    def _process_sensory_inputs(self, sensory_inputs: Dict[str, Any]) -> np.ndarray:
        """
        Convert the dictionary of sensory inputs into a flat numpy array for the neural network.

        The inputs include:
          - Creature's own state (energy, size, velocity, whether on food)
          - Vision-derived features (counts and nearest distances/angles to creatures and food)

        Specifically, we build a list of floats in this order:
          1. energy normalized by 100.0
          2. size normalized by 5.0
          3. velocity (raw)
          4. on_food flag (1.0 if True, else 0.0)
          5. number of creatures in vision / 10
          6. number of food items in vision / 10
          7. nearest creature distance normalized by 10
          8. cosine of nearest creature angle
          9. sine of nearest creature angle
         10. nearest food distance normalized by 10
         11. cosine of nearest food angle
         12. sine of nearest food angle

        Returns:
            np.ndarray of length input_size, dtype float32
        """
        creature_state = sensory_inputs.get('creature_state', {})

        # --- Normalize creature state features ---
        # Energy scaled by 100 so typical energies ~ [0,1]
        # Size scaled by 5 so typical sizes ~ [0,1]
        inputs: List[float] = [
            creature_state.get('energy', 0) / 100.0,
            creature_state.get('size', 1.0) / 5.0,
            creature_state.get('velocity', 1.0),        # velocity is already small
            1.0 if sensory_inputs.get('on_food', False) else 0.0
        ]

        # --- Gather vision data about nearby creatures and food ---
        vision = sensory_inputs.get('vision', [])
        num_creatures = 0
        num_food = 0
        nearest_creature_dist = 999.0    # Initialize to a large number
        nearest_creature_angle = 0.0
        nearest_food_dist = 999.0        # Initialize to a large number
        nearest_food_angle = 0.0

        # Iterate through all sensed objects to compute counts and nearest metrics
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

        # Normalize nearest distances by a maximum sensor range (10)
        nearest_creature_dist = min(nearest_creature_dist, 10.0) / 10.0
        nearest_food_dist = min(nearest_food_dist, 10.0) / 10.0

        # --- Append vision-derived inputs to the list ---
        inputs.extend([
            num_creatures / 10.0,                 # Normalize the count of creatures (max assumed 10)
            num_food / 10.0,                      # Normalize the count of food items (max assumed 10)
            nearest_creature_dist,                # Normalized distance to nearest creature
            np.cos(nearest_creature_angle),       # X-component of nearest creature direction
            np.sin(nearest_creature_angle),       # Y-component of nearest creature direction
            nearest_food_dist,                    # Normalized distance to nearest food
            np.cos(nearest_food_angle),           # X-component of nearest food direction
            np.sin(nearest_food_angle)            # Y-component of nearest food direction
        ])

        # Convert to a numpy array of dtype float32 as required by TensorFlow
        return np.array(inputs, dtype=np.float32)

    def _map_output_to_action(self, output_probs: np.ndarray, sensory_inputs: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Convert the network's output probability array into a concrete action.

        Arguments:
          - output_probs: numpy array of length output_size containing probabilities
                           (softmax over raw logits).
          - sensory_inputs: original dictionary to access 'vision', 'on_food', and creature state.

        Steps:
          1. Define the discrete action types in a fixed list:
               ["MOVE", "REST", "EAT", "EAT_AT_CURRENT", "ATTACK", "FLEE"]
          2. The previously chosen action index (0–5) is stored in self._last_action_idx.
          3. Map that index to an action_type string.
          4. For "MOVE": use outputs[6] and outputs[7] to determine a continuous movement direction:
               - angle = 2π * output_probs[6]
               - speed_factor = output_probs[7] ∈ [0, 1]
               - dx = cos(angle) * velocity * speed_factor
               - dy = sin(angle) * velocity * speed_factor
          5. For "REST": no additional parameters.
          6. For "EAT": search through vision to find the nearest food object; return that or fallback to "REST".
          7. For "EAT_AT_CURRENT": if on_food is True, return that, else fallback to "REST".
          8. For "ATTACK": search vision for the nearest creature; return that or fallback to "REST".
          9. For "FLEE": find nearest creature, compute opposite direction at full velocity, else "REST".
        """
        # Define mapping from index to action type
        action_types = ["MOVE", "REST", "EAT", "EAT_AT_CURRENT", "ATTACK", "FLEE"]
        # Retrieve the action_type chosen in decide()
        action_type = action_types[self._last_action_idx]

        # Extract creature_state for movement parameters
        creature_state = sensory_inputs.get('creature_state', {})
        velocity = creature_state.get('velocity', 1.0)

        # Retrieve vision list and on_food flag if needed below
        vision = sensory_inputs.get('vision', [])
        on_food = sensory_inputs.get('on_food', False)

        if action_type == "MOVE":
            # The next two probabilities (indices 6 and 7) encode continuous movement:
            # - output_probs[6] is used to choose an angle in [0, 2π)
            # - output_probs[7] is used as a speed factor ∈ [0, 1]
            angle = 2 * np.pi * output_probs[len(action_types)]
            speed_factor = output_probs[len(action_types) + 1]
            dx = np.cos(angle) * velocity * speed_factor
            dy = np.sin(angle) * velocity * speed_factor
            return "MOVE", (dx, dy)

        elif action_type == "REST":
            # No movement, creature rests
            return "REST", None

        elif action_type == "EAT":
            # Find the nearest food object from vision.
            nearest_food = None
            nearest_dist = float('inf')
            for t, obj, dist, angle in vision:
                if t == "food" and dist < nearest_dist:
                    nearest_food = obj
                    nearest_dist = dist
            if nearest_food:
                return "EAT", nearest_food
            else:
                # If no food is detected, fallback to resting
                return "REST", None

        elif action_type == "EAT_AT_CURRENT":
            # If the creature is on food, eat at the current position
            if on_food:
                return "EAT_AT_CURRENT", None
            else:
                # Otherwise, fallback to resting
                return "REST", None

        elif action_type == "ATTACK":
            # Find nearest creature to attack
            nearest_creature = None
            nearest_dist = float('inf')
            for t, obj, dist, angle in vision:
                if t == "creature" and dist < nearest_dist:
                    nearest_creature = obj
                    nearest_dist = dist
            if nearest_creature:
                return "ATTACK", nearest_creature
            else:
                # No creature to attack, fallback to resting
                return "REST", None

        elif action_type == "FLEE":
            # Find nearest creature to flee from
            nearest_creature = None
            nearest_dist = float('inf')
            nearest_angle = 0.0
            for t, obj, dist, angle in vision:
                if t == "creature" and dist < nearest_dist:
                    nearest_creature = obj
                    nearest_dist = dist
                    nearest_angle = angle
            if nearest_creature:
                # Compute opposite direction of nearest threat
                flee_angle = nearest_angle + np.pi
                dx = np.cos(flee_angle) * velocity
                dy = np.sin(flee_angle) * velocity
                return "FLEE", (dx, dy)
            else:
                # No threat detected, fallback to resting
                return "REST", None

        # Default fallback to rest if none of the above matched
        return "REST", None

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.2) -> 'NeuralNetwork':
        """
        Create a mutated copy of this network's weights for evolutionary purposes.

        Steps:
          1. Extract the current model weights as a list of numpy arrays: weights = model.get_weights()
             Each array corresponds to a weight matrix or bias vector in each Dense layer.
          2. For each weight array w:
             a. Make a copy w_copy = w.copy()
             b. Create a boolean mask of the same shape where each element is True with probability mutation_rate
             c. Generate a random mutation array drawn from N(0, mutation_scale) of the same shape
             d. For all positions where mask is True, add the corresponding random mutation to w_copy
          3. Collect all mutated arrays into new_weights list
          4. Instantiate a new NeuralNetwork with the same architecture (input_size, hidden_sizes, output_size, lr)
          5. Set its weights via child.model.set_weights(new_weights)
          6. Return the new mutated network

        Args:
            mutation_rate (float): Probability for each weight element to be mutated (e.g., 0.1 = 10% chance).
            mutation_scale (float): Standard deviation of Gaussian noise added to weights when mutating.

        Returns:
            A new NeuralNetwork instance with mutated weights.
        """
        # 1) Grab current weights from each layer as numpy arrays
        weights = self.model.get_weights()  # List[np.ndarray]: weight matrices and bias vectors
        new_weights = []

        # 2) For each weight array, apply random mutations
        for w in weights:
            w_copy = w.copy()
            # Create a mask where each element has probability mutation_rate of being mutated
            mask = np.random.random(w_copy.shape) < mutation_rate
            # Generate Gaussian noise array
            mutations = np.random.randn(*w_copy.shape) * mutation_scale
            # Add noise only where mask is True
            w_copy[mask] += mutations[mask]
            new_weights.append(w_copy)

        # 4) Create a fresh network with identical architecture
        child = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size, self.lr)
        # 5) Overwrite its initial random weights with our mutated weights
        child.model.set_weights(new_weights)
        return child

    def clone(self) -> 'NeuralNetwork':
        """
        Produce an exact copy of this network, including all weights.

        Steps:
          1. Extract current weights = self.model.get_weights()
          2. Create a new NeuralNetwork instance with the same architecture
          3. Deep-copy each weight array and set them on the new network’s model
          4. Return the new cloned network

        Returns:
            A new NeuralNetwork instance with identical weights to self.
        """
        # 1) Retrieve current weights from model
        weights = self.model.get_weights()

        # 2) Instantiate a new, identical network
        clone = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size, self.lr)

        # 3) Deep-copy each numpy array so the clone’s weights are independent
        clone.model.set_weights([w.copy() for w in weights])
        return clone
