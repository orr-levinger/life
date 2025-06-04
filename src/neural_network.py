# src/neural_network.py

import tensorflow as tf     # TensorFlow library for building and training neural networks
import numpy as np          # NumPy for numerical operations, especially array manipulations
from typing import Dict, Any, List, Tuple, Optional

class NeuralNetwork:
    """
    A small policy-network implemented in tf.keras for creature decision-making.
    The network architecture is:
      inputs → Dense(hidden1, ReLU) → Dense(hidden2, ReLU) → Dense(output_size, logits)
    We apply an episodic REINFORCE (policy-gradient) update rule at the end of each short episode.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, learning_rate: float = 0.01):
        """
        Initialize the neural network with:
          - input_size: number of float inputs (sensory + state features)
          - hidden_sizes: list of integers specifying the size of each hidden layer
          - output_size: number of outputs (action logits; here 8: 6 discrete actions + 2 continuous movement params)
          - learning_rate: learning rate for the Adam optimizer
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.lr = learning_rate

        # Build a tf.keras.Sequential model container
        self.model = tf.keras.Sequential()

        # Add the first hidden layer (specify input_shape for Keras)
        self.model.add(tf.keras.layers.Dense(hidden_sizes[0], activation='relu', input_shape=(input_size,)))

        # Add any additional hidden layers
        for h in hidden_sizes[1:]:
            self.model.add(tf.keras.layers.Dense(h, activation='relu'))

        # Add the final output layer (raw logits)
        self.model.add(tf.keras.layers.Dense(output_size, activation=None))

        # Create an Adam optimizer for updates
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # Placeholders for the last forward pass (used internally)
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
          8. Call _map_output_to_action(probs_np, sensory_inputs, action_idx) to get (action_type, action_params)
          9. Return (action_type, action_params)
        """
        # 1) Convert sensory inputs to a flat numpy vector
        input_vec = self._process_sensory_inputs(sensory_inputs)   # shape: (input_size,)
        # 2) Create a batch of size 1 (model expects 2D inputs)
        input_batch = np.expand_dims(input_vec, axis=0).astype(np.float32)  # shape: (1, input_size)
        # Store this batch for potential gradient computation later
        self._last_input = input_batch

        # 3) Run the forward pass through the model
        logits = self.model(input_batch)                     # Tensor shape: (1, output_size)
        # Save raw logits for potential use
        self._last_logits = logits

        # 4) Compute softmax probabilities from logits for action selection
        probs = tf.nn.softmax(logits, axis=-1)               # Tensor shape: (1, output_size)
        probs_np = probs.numpy().flatten()                   # numpy array shape: (output_size,)

        # 6) Among the first 6 outputs, pick the index of the largest probability
        action_idx = int(np.argmax(probs_np[:6]))
        # Store the chosen action index to compute policy-gradient later
        self._last_action_idx = action_idx

        # 8) Map the entire probability vector and sensory inputs to a concrete action & parameters
        action_type, action_params = self._map_output_to_action(probs_np, sensory_inputs, action_idx)
        return action_type, action_params

    def apply_reward(self, reward: float, terminal: bool) -> None:
        """
        After the environment has been updated and energy change (reward) is known,
        perform a single REINFORCE policy-gradient update—**but only at terminal**.

        Args:
            reward (float): The scalar reward (possibly 0.0 if non-terminal step)
            terminal (bool): True if this step ended the short episode (ate or died), False otherwise.
        """
        # Only train if we have stored forward-pass data AND it's terminal
        if not terminal or self._last_input is None or self._last_logits is None or self._last_action_idx is None:
            # If it's not terminal, we postpone learning until the episode ends
            return

        with tf.GradientTape() as tape:
            # Recompute logits for the stored input (fresh forward pass)
            logits = self.model(self._last_input)               # shape: (1, output_size)
            probs = tf.nn.softmax(logits, axis=-1)               # shape: (1, output_size)
            chosen_prob = probs[0, self._last_action_idx]        # scalar tensor

            # Compute REINFORCE loss: −log(π(a|s)) * reward
            loss = -tf.math.log(chosen_prob + 1e-8) * reward     # scalar tensor

        # Compute gradients of loss w.r.t. each model parameter
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients via optimizer
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Clear stored data to avoid reusing stale forward-pass information
        self._last_input = None
        self._last_logits = None
        self._last_action_idx = None

    def train_reinforce(self, input_vec: np.ndarray, action_idx: int, reward: float) -> None:
        """
        Perform a single REINFORCE gradient step on one (state,action) pair,
        using the *final reward* for that short episode.

        This is called by Creature.apply_action(...) once an episode ends
        to train on all collected (state,action) pairs.

        Args:
            input_vec: np.ndarray of shape (input_size,)
            action_idx: integer in [0..5], which discrete action was taken
            reward: float, the total (terminal) reward for that entire episode
        """
        # 1) Make a batch of shape (1, input_size):
        input_batch = np.expand_dims(input_vec, axis=0).astype(np.float32)  # (1, input_size)

        with tf.GradientTape() as tape:
            logits = self.model(input_batch)                # (1, output_size)
            probs = tf.nn.softmax(logits, axis=-1)          # (1, output_size)
            chosen_prob = probs[0, action_idx]               # scalar
            # REINFORCE loss = −log π(a|s) * reward
            loss = -tf.math.log(chosen_prob + 1e-8) * reward # scalar

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_supervised(self, input_vec: np.ndarray, action_idx: int) -> None:
        """Perform a supervised cross-entropy update for a single example."""
        input_batch = np.expand_dims(input_vec, axis=0).astype(np.float32)
        target = tf.convert_to_tensor([action_idx], dtype=tf.int32)
        with tf.GradientTape() as tape:
            logits = self.model(input_batch)[:, :6]
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                target, logits, from_logits=True
            )
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def _process_sensory_inputs(self, sensory_inputs: Dict[str, Any]) -> np.ndarray:
        """
        Convert the dictionary of sensory inputs into a flat numpy array for the neural network.

        The inputs include:
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
        # Normalize creature state features
        inputs: List[float] = [
            creature_state.get('position', (0.0, 0.0)),
            creature_state.get('energy', 0) / 100.0,
            creature_state.get('size', 1.0) / 5.0,
            creature_state.get('velocity', 1.0),
            1.0 if sensory_inputs.get('on_food', False) else 0.0
        ]

        # Gather vision data about nearby creatures and food
        vision = sensory_inputs.get('vision', [])
        num_creatures = 0
        num_food = 0
        nearest_creature_dist = 999.0    # Initialize large
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

        # Normalize nearest distances by a max sensor range of 10
        nearest_creature_dist = min(nearest_creature_dist, 10.0) / 10.0
        nearest_food_dist = min(nearest_food_dist, 10.0) / 10.0

        inputs.extend([
            num_creatures / 10.0,                 # Normalize count of creatures
            num_food / 10.0,                      # Normalize count of food items
            nearest_creature_dist,                # Normalized distance to nearest creature
            np.cos(nearest_creature_angle),       # X-component of nearest creature direction
            np.sin(nearest_creature_angle),       # Y-component of nearest creature direction
            nearest_food_dist,                    # Normalized distance to nearest food
            np.cos(nearest_food_angle),           # X-component of nearest food direction
            np.sin(nearest_food_angle)            # Y-component of nearest food direction
        ])
        if self.input_size >= 14:
            pos = creature_state.get('position', (0.0, 0.0))
            inputs.append(pos[0] / 10.0)
            inputs.append(pos[1] / 10.0)
        return np.array(inputs, dtype=np.float32)

    def _map_output_to_action(self,
                              output_probs: np.ndarray,
                              sensory_inputs: Dict[str, Any],
                              action_idx: int) -> Tuple[str, Any]:
        """
        Convert the network’s chosen discrete action_index (0–5) plus the
        continuous‐movement parameters (indices 6 & 7) into a concrete (action_type, action_params).

        Arguments:
          - output_probs: numpy array of length output_size containing softmax probabilities
          - sensory_inputs: original dictionary to access 'vision', 'on_food', 'creature_state'
          - action_idx: integer in [0..5], chosen discrete action

        Returns:
            (action_type, action_params)
        """
        action_types = ["MOVE", "REST", "EAT", "EAT_AT_CURRENT", "ATTACK", "FLEE"]
        action_type = action_types[action_idx]

        creature_state = sensory_inputs.get('creature_state', {})
        velocity = creature_state.get('velocity', 1.0)
        vision = sensory_inputs.get('vision', [])
        on_food = sensory_inputs.get('on_food', False)

        if action_type == "MOVE":
            angle = 2 * np.pi * output_probs[len(action_types)]
            speed_factor = output_probs[len(action_types) + 1]
            dx = np.cos(angle) * velocity * speed_factor
            dy = np.sin(angle) * velocity * speed_factor
            return "MOVE", (dx, dy)

        elif action_type == "REST":
            return "REST", None

        elif action_type == "EAT":
            nearest_food = None
            nearest_dist = float('inf')
            for t, obj, dist, ang in vision:
                if t == "food" and dist < nearest_dist:
                    nearest_food = obj
                    nearest_dist = dist
            if nearest_food:
                return "EAT", nearest_food
            else:
                return "REST", None

        elif action_type == "EAT_AT_CURRENT":
            if on_food:
                return "EAT_AT_CURRENT", None
            else:
                return "REST", None

        elif action_type == "ATTACK":
            nearest_creature = None
            nearest_dist = float('inf')
            for t, obj, dist, ang in vision:
                if t == "creature" and dist < nearest_dist:
                    nearest_creature = obj
                    nearest_dist = dist
            if nearest_creature:
                return "ATTACK", nearest_creature
            else:
                return "REST", None

        elif action_type == "FLEE":
            nearest_creature = None
            nearest_dist = float('inf')
            nearest_angle = 0.0
            for t, obj, dist, ang in vision:
                if t == "creature" and dist < nearest_dist:
                    nearest_creature = obj
                    nearest_dist = dist
                    nearest_angle = ang
            if nearest_creature:
                flee_angle = nearest_angle + np.pi
                dx = np.cos(flee_angle) * velocity
                dy = np.sin(flee_angle) * velocity
                return "FLEE", (dx, dy)
            else:
                return "REST", None

        # Default fallback
        return "REST", None

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.2) -> 'NeuralNetwork':
        """
        Create a mutated copy of this network’s weights for evolutionary purposes.

        Returns:
            A new NeuralNetwork instance with the same architecture but mutated weights.
        """
        weights = self.model.get_weights()  # List[np.ndarray] of weight matrices + bias vectors
        new_weights = []

        for w in weights:
            w_copy = w.copy()
            mask = np.random.random(w_copy.shape) < mutation_rate
            mutations = np.random.randn(*w_copy.shape) * mutation_scale
            w_copy[mask] += mutations[mask]
            new_weights.append(w_copy)

        child = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size, self.lr)
        child.model.set_weights(new_weights)
        return child

    def clone(self) -> 'NeuralNetwork':
        """
        Produce an exact copy of this network, including all weights.

        Returns:
            A new NeuralNetwork instance with identical weights to self.
        """
        weights = self.model.get_weights()
        clone = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size, self.lr)
        clone.model.set_weights([w.copy() for w in weights])
        return clone
