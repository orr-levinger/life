# ===============================================================
#   src/tf_network.py
# ===============================================================
import tensorflow as tf
import numpy as np
from typing import Dict, Any

class TFNeuralNetwork:
    """
    A small policy-network in tf.keras. We do:
      inputs → Dense(hidden1, ReLU) → Dense(hidden2, ReLU) → Dense(output_size, softmax)
    Then train with REINFORCE: log-prob(action) * reward.
    """
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.lr = learning_rate

        # Build a Sequential model
        layers = []
        # Hidden layers
        for h in hidden_sizes:
            layers.append(tf.keras.layers.Dense(h, activation='relu'))
        # Output layer (logits → softmax later in training loop)
        layers.append(tf.keras.layers.Dense(output_size, activation=None))
        self.model = tf.keras.Sequential(layers)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # For tracking last input + action
        self._last_logits = None    # raw logits tensor
        self._last_action_idx = None
        self._last_input = None     # shape (1, input_size)

    def decide(self, sensory_inputs: Dict[str, Any]) -> (str, Any):
        """
        1) Build a batch of size 1 input vector from sensory_inputs  
        2) Run model → logits → softmax(probs)  
        3) Pick argmax as action, store index for training  
        4) Return (action_type, action_param) just like before
        """
        # 1) process inputs → flat vector
        input_vec = self._process_sensory_inputs(sensory_inputs)  # np.array shape=(input_size,)
        input_batch = np.expand_dims(input_vec, axis=0).astype(np.float32)  # shape=(1, input_size)
        self._last_input = input_batch

        # 2) forward pass
        logits = self.model(input_batch)                     # shape=(1, output_size)
        probs = tf.nn.softmax(logits, axis=-1)               # shape=(1, output_size)
        probs_np = probs.numpy().flatten()                   # shape=(output_size,)
        self._last_logits = logits

        # 3) choose action_idx = argmax(probs)
        action_idx = int(np.argmax(probs_np[:6]))  # first 6 for action types
        self._last_action_idx = action_idx

        # 4) map to (action_type, action_params) exactly like in NumPy version
        action_type, action_params = self._map_output_to_action(probs_np, sensory_inputs)
        return action_type, action_params

    def apply_reward(self, reward: float):
        """
        Perform one REINFORCE gradient update:
          loss = −log π(a|s) * reward
        """
        if self._last_input is None or self._last_logits is None or self._last_action_idx is None:
            return

        with tf.GradientTape() as tape:
            # Recompute logits for the stored input
            logits = self.model(self._last_input)                  # shape=(1, output_size)
            probs = tf.nn.softmax(logits, axis=-1)                  # shape=(1, output_size)
            # Extract probability of the chosen action
            chosen_prob = probs[0, self._last_action_idx]           # scalar
            # loss = −log π(a|s) * reward
            loss = -tf.math.log(chosen_prob + 1e-8) * reward

        # Compute gradients and apply
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Clear stored state
        self._last_input = None
        self._last_logits = None
        self._last_action_idx = None

    def _process_sensory_inputs(self, sensory_inputs: Dict[str, Any]) -> np.ndarray:
        """
        Same as your NumPy version: flatten creature_state, vision, etc., into a 1D np.array.
        """
        creature_state = sensory_inputs.get('creature_state', {})
        inputs = [
            creature_state.get('energy', 0) / 100.0,
            creature_state.get('size', 1.0) / 5.0,
            creature_state.get('velocity', 1.0),
            1.0 if sensory_inputs.get('on_food', False) else 0.0
        ]

        vision = sensory_inputs.get('vision', [])
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

        nearest_creature_dist = min(nearest_creature_dist, 10.0) / 10.0
        nearest_food_dist = min(nearest_food_dist, 10.0) / 10.0

        inputs.extend([
            num_creatures / 10.0,
            num_food / 10.0,
            nearest_creature_dist,
            np.cos(nearest_creature_angle),
            np.sin(nearest_creature_angle),
            nearest_food_dist,
            np.cos(nearest_food_angle),
            np.sin(nearest_food_angle)
        ])

        return np.array(inputs, dtype=np.float32)

    def _map_output_to_action(self, output_probs: np.ndarray, sensory_inputs: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Exactly the same mapping logic as in your NumPy _map_output_to_action.
        """
        action_types = ["MOVE", "REST", "EAT", "EAT_AT_CURRENT", "ATTACK", "FLEE"]
        action_type = action_types[self._last_action_idx]
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
            for t, obj, dist, angle in vision:
                if t == "food" and dist < nearest_dist:
                    nearest_food = obj
                    nearest_dist = dist
            return ("EAT", nearest_food) if nearest_food else ("REST", None)

        elif action_type == "EAT_AT_CURRENT":
            return ("EAT_AT_CURRENT", None) if on_food else ("REST", None)

        elif action_type == "ATTACK":
            nearest_creature = None
            nearest_dist = float('inf')
            for t, obj, dist, angle in vision:
                if t == "creature" and dist < nearest_dist:
                    nearest_creature = obj
                    nearest_dist = dist
            return ("ATTACK", nearest_creature) if nearest_creature else ("REST", None)

        elif action_type == "FLEE":
            nearest_creature = None
            nearest_dist = float('inf')
            nearest_angle = 0.0
            for t, obj, dist, angle in vision:
                if t == "creature" and dist < nearest_dist:
                    nearest_creature = obj
                    nearest_dist = dist
                    nearest_angle = angle
            if nearest_creature:
                flee_angle = nearest_angle + np.pi
                dx = np.cos(flee_angle) * velocity
                dy = np.sin(flee_angle) * velocity
                return "FLEE", (dx, dy)
            else:
                return "REST", None

        return "REST", None
