import unittest
import numpy as np
import tensorflow as tf
import sys, os
import logging
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestNeuralNetworkLearning(unittest.TestCase):

    def test_move_toward_food_sequence(self):
        """Train on short sequences requiring movement toward food then evaluate on a new scenario."""
        tf.random.set_seed(2)
        np.random.seed(2)
        random.seed(2)
        nn = NeuralNetwork(
            input_size=14, hidden_sizes=[8], output_size=8, learning_rate=0.1
        )

        def build_state(creature_pos, food_pos, energy):
            dx, dy = food_pos[0] - creature_pos[0], food_pos[1] - creature_pos[1]
            dist = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx) if dist > 0 else 0.0
            return {
                "vision": [("food", object(), dist, angle)],
                "on_food": dist < 0.5,
                "creature_state": {
                    "energy": energy,
                    "size": 1.0,
                    "velocity": 1.0,
                    "max_energy": 100.0,
                    "position": creature_pos,
                },
            }

        train_pairs = [((0.0, 0.0), (6.0, 0.0)), ((-4.0, 2.0), (1.0, 2.0))]

        for _ in range(300):
            for start, food in train_pairs:
                pos = np.array(start)
                energy = 10.0
                for _ in range(10):
                    state = build_state(pos, food, energy)
                    dist = np.hypot(food[0] - pos[0], food[1] - pos[1])
                    if dist < 0.5:
                        vec = nn._process_sensory_inputs(state)
                        nn.train_supervised(vec, 2)  # EAT
                        break
                    vec = nn._process_sensory_inputs(state)
                    nn.train_supervised(vec, 0)  # MOVE
                    step = np.array(
                        [
                            np.cos(
                                angle := np.arctan2(food[1] - pos[1], food[0] - pos[0])
                            ),
                            np.sin(angle),
                        ]
                    )
                    pos += step
                    energy -= 0.1

        # Evaluation on unseen scenario
        pos = np.array([3.0, -3.0])
        food = np.array([9.0, -3.0])
        energy = 10.0
        success = False
        for _ in range(10):
            state = build_state(tuple(pos), tuple(food), energy)
            vec = nn._process_sensory_inputs(state)
            logits = nn.model(np.expand_dims(vec, 0))
            probs = tf.nn.softmax(logits, axis=-1).numpy().flatten()
            act = int(np.argmax(probs[:6]))
            if act == 2 and np.hypot(*(food - pos)) < 0.5:
                success = True
                break
            if act == 0:
                direction = food - pos
                step = direction / np.linalg.norm(direction)
                pos += step
                energy -= 0.1
            else:
                break
        self.assertTrue(success, "NN failed to move to food and eat in new scenario")


if __name__ == "__main__":
    unittest.main()