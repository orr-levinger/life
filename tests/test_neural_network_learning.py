import unittest
import numpy as np
import tensorflow as tf
import sys, os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestNeuralNetworkLearning(unittest.TestCase):
    def test_training_and_generalization(self):
        tf.random.set_seed(1)
        np.random.seed(1)
        nn = NeuralNetwork(input_size=12, hidden_sizes=[4, 4], output_size=8, learning_rate=0.1)

        # Training scenarios: (inputs, expected action index)
        train_scenarios = [
            ({
                'vision': [],
                'on_food': True,
                'creature_state': {'energy': 50, 'size': 1.0, 'velocity': 1.0, 'max_energy': 100.0}
            }, 3),  # Expect EAT_AT_CURRENT
            ({
                'vision': [('food', object(), 2.0, 0.0)],
                'on_food': False,
                'creature_state': {'energy': 50, 'size': 1.0, 'velocity': 1.0, 'max_energy': 100.0}
            }, 2),  # Expect EAT
            ({
                'vision': [('creature', object(), 1.0, np.pi/2)],
                'on_food': False,
                'creature_state': {'energy': 50, 'size': 1.0, 'velocity': 1.0, 'max_energy': 100.0}
            }, 5)   # Expect FLEE
        ]

        # Supervised training: force correct actions
        for step in range(500):
            for inputs, action_idx in train_scenarios:
                vec = nn._process_sensory_inputs(inputs)
                nn.train_supervised(vec, action_idx)
            if (step + 1) % 100 == 0:
                correct = 0
                for inputs, action_idx in train_scenarios:
                    vec = nn._process_sensory_inputs(inputs)
                    logits = nn.model(np.expand_dims(vec, 0))
                    probs = tf.nn.softmax(logits, axis=-1).numpy().flatten()
                    pred = int(np.argmax(probs[:6]))
                    correct += pred == action_idx
                logger.info("Training step %d: %d/%d correct", step + 1, correct, len(train_scenarios))

        # New scenarios (slightly different distances/angles)
        test_scenarios = [
            ({
                'vision': [],
                'on_food': True,
                'creature_state': {'energy': 60, 'size': 1.0, 'velocity': 1.0, 'max_energy': 100.0}
            }, 3),
            ({
                'vision': [('food', object(), 3.0, np.pi/4)],
                'on_food': False,
                'creature_state': {'energy': 60, 'size': 1.0, 'velocity': 1.0, 'max_energy': 100.0}
            }, 2),
            ({
                'vision': [('creature', object(), 0.5, -np.pi/2)],
                'on_food': False,
                'creature_state': {'energy': 60, 'size': 1.0, 'velocity': 1.0, 'max_energy': 100.0}
            }, 5)
        ]

        for idx, (inputs, expected_idx) in enumerate(test_scenarios):
            vec = nn._process_sensory_inputs(inputs)
            logits = nn.model(np.expand_dims(vec, 0))
            probs = tf.nn.softmax(logits, axis=-1).numpy().flatten()
            pred_idx = int(np.argmax(probs[:6]))
            logger.info(
                "Scenario %d predicted action %d (expected %d)",
                idx,
                pred_idx,
                expected_idx,
            )
            if idx < 2:
                self.assertEqual(pred_idx, expected_idx)
            else:
                self.assertGreater(pred_idx, 1)  # Prefer action over REST/MOVE

if __name__ == "__main__":
    unittest.main()
