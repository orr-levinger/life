import math
import unittest

import numpy as np

from src.brain import N_SENSES
from src.learner import LEARNING_RATE, Learner, NetWeights, gae


def learner_with(slots: int, seed: int = 0, lr: float = LEARNING_RATE) -> Learner:
    learner = Learner(slots, seed)
    rng = np.random.default_rng(seed)
    for _ in range(slots):
        learner.allocate(NetWeights.random(rng), lr)
    return learner


def same(a: NetWeights, b: NetWeights) -> bool:
    return all(np.allclose(x, y) for x, y in zip(a.arrays(), b.arrays()))


def food_senses(angles: np.ndarray) -> np.ndarray:
    senses = np.zeros((len(angles), N_SENSES))
    senses[:, 3] = 1.0
    senses[:, 4] = 0.5
    senses[:, 5] = np.cos(angles)
    senses[:, 6] = np.sin(angles)
    return senses


def toy_rewards(actions, angles: np.ndarray) -> np.ndarray:
    """Reward = cosine between the move and the food direction; eating or attacking earns nothing."""
    rewards = []
    for (choice, (dx, dy, _)), angle in zip(actions, angles):  # the third: a call's volume
        length = math.hypot(dx, dy)
        rewards.append((dx * math.cos(angle) + dy * math.sin(angle)) / length if choice == 0 and length > 0 else 0.0)
    return np.array(rewards)


def live(learner: Learner, slot: int, steps: int) -> None:
    """`steps` steps of life with something to sense (all-zero senses would leave the input weights untouched)."""
    for _ in range(steps):
        learner.act([slot], np.full((1, N_SENSES), 0.5), attacks_on=True)
        learner.record(slot, 1.0, False)


def outputs(weights: NetWeights, senses: np.ndarray):
    """What a network says for each of `senses`: its choice odds and its move."""
    import src.learner as L
    import tensorflow as tf
    learner = Learner(1, 0)
    learner.allocate(weights, LEARNING_RATE)
    p = {name: tf.gather(var, [0]) for name, var in learner._params.items()}
    logits, mean, _, _ = L._heads(p, tf.constant(senses[None], tf.float32), tf.ones((1, len(senses))),
                                  tf.zeros((1, len(senses))),  # no calls
                                  tf.constant([weights.size], tf.float32))
    return logits.numpy(), mean.numpy()


class TestBrainSize(unittest.TestCase):
    def senses(self):
        return np.random.default_rng(1).uniform(-1, 1, (100, N_SENSES))

    def test_new_neuron_is_silent_at_birth(self):
        parent = NetWeights.random(np.random.default_rng(2))
        self.assertEqual(list(parent.size), [1, 32, 0, 0])  # the first animals: 1 layer of 32
        baby = parent.with_neuron(0, on=True)
        self.assertEqual(list(baby.size), [1, 33, 0, 0])
        for a, b in zip(outputs(parent, self.senses()), outputs(baby, self.senses())):
            self.assertTrue(np.allclose(a, b, atol=1e-6))  # acts exactly like its parent at birth

    def test_new_layer_is_silent_at_birth(self):
        parent = NetWeights.random(np.random.default_rng(3))
        baby = parent.with_layer(on=True)
        self.assertEqual(list(baby.size), [2, 32, 32, 0])
        for a, b in zip(outputs(parent, self.senses()), outputs(baby, self.senses())):
            self.assertTrue(np.allclose(a, b, atol=1e-6))

    def test_removed_neuron_changes_behavior(self):
        parent = NetWeights.random(np.random.default_rng(4))
        parent = NetWeights(**{**parent.__dict__, "wm": np.random.default_rng(5).normal(0, 1, parent.wm.shape)})
        baby = parent.with_neuron(0, on=False)
        self.assertEqual(list(baby.size), [1, 31, 0, 0])
        self.assertFalse(np.allclose(outputs(parent, self.senses())[1], outputs(baby, self.senses())[1]))

    def test_size_genes_stay_in_bounds(self):
        import random
        rng, brain = random.Random(6), NetWeights.random(np.random.default_rng(6))
        for _ in range(1000):
            brain = brain.grown(rng, layer_share=0.3, neurons=(1, 64), layers=(1, 3))
            layers, *neurons = (int(v) for v in brain.size)
            self.assertTrue(1 <= layers <= 3)
            self.assertTrue(all(1 <= n <= 64 for n in neurons[:layers]))
            self.assertTrue(all(n == 0 for n in neurons[layers:]))

    def test_a_learner_says_how_many_neurons_it_has(self):
        learner = Learner(1, 0)
        slot = learner.allocate(NetWeights.random(np.random.default_rng(7)).with_layer(on=True), LEARNING_RATE)
        self.assertEqual(learner.active_neurons(slot), 64)  # 32 + 32


class TestBrainGenes(unittest.TestCase):
    def test_layers_neurons_and_weights_are_separate_genes(self):
        import random
        from src.rules import starting_rules
        rules = starting_rules()
        for kind in (None, "layers", "neurons", "weights"):
            learner = Learner(2, 0)
            parent = learner.allocate(NetWeights.random(np.random.default_rng(8)), LEARNING_RATE)
            baby = learner.baby(parent, np.random.default_rng(9), LEARNING_RATE, "learned", [kind] if kind else [], random.Random(3), rules)
            before, after = learner.current(parent), learner.current(baby)
            sizes = (list(before.size), list(after.size))
            moved = sum(int((a != b).sum()) for a, b in zip(before.arrays()[:-1], after.arrays()[:-1]))
            total = sum(a.size for a in before.arrays()[:-1])
            if kind is None:
                self.assertEqual((sizes[0], moved), (sizes[1], 0))  # an exact copy
            elif kind == "layers":
                self.assertNotEqual(sizes[0][0], sizes[1][0])
                self.assertEqual(sizes[0][1], sizes[1][1])
            elif kind == "neurons":
                self.assertEqual(sizes[0][0], sizes[1][0])
                self.assertEqual(abs(sizes[1][1] - sizes[0][1]), 1)  # one neuron more or fewer
            else:
                self.assertEqual(sizes[0], sizes[1])
                self.assertTrue(0.01 * total < moved < 0.03 * total)  # about 2 % of its weights
                largest = max(float(np.abs(a - b).max()) for a, b in zip(before.arrays()[:-1], after.arrays()[:-1]))
                self.assertLess(largest, 0.15)  # each by a little (normal noise of 0.02)

    def test_learning_reads_its_rules(self):
        from src.rules import starting_rules
        learner = Learner(1, 0)
        learner.rules = {**starting_rules(), "learn_rollout": 16.0}
        slot = learner.allocate(NetWeights.random(np.random.default_rng(1)), LEARNING_RATE)
        live(learner, slot, 17)
        self.assertEqual(learner.update(), 1)  # a round after 16 steps, not 64


class TestLearner(unittest.TestCase):
    def test_the_spreads_stay_bounded(self):
        import src.learner as L
        import tensorflow as tf
        learner = learner_with(1)
        learner._params["log_std"][0].assign([50.0, -50.0, 50.0])  # pushed far by exploring, never used
        p = {name: tf.gather(var, [0]) for name, var in learner._params.items()}
        _, _, log_std, _ = L._heads(p, tf.zeros((1, 1, N_SENSES)), tf.ones((1, 1)), tf.zeros((1, 1)),
                                    tf.constant([[1.0, 32.0, 0.0, 0.0]]))
        self.assertTrue(np.all(np.abs(log_std.numpy()) <= max(abs(L.LOG_STD_LOW), abs(L.LOG_STD_HIGH))))
        moves = [m for _, m in learner.act([0], np.zeros((1, N_SENSES)), True, learning=False, calls=[1])]
        self.assertTrue(np.all(np.isfinite(moves)))

    def test_act_shapes_and_ranges(self):
        learner = learner_with(4)
        actions = learner.act([0, 1, 2, 3], np.zeros((4, N_SENSES)), attacks_on=True)
        self.assertEqual(len(actions), 4)
        for choice, (dx, dy, volume) in actions:
            self.assertIn(choice, (0, 1, 2))  # move, eat or attack: no calls in its genes
            self.assertTrue(-1.0 <= dx <= 1.0 and -1.0 <= dy <= 1.0 and 0.0 <= volume <= 1.0)

    def test_attack_masked(self):
        learner = learner_with(10)
        choices = [choice for _ in range(100)
                   for choice, _ in learner.act(list(range(10)), np.zeros((10, N_SENSES)), attacks_on=False, learning=False)]
        self.assertEqual(set(choices), {0, 1})  # move or eat: attack is masked
        allowed = [choice for _ in range(100)
                   for choice, _ in learner.act(list(range(10)), np.zeros((10, N_SENSES)), attacks_on=True, learning=False)]
        self.assertEqual(set(allowed), {0, 1, 2})

    def test_gae_matches_hand_example(self):
        values = np.array([0.5, 0.2, 0.4])
        adv, ret = gae(np.array([1.0, 0.0, 2.0]), values, np.array([0.0, 0.0, 0.0]), 1.0, gamma=0.5, lam=1.0)
        np.testing.assert_allclose(adv, [1.125, 1.05, 2.1])
        np.testing.assert_allclose(ret, [1.625, 1.25, 2.5])
        adv, _ = gae(np.array([1.0, 0.0, 2.0]), values, np.array([0.0, 1.0, 0.0]), 1.0, gamma=0.5, lam=1.0)
        np.testing.assert_allclose(adv, [0.5, -0.2, 2.1])

    def test_allocate_and_release(self):
        learner = Learner(3, 0)
        rng = np.random.default_rng(0)
        nets = [NetWeights.random(rng) for _ in range(4)]
        self.assertEqual([learner.allocate(n, 1e-3 * (i + 1)) for i, n in enumerate(nets[:3])], [0, 1, 2])
        self.assertAlmostEqual(learner.lr(1), 2e-3)
        self.assertTrue(same(learner.inborn(2), nets[2]))
        with self.assertRaises(RuntimeError):
            learner.allocate(nets[3], 1e-3)
        learner.release(1)
        self.assertEqual(learner.allocate(nets[3], 5e-3), 1)
        self.assertTrue(same(learner.current(1), nets[3]))
        self.assertAlmostEqual(learner.lr(1), 5e-3)

    def test_update_changes_only_full_slots(self):
        learner = learner_with(2)
        for step in range(65):
            learner.act([0, 1], np.zeros((2, N_SENSES)), attacks_on=True)
            learner.record(0, 1.0, False)
            if step < 10:
                learner.record(1, 1.0, False)
        self.assertEqual(learner.update(), 1)
        self.assertFalse(same(learner.current(0), learner.inborn(0)))
        self.assertTrue(same(learner.current(1), learner.inborn(1)))

    def test_slot_reuse_starts_clean(self):
        learner = learner_with(1)
        live(learner, 0, 65)
        learner.update()  # the first animal has learned, and one step of its experience is left over
        learner.release(0)
        fresh = NetWeights.random(np.random.default_rng(99))
        self.assertEqual(learner.allocate(fresh, LEARNING_RATE), 0)
        self.assertTrue(same(learner.current(0), fresh))
        self.assertTrue(same(learner.inborn(0), fresh))
        live(learner, 0, 64)
        self.assertEqual(learner.update(), 0)  # the dead animal's leftover step did not count for the newborn
        live(learner, 0, 1)
        self.assertEqual(learner.update(), 1)

    def one_lesson(self, learner, slot, behaviour_shift):
        """A lesson for `slot`: a move it judges with advantage 1.5 and a target of 2.0; its odds of that move against
        the actor's are e^behaviour_shift."""
        import src.learner as L
        import tensorflow as tf
        p = {name: tf.gather(var, [slot]) for name, var in learner._params.items()}
        obs = tf.constant(np.full((1, 1, N_SENSES), 0.4), tf.float32)
        attacks = tf.ones((1, 1))
        calls = tf.zeros((1, 1))
        choice, move = tf.zeros((1, 1), tf.int32), tf.constant([[[0.3, -0.2, 0.5]]], tf.float32)
        logits, mean, log_std, value = L._heads(p, obs, attacks, calls, tf.constant([[1.0, 32.0, 0.0, 0.0]]))
        own = L._log_prob(logits, mean, log_std, choice, move)
        lb = {"obs": obs, "attacks": attacks, "calls": calls, "choice": choice, "move": move, "logp": own - behaviour_shift,
              "adv": tf.constant([[1.5]]), "target": tf.constant([[2.0]]), "mask": tf.ones((1, 1)),
              "weight": tf.constant([[0.5]]), "size": tf.constant([[1.0, 32.0, 0.0, 0.0]])}  # it learns half as much from others
        expected = 0.5 * (-1.5 + 0.5 * (float(value[0, 0]) - 2.0) ** 2) / 64  # its own value weight 0.5, a 64-step round
        return p, lb, expected

    def test_lessons_train_at_half_weight(self):
        import src.learner as L
        learner = learner_with(2)
        for _ in range(70):  # slot 1 lives 70 steps, each a lesson for slot 0, which lives only 10
            learner.act([0, 1], np.full((2, N_SENSES), 0.5), attacks_on=True)
            learner.record(1, 1.0, False)
            learner.watch(0, 1, reward=1.0, weight=0.5)
            if len(learner._buffers[0]) < 10:
                learner.record(0, 1.0, False)
            else:
                learner._pending.pop(0)
        learner._buffers[1] = []  # only slot 0 is looked at
        self.assertEqual(learner.update(), 0)
        self.assertTrue(same(learner.current(0), learner.inborn(0)))  # lessons alone change nothing

        p, lb, expected = self.one_lesson(learner, 0, 0.0)  # the watcher would make the move as likely as the actor
        self.assertAlmostEqual(float(L.lesson_loss(p, lb, 0.5, 64)[0]), expected, places=5)  # half an own step's loss

        trained = []
        for lessons in (False, True):
            learner = learner_with(2, seed=3)
            for _ in range(65):
                learner.act([0, 1], np.full((2, N_SENSES), 0.5), attacks_on=True)
                learner.record(0, 1.0, False)
                learner.record(1, -1.0, False)
                if lessons:
                    learner.watch(0, 1, reward=1.0, weight=0.5)
            learner._buffers[1] = learner._buffers[1][:10]
            learner.update()
            trained.append(learner.current(0))
        self.assertFalse(same(*trained))  # a round with a full rollout trains on the lessons too

    def test_a_lesson_counts_no_more_than_an_own_step(self):
        import src.learner as L
        learner = learner_with(1)
        _, _, expected = self.one_lesson(learner, 0, 0.0)
        p, lb, _ = self.one_lesson(learner, 0, 5.0)  # the watcher would make it e^5 times likelier than the actor
        self.assertAlmostEqual(float(L.lesson_loss(p, lb, 0.5, 64)[0]), expected, places=5)
        self.assertAlmostEqual(float(L.lesson_weight(math.exp(5.0), 0.5)), 0.5)
        self.assertAlmostEqual(float(L.lesson_weight(0.2, 0.5)), 0.1)

    def test_unfinished_lessons_are_not_saved(self):
        learner = learner_with(2)
        learner.act([0, 1], np.full((2, N_SENSES), 0.5), attacks_on=True)
        learner.record(1, 1.0, False)
        learner.watch(0, 1, reward=1.0, weight=0.5)
        self.assertEqual(len(learner._lessons[0]), 1)
        again = Learner(2, 0)
        again.load_state(learner.state())
        self.assertEqual(again._lessons, [[], []])

    def test_same_seed_same_actions(self):
        runs = []
        for _ in range(2):
            learner = learner_with(3, seed=5)
            runs.append([learner.act([0, 1, 2], np.ones((3, N_SENSES)) * 0.3, attacks_on=True) for _ in range(20)])
        self.assertEqual(runs[0], runs[1])

    def test_learns_toy_task(self):
        slots = list(range(8))
        learner = learner_with(8, seed=1)
        rng = np.random.default_rng(2)

        def mean_reward(calls: int) -> float:
            total = []
            for _ in range(calls):
                angles = rng.uniform(-math.pi, math.pi, 8)
                total.extend(toy_rewards(learner.act(slots, food_senses(angles), False, learning=False), angles))
            return float(np.mean(total))

        before = mean_reward(80)
        updates = 0
        while updates < 450 * 8:  # measured at the spec's learning rate: 0.77 after 300 updates and still rising
            angles = rng.uniform(-math.pi, math.pi, 8)
            rewards = toy_rewards(learner.act(slots, food_senses(angles), attacks_on=False), angles)
            for slot, reward in zip(slots, rewards):
                learner.record(slot, float(reward), False)
            updates += learner.update()
        after = mean_reward(80)
        self.assertLess(abs(before), 0.15)
        self.assertGreaterEqual(after, 0.8)


if __name__ == "__main__":
    unittest.main()
