"""Learning during life: each creature's own actor-critic network, trained with PPO. TensorFlow lives only here.

All slots' networks are stored together, so one TensorFlow call acts for every learning creature and one updates
every slot whose rollout is full. Each slot keeps its own weights, its own Adam state and its own learning rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .brain import jitter
from .settings import MUTATION_RATE, MUTATION_SCALE

N_IN = 12
N_HIDDEN = 32
N_CHOICES = 3  # 0 move, 1 eat, 2 attack
ATTACK = 2
ROLLOUT = 64
LEARNING_RATE = 3e-4  # the Learning team's rate, and the first generation's in the Both team
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 4
VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
INITIAL_SPREAD = 0.5  # starting standard deviation of the move
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
MASKED = -1e9
LOG_2PI = math.log(2.0 * math.pi)

SHAPES: Dict[str, Tuple[int, ...]] = {
    "w1": (N_IN, N_HIDDEN), "b1": (N_HIDDEN,),
    "wc": (N_HIDDEN, N_CHOICES), "bc": (N_CHOICES,),
    "wm": (N_HIDDEN, 2), "bm": (2,), "log_std": (2,),
    "wh": (N_IN, N_HIDDEN), "bh": (N_HIDDEN,),  # the critic's own hidden layer: its large errors never reach the actor's
    "wv": (N_HIDDEN, 1), "bv": (1,),
}


@dataclass(frozen=True)
class NetWeights:
    """One creature's actor-critic network: the actor's hidden layer, choice head, move head and spread; the critic's
    own hidden layer and value head."""
    w1: np.ndarray
    b1: np.ndarray
    wc: np.ndarray
    bc: np.ndarray
    wm: np.ndarray
    bm: np.ndarray
    log_std: np.ndarray
    wh: np.ndarray
    bh: np.ndarray
    wv: np.ndarray
    bv: np.ndarray

    @classmethod
    def random(cls, np_rng: np.random.Generator) -> NetWeights:
        """Hidden and critic weights scaled 1/sqrt(inputs); the policy heads start small so the first odds are even."""
        return cls(
            w1=np_rng.normal(0.0, 1.0 / math.sqrt(N_IN), SHAPES["w1"]), b1=np.zeros(N_HIDDEN),
            wc=np_rng.normal(0.0, 0.01, SHAPES["wc"]), bc=np.zeros(N_CHOICES),
            wm=np_rng.normal(0.0, 0.01, SHAPES["wm"]), bm=np.zeros(2),
            log_std=np.full(2, math.log(INITIAL_SPREAD)),
            wh=np_rng.normal(0.0, 1.0 / math.sqrt(N_IN), SHAPES["wh"]), bh=np.zeros(N_HIDDEN),
            wv=np_rng.normal(0.0, 1.0 / math.sqrt(N_HIDDEN), SHAPES["wv"]), bv=np.zeros(1),
        )

    def arrays(self) -> Tuple[np.ndarray, ...]:
        return tuple(getattr(self, f.name) for f in fields(self))

    def mutate(self, np_rng: np.random.Generator, rate: float, scale: float) -> NetWeights:
        return NetWeights(*(jitter(values, np_rng, rate, scale) for values in self.arrays()))

    def save(self, path: str | Path) -> None:
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)})


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float,
        gamma: float = GAMMA, lam: float = LAM) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation. dones[t] = 1 means the life ended after step t (nothing bootstraps past it)."""
    advantages = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        running = delta + gamma * lam * alive * running
        advantages[t] = running
    return advantages, advantages + values


def _heads(p: Dict[str, tf.Tensor], x: tf.Tensor, attacks: tf.Tensor):
    """x [K, T, 12], attacks [K, T] (1 allowed, 0 masked) -> choice logits, move mean, log spread, critic value."""
    h = tf.tanh(tf.einsum("kti,kih->kth", x, p["w1"]) + p["b1"][:, None, :])
    logits = tf.einsum("kth,khc->ktc", h, p["wc"]) + p["bc"][:, None, :]
    logits += MASKED * (1.0 - attacks)[..., None] * tf.one_hot(ATTACK, N_CHOICES)
    mean = tf.tanh(tf.einsum("kth,khm->ktm", h, p["wm"]) + p["bm"][:, None, :])
    log_std = tf.broadcast_to(p["log_std"][:, None, :], tf.shape(mean))
    critic_h = tf.tanh(tf.einsum("kti,kih->kth", x, p["wh"]) + p["bh"][:, None, :])
    value = tf.einsum("kth,khv->ktv", critic_h, p["wv"])[..., 0] + p["bv"][:, None, 0]
    return logits, mean, log_std, value


def _log_prob(logits, mean, log_std, choice, move):
    choice_logp = tf.reduce_sum(tf.nn.log_softmax(logits) * tf.one_hot(choice, N_CHOICES), axis=-1)
    move_logp = tf.reduce_sum(-0.5 * tf.square((move - mean) / tf.exp(log_std)) - log_std - 0.5 * LOG_2PI, axis=-1)
    return choice_logp + tf.cast(tf.equal(choice, 0), tf.float32) * move_logp


def _entropy(logits, log_std):
    probs = tf.nn.softmax(logits)
    choice_entropy = -tf.reduce_sum(probs * tf.nn.log_softmax(logits), axis=-1)
    return choice_entropy + tf.reduce_sum(log_std + 0.5 * (LOG_2PI + 1.0), axis=-1)


class Learner:
    def __init__(self, slots: int, seed: int) -> None:
        tf.config.experimental.enable_op_determinism()
        self.slots = slots
        self._gen = tf.random.Generator.from_seed(seed)
        self._params = {n: tf.Variable(tf.zeros((slots,) + s)) for n, s in SHAPES.items()}
        self._m = {n: tf.Variable(tf.zeros((slots,) + s)) for n, s in SHAPES.items()}
        self._v = {n: tf.Variable(tf.zeros((slots,) + s)) for n, s in SHAPES.items()}
        self._t = np.zeros(slots)
        self._lr = np.zeros(slots)
        self._inborn: List[Optional[NetWeights]] = [None] * slots
        self._buffers: List[List[dict]] = [[] for _ in range(slots)]
        self._pending: Dict[int, dict] = {}
        self._free: List[int] = list(range(slots))

    def allocate(self, weights: NetWeights, lr: float) -> int:
        """A newborn's network in the lowest free slot: these are its inborn weights, and its optimizer state and
        experience start empty (nothing of the slot's last animal is left)."""
        if not self._free:
            raise RuntimeError(f"all {self.slots} learner slots are taken")
        slot = self._free.pop(0)
        for name in SHAPES:
            self._params[name][slot].assign(tf.cast(getattr(weights, name), tf.float32))
            self._m[name][slot].assign(tf.zeros(SHAPES[name]))
            self._v[name][slot].assign(tf.zeros(SHAPES[name]))
        self._t[slot] = 0
        self._lr[slot] = lr
        self._inborn[slot] = weights
        self._buffers[slot] = []
        self._pending.pop(slot, None)
        return slot

    def release(self, slot: int) -> None:
        """The slot of an animal that died is free for a newborn."""
        self._free.append(slot)
        self._free.sort()

    def has_room(self) -> bool:
        return bool(self._free)

    def newcomer(self, np_rng: np.random.Generator) -> int:
        """A newcomer of the primordial soup: a random network at the standard learning speed. Returns its slot."""
        return self.allocate(NetWeights.random(np_rng), LEARNING_RATE)

    def baby(self, parent_slot: int, np_rng: np.random.Generator, lr_factor: float, inherit: str) -> int:
        """A baby's network: the parent's brain as it is now ("learned") or as the parent was born ("inborn"),
        mutated, with the parent's learning speed × `lr_factor`. Returns the baby's slot."""
        genes = self.current(parent_slot) if inherit == "learned" else self.inborn(parent_slot)
        return self.allocate(genes.mutate(np_rng, MUTATION_RATE, MUTATION_SCALE), self.lr(parent_slot) * lr_factor)

    def lr(self, slot: int) -> float:
        return float(self._lr[slot])

    def inborn(self, slot: int) -> NetWeights:
        weights = self._inborn[slot]
        if weights is None:
            raise ValueError(f"slot {slot} has no network")
        return weights

    def current(self, slot: int) -> NetWeights:
        return NetWeights(**{name: self._params[name][slot].numpy() for name in SHAPES})

    def act(self, slots: Sequence[int], senses: np.ndarray, attacks_on: bool,
            learning: bool = True) -> List[Tuple[int, Tuple[float, float]]]:
        """One call for all `slots`: each rolls its own dice for a choice and a move (each axis in -1..1)."""
        idx = tf.constant(list(slots), dtype=tf.int32)
        p = {name: tf.gather(var, idx) for name, var in self._params.items()}
        x = tf.constant(senses[:, None, :], dtype=tf.float32)
        attacks = tf.fill([len(slots), 1], 1.0 if attacks_on else 0.0)
        logits, mean, log_std, value = _heads(p, x, attacks)
        gumbel = -tf.math.log(-tf.math.log(self._gen.uniform(tf.shape(logits), 1e-12, 1.0)))
        choice = tf.argmax(logits + gumbel, axis=-1, output_type=tf.int32)
        move = mean + tf.exp(log_std) * self._gen.normal(tf.shape(mean))
        logp = _log_prob(logits, mean, log_std, choice, move)

        choice_np, move_np = choice.numpy()[:, 0], move.numpy()[:, 0, :]
        if learning:
            logp_np, value_np = logp.numpy()[:, 0], value.numpy()[:, 0]
            for i, slot in enumerate(slots):
                self._pending[slot] = {"obs": senses[i], "choice": int(choice_np[i]), "move": move_np[i],
                                       "logp": float(logp_np[i]), "value": float(value_np[i]),
                                       "attacks": 1.0 if attacks_on else 0.0}
        return [(int(c), (float(np.clip(m[0], -1.0, 1.0)), float(np.clip(m[1], -1.0, 1.0))))
                for c, m in zip(choice_np, move_np)]

    def record(self, slot: int, reward: float, done: bool) -> None:
        step = self._pending.pop(slot, None)
        if step is None:
            raise ValueError(f"slot {slot} has no step waiting for a reward")
        step["reward"] = reward
        step["done"] = 1.0 if done else 0.0
        self._buffers[slot].append(step)

    def update(self) -> int:
        """One PPO update for every slot with a full rollout (64 steps plus the one that bootstraps them)."""
        full = [slot for slot in range(self.slots) if len(self._buffers[slot]) > ROLLOUT]
        if not full:
            return 0
        batch = {key: [] for key in ("obs", "choice", "move", "logp", "attacks", "adv", "ret")}
        for slot in full:
            steps = self._buffers[slot][:ROLLOUT]
            column = {key: np.array([s[key] for s in steps]) for key in steps[0]}
            adv, ret = gae(column["reward"], column["value"], column["done"], self._buffers[slot][ROLLOUT]["value"])
            batch["adv"].append((adv - adv.mean()) / (adv.std() + 1e-8))
            batch["ret"].append(ret)
            for key in ("obs", "choice", "move", "logp", "attacks"):
                batch[key].append(column[key])
            self._buffers[slot] = self._buffers[slot][ROLLOUT:]
        tensors = {key: tf.constant(np.stack(values), dtype=tf.int32 if key == "choice" else tf.float32)
                   for key, values in batch.items()}
        for _ in range(EPOCHS):
            self._ppo_step(full, tensors)
        return len(full)

    def _ppo_step(self, slots: List[int], b: Dict[str, tf.Tensor]) -> None:
        idx = tf.constant(slots, dtype=tf.int32)
        p = {name: tf.gather(var, idx) for name, var in self._params.items()}
        with tf.GradientTape() as tape:
            tape.watch(list(p.values()))
            logits, mean, log_std, value = _heads(p, b["obs"], b["attacks"])
            ratio = tf.exp(_log_prob(logits, mean, log_std, b["choice"], b["move"]) - b["logp"])
            clipped = tf.clip_by_value(ratio, 1.0 - CLIP, 1.0 + CLIP)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * b["adv"], clipped * b["adv"]), axis=1)
            value_loss = tf.reduce_mean(tf.square(value - b["ret"]), axis=1)
            entropy = tf.reduce_mean(_entropy(logits, log_std), axis=1)
            loss = tf.reduce_sum(policy_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * entropy)
        grads = tape.gradient(loss, list(p.values()))
        self._adam(slots, idx, p, dict(zip(p.keys(), grads)))

    def _adam(self, slots: List[int], idx: tf.Tensor, p: Dict[str, tf.Tensor], grads: Dict[str, tf.Tensor]) -> None:
        beta1, beta2 = ADAM_BETAS
        self._t[slots] += 1
        t = self._t[slots]
        scatter_at = idx[:, None]
        for name, grad in grads.items():
            per_slot = (len(slots),) + (1,) * len(SHAPES[name])
            m = beta1 * tf.gather(self._m[name], idx) + (1.0 - beta1) * grad
            v = beta2 * tf.gather(self._v[name], idx) + (1.0 - beta2) * tf.square(grad)
            m_hat = m / tf.constant((1.0 - beta1 ** t).reshape(per_slot), tf.float32)
            v_hat = v / tf.constant((1.0 - beta2 ** t).reshape(per_slot), tf.float32)
            lr = tf.constant(self._lr[slots].reshape(per_slot), tf.float32)
            self._params[name].scatter_nd_update(scatter_at, p[name] - lr * m_hat / (tf.sqrt(v_hat) + ADAM_EPS))
            self._m[name].scatter_nd_update(scatter_at, m)
            self._v[name].scatter_nd_update(scatter_at, v)
