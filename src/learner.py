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
from .rules import rule, starting_rules

N_IN = 20
N_HIDDEN = 32
N_CHOICES = 3  # 0 move, 1 eat, 2 attack
ATTACK = 2
LEARNING_RATE = rule("gene.learning_speed.start").start  # a newcomer's, from the rules
INITIAL_SPREAD = rule("learn_spread").start  # a newborn's first spread, from the rules
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
MASKED = -1e9
LOG_2PI = math.log(2.0 * math.pi)

ROOM = 64  # neurons a layer has room for
MAX_LAYERS = 3
FIRST_LAYERS, FIRST_NEURONS = 1, 32  # the first animals' brain
N_CRITIC = 32  # the critic's own hidden layer: learning's machinery, not a gene

SHAPES: Dict[str, Tuple[int, ...]] = {
    "w1": (N_IN, ROOM), "b1": (ROOM,),
    "w2": (ROOM, ROOM), "b2": (ROOM,), "s2": (1,),  # layers 2 and 3 add to what they receive, × their strength s
    "w3": (ROOM, ROOM), "b3": (ROOM,), "s3": (1,),
    "wc": (ROOM, N_CHOICES), "bc": (N_CHOICES,),
    "wm": (ROOM, 2), "bm": (2,), "log_std": (2,),
    "wh": (N_IN, N_CRITIC), "bh": (N_CRITIC,),  # the critic's own hidden layer: its large errors never reach the actor's
    "wv": (N_CRITIC, 1), "bv": (1,),
}
_LAYER_WEIGHTS = (("w1", "b1"), ("w2", "b2"), ("w3", "b3"))


@dataclass(frozen=True)
class NetWeights:
    """One creature's actor-critic network: room for 3 actor layers of 64 neurons, its choice head, move head and
    spread; the critic's own hidden layer and value head. `size` is its brain's genes: [layers, neurons of layer 1, 2,
    3] (a layer not in use has 0); only those neurons and layers act."""
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    s2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    s3: np.ndarray
    wc: np.ndarray
    bc: np.ndarray
    wm: np.ndarray
    bm: np.ndarray
    log_std: np.ndarray
    wh: np.ndarray
    bh: np.ndarray
    wv: np.ndarray
    bv: np.ndarray
    size: np.ndarray

    @classmethod
    def random(cls, np_rng: np.random.Generator, layers: int = FIRST_LAYERS, neurons: int = FIRST_NEURONS,
               spread: float = INITIAL_SPREAD) -> NetWeights:
        """Hidden and critic weights scaled 1/sqrt(inputs); the policy heads start small so the first odds are even; a
        deeper layer starts with strength 0, so it says nothing until it learns or mutates."""
        size = np.array([layers] + [neurons if layer < layers else 0 for layer in range(MAX_LAYERS)], dtype=float)
        return cls(
            w1=np_rng.normal(0.0, 1.0 / math.sqrt(N_IN), SHAPES["w1"]), b1=np.zeros(ROOM),
            w2=np_rng.normal(0.0, 1.0 / math.sqrt(ROOM), SHAPES["w2"]), b2=np.zeros(ROOM), s2=np.zeros(1),
            w3=np_rng.normal(0.0, 1.0 / math.sqrt(ROOM), SHAPES["w3"]), b3=np.zeros(ROOM), s3=np.zeros(1),
            wc=np_rng.normal(0.0, 0.01, SHAPES["wc"]), bc=np.zeros(N_CHOICES),
            wm=np_rng.normal(0.0, 0.01, SHAPES["wm"]), bm=np.zeros(2),
            log_std=np.full(2, math.log(spread)),
            wh=np_rng.normal(0.0, 1.0 / math.sqrt(N_IN), SHAPES["wh"]), bh=np.zeros(N_CRITIC),
            wv=np_rng.normal(0.0, 1.0 / math.sqrt(N_CRITIC), SHAPES["wv"]), bv=np.zeros(1),
            size=size,
        )

    def arrays(self) -> Tuple[np.ndarray, ...]:
        return tuple(getattr(self, f.name) for f in fields(self))

    def _with(self, **changes) -> NetWeights:
        return NetWeights(**{**{f.name: getattr(self, f.name) for f in fields(self)}, **changes})

    def mutate(self, np_rng: np.random.Generator, rate: float, scale: float) -> NetWeights:
        """The weights changed a little (each with probability `rate`, by normal noise of std `scale`); its size kept."""
        return self._with(**{name: jitter(getattr(self, name), np_rng, rate, scale) for name in SHAPES})

    def with_neuron(self, layer: int, on: bool) -> NetWeights:
        """One more neuron in `layer` (0-based), silent at first (its incoming weights 0, so it says 0), or one fewer."""
        size = self.size.copy()
        if not on:
            size[1 + layer] -= 1
            return self._with(size=size)
        j = int(size[1 + layer])
        w_name, b_name = _LAYER_WEIGHTS[layer]
        w, bias = getattr(self, w_name).copy(), getattr(self, b_name).copy()
        w[:, j], bias[j] = 0.0, 0.0
        size[1 + layer] += 1
        return self._with(size=size, **{w_name: w, b_name: bias})

    def with_layer(self, on: bool) -> NetWeights:
        """One more layer (as many neurons as the first, strength 0, so it says nothing at first), or the last one gone."""
        size, layers = self.size.copy(), int(self.size[0])
        if on:
            size[0], size[1 + layers] = layers + 1, size[1]
            return self._with(size=size, **{f"s{layers + 1}": np.zeros(1)})
        size[0], size[layers] = layers - 1, 0.0
        return self._with(size=size)

    def grown(self, rng, layer_share: float, neurons: Tuple[float, float], layers: Tuple[float, float]) -> NetWeights:
        """One change to the brain's size, kept in its limits: with probability `layer_share` a layer added or taken
        away, else a neuron added or taken away in one of its layers."""
        count = int(self.size[0])
        if rng.random() < layer_share:
            choices = [on for on, ok in ((True, count < layers[1]), (False, count > layers[0])) if ok]
            return self.with_layer(rng.choice(choices)) if choices else self
        layer = rng.randrange(count)
        n = self.size[1 + layer]
        choices = [on for on, ok in ((True, n < neurons[1]), (False, n > neurons[0])) if ok]
        return self.with_neuron(layer, rng.choice(choices)) if choices else self

    def save(self, path: str | Path) -> None:
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)})


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float,
        gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
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


def _heads(p: Dict[str, tf.Tensor], x: tf.Tensor, attacks: tf.Tensor, size: tf.Tensor):
    """x [K, T, 20], attacks [K, T] (1 allowed, 0 masked), size [K, 4] (layers, neurons of each) -> choice logits, move
    mean, log spread, critic value. Only a creature's own neurons and layers act."""
    neuron = tf.range(ROOM, dtype=tf.float32)[None, :]
    mask = [tf.cast(neuron < size[:, 1 + layer:2 + layer], tf.float32)[:, None, :] for layer in range(MAX_LAYERS)]
    h = tf.tanh(tf.einsum("kti,kih->kth", x, p["w1"]) + p["b1"][:, None, :]) * mask[0]
    for layer, (w, b, s) in enumerate((("w2", "b2", "s2"), ("w3", "b3", "s3")), start=1):
        on = tf.cast(size[:, 0:1] > layer, tf.float32) * p[s]  # [K, 1]: its strength, 0 when it is not in use
        h = h + on[:, None, :] * tf.tanh(tf.einsum("kth,khg->ktg", h, p[w]) + p[b][:, None, :]) * mask[layer]
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


def lesson_weight(ratio, learns):
    """How much a lesson counts: the watcher's learning from others, times how likely it would have done the same
    against the actor (its odds ÷ the actor's), never more than 1."""
    return tf.cast(learns, tf.float32) * tf.minimum(1.0, tf.cast(ratio, tf.float32))


def lesson_loss(p: Dict[str, tf.Tensor], lb: Dict[str, tf.Tensor], value_weight: float, rollout: float) -> tf.Tensor:
    """Each slot's loss from its lessons [K]: per lesson, its weight times (minus the advantage, plus the value error);
    summed over a slot's lessons and divided by a rollout's steps, so a lesson counts against an own step's mean."""
    logits, mean, log_std, value = _heads(p, lb["obs"], lb["attacks"], lb["size"])
    logp = _log_prob(logits, mean, log_std, lb["choice"], lb["move"])
    weight = tf.stop_gradient(lesson_weight(tf.exp(logp - lb["logp"]), lb["weight"])) * lb["mask"]
    policy = -tf.exp(logp - tf.stop_gradient(logp)) * lb["adv"]  # its value is -adv; its gradient -adv × d logp
    error = value_weight * tf.square(value - lb["target"])
    return tf.reduce_sum(weight * (policy + error), axis=1) / rollout


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
        self.rules = starting_rules()  # its numbers (a world hands it its own rules)
        self._size = np.zeros((slots, 1 + MAX_LAYERS))  # each slot's brain genes: layers, neurons of each
        self._lessons: List[List[dict]] = [[] for _ in range(slots)]  # steps of others each slot saw: {step, next_obs}
        self._last: Dict[int, dict] = {}  # each slot's last recorded step, which a watcher may take as a lesson
        self._awaiting: Dict[int, List[dict]] = {}  # lessons of a slot's last step, waiting for what it senses next

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
        self._size[slot] = weights.size
        self._buffers[slot] = []
        self._pending.pop(slot, None)
        self._lessons[slot] = []
        self._last.pop(slot, None)
        self._awaiting.pop(slot, None)
        return slot

    def release(self, slot: int) -> None:
        """The slot of an animal that died is free for a newborn."""
        self._free.append(slot)
        self._free.sort()

    def state(self) -> Dict[str, np.ndarray]:
        """What a save keeps: every slot's weights, inborn weights, optimizer moments and step counts, learning speeds,
        which slots are free, and the dice. Unfinished experience (fewer than a rollout's steps) is not kept."""
        state = {"t": self._t.copy(), "lr": self._lr.copy(),
                 "free": np.array([slot in self._free for slot in range(self.slots)]),
                 "has_inborn": np.array([weights is not None for weights in self._inborn]),
                 "generator": self._gen.state.numpy(), "size": self._size.copy(),
                 "inborn_size": np.stack([np.zeros(1 + MAX_LAYERS) if w is None else w.size for w in self._inborn])}
        for name in SHAPES:
            state[f"params.{name}"] = self._params[name].numpy()
            state[f"m.{name}"] = self._m[name].numpy()
            state[f"v.{name}"] = self._v[name].numpy()
            state[f"inborn.{name}"] = np.stack([np.zeros(SHAPES[name]) if w is None else getattr(w, name)
                                                for w in self._inborn])
        return state

    def load_state(self, state: Dict[str, np.ndarray]) -> None:
        """Back to a saved state (see `state`); experience starts empty."""
        for name in SHAPES:
            self._params[name].assign(state[f"params.{name}"])
            self._m[name].assign(state[f"m.{name}"])
            self._v[name].assign(state[f"v.{name}"])
        self._t = state["t"].copy()
        self._lr = state["lr"].copy()
        self._free = [slot for slot in range(self.slots) if state["free"][slot]]
        self._size = state["size"].copy()
        self._inborn = [NetWeights(**{name: state[f"inborn.{name}"][slot] for name in SHAPES},
                                   size=state["inborn_size"][slot].copy())
                        if state["has_inborn"][slot] else None for slot in range(self.slots)]
        self._gen.reset(state["generator"])
        self._buffers = [[] for _ in range(self.slots)]
        self._pending = {}
        self._lessons = [[] for _ in range(self.slots)]
        self._last, self._awaiting = {}, {}

    def has_room(self) -> bool:
        return bool(self._free)

    def newcomer(self, np_rng: np.random.Generator, lr: float, layers: int, neurons: int, spread: float) -> int:
        """A newcomer of the primordial soup: a random network of `layers` layers of `neurons`, learning at `lr`."""
        return self.allocate(NetWeights.random(np_rng, layers, neurons, spread), lr)

    def baby(self, parent_slot: int, np_rng: np.random.Generator, lr_factor: float, inherit: str, genes_changed: List[str],
             rng, rules: Dict[str, float]) -> int:
        """A baby's network: the parent's brain as it is now ("learned") or as the parent was born ("inborn"), with the
        parent's learning speed × `lr_factor`; each of its brain genes among the baby's changed genes changes:
        "layers" a layer more or fewer, "neurons" a neuron more or fewer, "weights" a few weights a little. Otherwise
        the brain is an exact copy. Returns the baby's slot."""
        genes = self.current(parent_slot) if inherit == "learned" else self.inborn(parent_slot)
        limits = ((rules["gene.neurons.low"], rules["gene.neurons.high"]), (rules["gene.layers.low"], rules["gene.layers.high"]))
        if "layers" in genes_changed:
            genes = genes.grown(rng, 1.0, *limits)
        if "neurons" in genes_changed:
            genes = genes.grown(rng, 0.0, *limits)
        if "weights" in genes_changed:
            genes = genes.mutate(np_rng, rules["gene.weights.share"], rules["gene.weights.size"])
        return self.allocate(genes, self.lr(parent_slot) * lr_factor)

    def reshape(self, slot: int, layers: int, neurons: List[int]) -> None:
        """The owner's change to a living brain's size: `layers` layers of `neurons` each, its weights kept where they
        stay (a new neuron or layer says nothing at first); its inborn genes change the same way, so its babies inherit
        it."""
        def resized(weights: NetWeights) -> NetWeights:
            while int(weights.size[0]) < layers:
                weights = weights.with_layer(True)
            while int(weights.size[0]) > layers:
                weights = weights.with_layer(False)
            for layer, count in enumerate(neurons):
                while int(weights.size[1 + layer]) < count:
                    weights = weights.with_neuron(layer, True)
                while int(weights.size[1 + layer]) > count:
                    weights = weights.with_neuron(layer, False)
            return weights
        weights = resized(self.current(slot))
        for name in SHAPES:
            self._params[name][slot].assign(tf.cast(getattr(weights, name), tf.float32))
        self._size[slot] = weights.size
        self._inborn[slot] = resized(self.inborn(slot))

    def set_lr(self, slot: int, lr: float) -> None:
        """The owner's change to a living creature's learning speed."""
        self._lr[slot] = lr

    def active_neurons(self, slot: int) -> int:
        """How many neurons a slot's brain uses: the neurons of each layer it has."""
        layers = int(self._size[slot][0])
        return int(self._size[slot][1:1 + layers].sum())

    def brain_size(self, slot: int) -> List[int]:
        """Its neurons in each layer it has."""
        layers = int(self._size[slot][0])
        return [int(n) for n in self._size[slot][1:1 + layers]]

    def layers(self, slot: int) -> int:
        return int(self._size[slot][0])

    def lr(self, slot: int) -> float:
        return float(self._lr[slot])

    def inborn(self, slot: int) -> NetWeights:
        weights = self._inborn[slot]
        if weights is None:
            raise ValueError(f"slot {slot} has no network")
        return weights

    def current(self, slot: int) -> NetWeights:
        return NetWeights(**{name: self._params[name][slot].numpy() for name in SHAPES}, size=self._size[slot].copy())

    def act(self, slots: Sequence[int], senses: np.ndarray, attacks_on: bool,
            learning: bool = True) -> List[Tuple[int, Tuple[float, float]]]:
        """One call for all `slots`: each rolls its own dice for a choice and a move (each axis in -1..1)."""
        idx = tf.constant(list(slots), dtype=tf.int32)
        p = {name: tf.gather(var, idx) for name, var in self._params.items()}
        x = tf.constant(senses[:, None, :], dtype=tf.float32)
        attacks = tf.fill([len(slots), 1], 1.0 if attacks_on else 0.0)
        logits, mean, log_std, value = _heads(p, x, attacks, tf.constant(self._size[list(slots)], tf.float32))
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
        for lesson in self._awaiting.pop(slot, []):
            lesson["next_obs"] = step["obs"]
        self._last[slot] = step

    def watch(self, watcher: int, actor: int, reward: float, weight: float) -> None:
        """The actor's step just recorded is a lesson for the watcher, with the reward the watcher's genes give it and
        the weight of its learning from others; used in the watcher's next learning round (once the actor's next senses
        are known, unless the step ended its life)."""
        step = self._last[actor]
        lesson = {"step": step, "next_obs": None, "reward": reward, "weight": weight}
        if not step["done"]:
            self._awaiting.setdefault(actor, []).append(lesson)
        self._lessons[watcher].append(lesson)

    def _lesson_batch(self, slots: List[int]) -> Optional[Dict[str, tf.Tensor]]:
        """The ready lessons of each slot (its step ended a life, or its next senses are known), judged one step
        ahead by the watcher's own value, advantages normalized per slot; padded to one length. The rest wait."""
        ready = []
        for slot in slots:
            done = [l for l in self._lessons[slot] if l["step"]["done"] or l["next_obs"] is not None]
            self._lessons[slot] = [l for l in self._lessons[slot] if not (l["step"]["done"] or l["next_obs"] is not None)]
            ready.append(done)
        length = max(len(r) for r in ready)
        if length == 0:
            return None
        shape = (len(slots), length)
        obs, next_obs = np.zeros(shape + (N_IN,)), np.zeros(shape + (N_IN,))
        choice, move = np.zeros(shape, np.int32), np.zeros(shape + (2,))
        logp, attacks, reward, done, mask, weight = (np.zeros(shape) for _ in range(6))
        for k, lessons in enumerate(ready):
            for t, lesson in enumerate(lessons):
                step = lesson["step"]
                obs[k, t], choice[k, t], move[k, t] = step["obs"], step["choice"], step["move"]
                logp[k, t], attacks[k, t], done[k, t] = step["logp"], step["attacks"], step["done"]
                reward[k, t], weight[k, t] = lesson["reward"], lesson["weight"]
                if lesson["next_obs"] is not None:
                    next_obs[k, t] = lesson["next_obs"]
                mask[k, t] = 1.0
        p = {name: tf.gather(var, tf.constant(slots, dtype=tf.int32)) for name, var in self._params.items()}
        both = tf.constant(np.concatenate([obs, next_obs], axis=1), tf.float32)
        size = tf.constant(self._size[slots], tf.float32)
        value = _heads(p, both, tf.ones(both.shape[:2]), size)[3].numpy()
        target = reward + self.rules["learn_discount"] * value[:, length:] * (1.0 - done)
        adv = target - value[:, :length]
        for k in range(len(slots)):
            known = mask[k] > 0
            if known.any():
                adv[k, known] = (adv[k, known] - adv[k, known].mean()) / (adv[k, known].std() + 1e-8)
        adv *= mask
        arrays = {"obs": obs, "choice": choice, "move": move, "logp": logp, "attacks": attacks, "adv": adv,
                  "target": target, "mask": mask, "weight": weight}
        batch = {key: tf.constant(value, dtype=tf.int32 if key == "choice" else tf.float32) for key, value in arrays.items()}
        return {**batch, "size": size}

    def update(self) -> int:
        """One PPO update for every slot with a full rollout (64 steps plus the one that bootstraps them)."""
        rollout = round(self.rules["learn_rollout"])
        full = [slot for slot in range(self.slots) if len(self._buffers[slot]) > rollout]
        if not full:
            return 0
        batch = {key: [] for key in ("obs", "choice", "move", "logp", "attacks", "adv", "ret")}
        for slot in full:
            steps = self._buffers[slot][:rollout]
            column = {key: np.array([s[key] for s in steps]) for key in steps[0]}
            adv, ret = gae(column["reward"], column["value"], column["done"], self._buffers[slot][rollout]["value"],
                           self.rules["learn_discount"], self.rules["learn_credit"])
            batch["adv"].append((adv - adv.mean()) / (adv.std() + 1e-8))
            batch["ret"].append(ret)
            for key in ("obs", "choice", "move", "logp", "attacks"):
                batch[key].append(column[key])
            self._buffers[slot] = self._buffers[slot][rollout:]
        tensors = {key: tf.constant(np.stack(values), dtype=tf.int32 if key == "choice" else tf.float32)
                   for key, values in batch.items()}
        lessons = self._lesson_batch(full)
        size = tf.constant(self._size[full], tf.float32)
        for _ in range(round(self.rules["learn_passes"])):
            self._ppo_step(full, tensors, lessons, size)
        return len(full)

    def _ppo_step(self, slots: List[int], b: Dict[str, tf.Tensor], lessons: Optional[Dict[str, tf.Tensor]],
                  size: tf.Tensor) -> None:
        idx = tf.constant(slots, dtype=tf.int32)
        p = {name: tf.gather(var, idx) for name, var in self._params.items()}
        with tf.GradientTape() as tape:
            tape.watch(list(p.values()))
            logits, mean, log_std, value = _heads(p, b["obs"], b["attacks"], size)
            ratio = tf.exp(_log_prob(logits, mean, log_std, b["choice"], b["move"]) - b["logp"])
            clip = self.rules["learn_clip"]
            clipped = tf.clip_by_value(ratio, 1.0 - clip, 1.0 + clip)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * b["adv"], clipped * b["adv"]), axis=1)
            value_loss = tf.reduce_mean(tf.square(value - b["ret"]), axis=1)
            entropy = tf.reduce_mean(_entropy(logits, log_std), axis=1)
            guess, explore = self.rules["learn_guess"], self.rules["learn_explore"]
            loss = tf.reduce_sum(policy_loss + guess * value_loss - explore * entropy)
            if lessons is not None:
                loss += tf.reduce_sum(lesson_loss(p, lessons, guess, self.rules["learn_rollout"]))
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
