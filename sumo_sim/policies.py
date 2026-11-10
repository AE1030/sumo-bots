"""Scripted opponent policies.

Interface: `policy.reset(seed)` at the start of an episode, then
`policy.act(obs) -> np.ndarray([left, right])` every control step, where `obs`
is a single robot's observation vector (see sensors.py). Policies only see
the noisy observation, never ground truth, so any of them can drive either robot.
"""

from __future__ import annotations

import numpy as np

from .sensors import OBS_INDEX

TOF = slice(OBS_INDEX["tof_m60"], OBS_INDEX["tof_p60"] + 1)
LINE = slice(OBS_INDEX["line_fl"], OBS_INDEX["line_rr"] + 1)
TOF_ANGLES = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
CENTER_TOF = 2


class Policy:
    name = "base"

    def __init__(self, see_range: float = 1.5, seed: int | None = None):
        self.see_range = see_range
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs)

    # helpers
    def seen(self, obs) -> np.ndarray:
        return obs[TOF] < self.see_range

    def steer_toward(self, obs, forward: float = 1.0, gain: float = 0.8) -> np.ndarray:
        """Drive forward while turning toward the closest ToF return."""
        tof = obs[TOF]
        k = int(np.argmin(tof))
        turn = gain * TOF_ANGLES[k] / 60.0
        return np.clip([forward - turn, forward + turn], -1.0, 1.0)


class IdlePolicy(Policy):
    name = "idle"

    def act(self, obs):
        return np.zeros(2)


class RandomPolicy(Policy):
    """Random wheel commands held for a random number of control steps."""

    name = "random"

    def __init__(self, min_hold: int = 5, max_hold: int = 25, **kw):
        super().__init__(**kw)
        self.min_hold, self.max_hold = min_hold, max_hold
        self._left = 0
        self._action = np.zeros(2)

    def reset(self, seed=None):
        super().reset(seed)
        self._left = 0

    def act(self, obs):
        if self._left <= 0:
            self._action = self.rng.uniform(-1, 1, 2)
            self._left = int(self.rng.integers(self.min_hold, self.max_hold + 1))
        self._left -= 1
        return self._action.copy()


class ChargerPolicy(Policy):
    """Rotate in place until the center ToF sees the opponent, then full speed ahead."""

    name = "charger"

    def __init__(self, spin_speed: float = 0.5, memory: int = 5, **kw):
        super().__init__(**kw)
        self.spin_speed, self.memory = spin_speed, memory
        self._dir = 1.0
        self._since_seen = memory + 1

    def reset(self, seed=None):
        super().reset(seed)
        self._dir = self.rng.choice([-1.0, 1.0])
        self._since_seen = self.memory + 1

    def act(self, obs):
        if obs[TOF][CENTER_TOF] < self.see_range:
            self._since_seen = 0
        else:
            self._since_seen += 1
        if self._since_seen <= self.memory:
            return np.array([1.0, 1.0])
        s = self.spin_speed * self._dir
        return np.array([-s, s])


class SearcherPolicy(Policy):
    """Spin or sweep until any ToF sees something, then charge toward it.

    mode: "spin", "sweep" or "auto" (chosen at random each episode). While
    searching it backs away from the border if `avoid_edges` is set.
    """

    name = "searcher"

    def __init__(self, mode: str = "auto", spin_speed: float = 0.6, sweep_period: int = 40,
                 avoid_edges: bool = True, **kw):
        super().__init__(**kw)
        self.mode_cfg, self.spin_speed, self.sweep_period = mode, spin_speed, sweep_period
        self.avoid_edges = avoid_edges
        self._t = 0
        self._escape = 0
        self._dir = 1.0
        self.mode = "spin"

    def reset(self, seed=None):
        super().reset(seed)
        self._t = 0
        self._escape = 0
        self._dir = self.rng.choice([-1.0, 1.0])
        self.mode = self.rng.choice(["spin", "sweep"]) if self.mode_cfg == "auto" else self.mode_cfg

    def act(self, obs):
        self._t += 1
        if self.seen(obs).any():
            self._escape = 0
            return self.steer_toward(obs)
        line = obs[LINE]
        if self.avoid_edges and (self._escape > 0 or line.any()):
            if line.any():
                self._escape = 15
            self._escape -= 1
            return _escape_action(line, self._escape)
        if self.mode == "spin":
            s = self.spin_speed * self._dir
            return np.array([-s, s])
        # sweep: drive forward on an arc that flips curvature every half period
        phase = (self._t // (self.sweep_period // 2)) % 2
        turn = 0.4 * (1 if phase == 0 else -1) * self._dir
        return np.array([0.6 - turn, 0.6 + turn])


def _escape_action(line: np.ndarray, steps_left: int) -> np.ndarray:
    """Back up (or drive forward if the rear is on the line) while turning."""
    front = line[0] or line[1]
    rear = line[2] or line[3]
    if rear and not front:
        return np.array([1.0, 1.0])
    if steps_left > 7:
        return np.array([-1.0, -1.0])
    return np.array([-0.8, 0.8])


class RuleBasedPolicy(Policy):
    """Baseline: charge if the opponent is seen, back up and turn if a line sensor fires,
    otherwise rotate slowly to search."""

    name = "rule_based"

    def __init__(self, escape_steps: int = 20, search_speed: float = 0.5, **kw):
        super().__init__(**kw)
        self.escape_steps, self.search_speed = escape_steps, search_speed
        self._escape = 0
        self._front = True
        self._dir = 1.0

    def reset(self, seed=None):
        super().reset(seed)
        self._escape = 0
        self._dir = self.rng.choice([-1.0, 1.0])

    def act(self, obs):
        line = obs[LINE]
        if line.any() and self._escape <= 0:
            self._escape = self.escape_steps
            self._front = bool(line[0] or line[1]) or not bool(line[2] or line[3])
            # turn away from the side that fired
            left = line[0] + line[2]
            right = line[1] + line[3]
            self._dir = -1.0 if left > right else 1.0 if right > left else self._dir
        if self._escape > 0:
            self._escape -= 1
            back = -1.0 if self._front else 1.0
            if self._escape > self.escape_steps // 2:
                return np.array([back, back])
            s = 0.8 * self._dir
            return np.array([-s, s])
        if self.seen(obs).any():
            return self.steer_toward(obs)
        s = self.search_speed * self._dir
        return np.array([-s, s])


POLICIES = {
    p.name: p for p in (IdlePolicy, RandomPolicy, ChargerPolicy, SearcherPolicy, RuleBasedPolicy)
}


def make_policy(name: str, seed: int | None = None, **kwargs) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name](seed=seed, **kwargs)
