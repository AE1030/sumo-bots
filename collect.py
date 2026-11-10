"""Collect transition data for world-model training.

    python collect.py --episodes 200 --out data/sumo.npz
    python collect.py --episodes 50 --p0 rule_based --p1 random --randomize --seed 1

Each row of the output is one control step of one episode, for both robots
(axis 1 is the robot index). Keys:

    obs           (T, 2, D) float32  observation the policies acted on
    action        (T, 2, 2) float32  commanded [left, right] motor action
    next_obs      (T, 2, D) float32  observation after the step
    obs_pair      (T, 2, 2, D) float32  two extra independently noised views of
                                     the same state as `obs` (get_observation_pair)
    done          (T,) bool          terminated or truncated
    terminated    (T,) bool          a robot left the ring
    truncated     (T,) bool          timeout
    episode_id    (T,) int32
    step          (T,) int32         control step within the episode
    actor_mode    (T, 2) int8        which action source was used (see ACTOR_MODES)
    gt_<key>      ground truth from info at the *next* state (after the step), e.g.
                  gt_pos (T,2,2), gt_heading (T,2), gt_vel_world, gt_vel_body,
                  gt_yaw_rate, gt_dist_to_edge, gt_opp_rel_pos_body,
                  gt_opp_rel_pos_world, gt_opp_distance, gt_opp_bearing,
                  gt_contact (T,), gt_line_truth (T,2,4), gt_wheel_speed,
                  gt_applied_action, gt_time
    gt0_<key>     the same ground truth for the state of `obs` (before the step)
    ep_winner     (E,) int8   0, 1, -1 draw, -2 timeout
    ep_policies   (E, 2) str
    ep_params     (E,) str    JSON of per-episode domain-randomization parameters
    config        ()   str    JSON of the EnvConfig
    obs_fields    (D,) str    observation field names
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from sumo_sim import OBS_DIM, OBS_FIELDS, POLICIES, EnvConfig, SumoEnv, make_policy

GT_KEYS = (
    "pos", "heading", "vel_world", "vel_body", "yaw_rate", "dist_to_edge", "opp_rel_pos_body",
    "opp_rel_pos_world", "opp_distance", "opp_bearing", "contact", "line_truth", "wheel_speed",
    "applied_action", "time",
)  # fmt: skip

# Policy mixture used when --p0/--p1 is "mix". Engaging policies are weighted up
# so episodes contain contact and pushing.
POLICY_WEIGHTS = {"rule_based": 0.3, "charger": 0.25, "searcher": 0.25, "random": 0.15, "idle": 0.05}

# Action sources. "policy" is the robot's scripted policy; the others are
# ground-truth-guided perturbations used only to bias the data distribution
# toward contact, pushing and edge events (they are never model inputs).
ACTOR_MODES = ("policy", "seek_opponent", "seek_edge", "random_hold")


class EventBias:
    """Occasionally overrides a robot's policy for a short segment with a
    ground-truth-guided behaviour: charge the opponent, drive toward the edge,
    or hold a random command. Also adds small Gaussian action noise."""

    def __init__(self, rng, weights, seg_len=(10, 40), action_noise=0.1):
        self.rng = rng
        self.modes = np.array(ACTOR_MODES)
        w = np.array([weights[m] for m in ACTOR_MODES], float)
        self.p = w / w.sum()
        self.seg_len = seg_len
        self.action_noise = action_noise
        self.reset()

    def reset(self):
        self.mode = 0
        self.left = 0
        self.hold = np.zeros(2)

    def act(self, policy_action, gt, i):
        if self.left <= 0:
            self.mode = int(self.rng.choice(len(self.modes), p=self.p))
            self.left = int(self.rng.integers(*self.seg_len))
            self.hold = self.rng.uniform(-1, 1, 2)
        self.left -= 1
        mode = ACTOR_MODES[self.mode]
        if mode == "policy":
            a = policy_action
        elif mode == "seek_opponent":
            a = _steer(gt["opp_bearing"][i], forward=1.0)
        elif mode == "seek_edge":
            # Head radially outward from the ring center.
            pos, heading = gt["pos"][i], gt["heading"][i]
            bearing = np.arctan2(pos[1], pos[0]) - heading
            a = _steer(np.arctan2(np.sin(bearing), np.cos(bearing)), forward=0.7)
        else:
            a = self.hold
        a = np.asarray(a, float) + self.rng.normal(size=2) * self.action_noise
        return np.clip(a, -1, 1), self.mode


def _steer(bearing, forward=1.0, gain=1.5):
    turn = np.clip(gain * bearing, -1.0, 1.0)
    if abs(bearing) > np.pi / 2:  # target behind: turn in place
        return np.array([-turn, turn])
    return np.array([forward - turn, forward + turn])


def collect(args) -> dict:
    cfg = EnvConfig()
    cfg.randomization.enabled = args.randomize
    cfg.sim.start_mode = args.start_mode
    if args.timeout:
        cfg.sim.timeout = args.timeout
    env = SumoEnv(cfg)
    rng = np.random.default_rng(args.seed)
    names = list(POLICY_WEIGHTS)
    pw = np.array([POLICY_WEIGHTS[n] for n in names])
    pw = pw / pw.sum()
    bias_weights = dict(policy=1.0 - args.bias, seek_opponent=args.bias * 0.5,
                        seek_edge=args.bias * 0.25, random_hold=args.bias * 0.25)

    buf = defaultdict(list)
    ep_winner, ep_policies, ep_params = [], [], []
    for ep in range(args.episodes):
        chosen = [
            (spec if spec != "mix" else str(rng.choice(names, p=pw))) for spec in (args.p0, args.p1)
        ]
        policies = [make_policy(n, seed=int(rng.integers(2**31))) for n in chosen]
        for p in policies:
            p.reset()
        biases = [EventBias(rng, bias_weights, action_noise=args.action_noise) for _ in range(2)]
        options = {}
        if args.start_mode in ("random", "mixed") and rng.random() < args.edge_start:
            # Start some episodes closer to the border to get more edge events.
            options = {"start_mode": "random", "random_start_radius": 0.65}
        obs, info = env.reset(seed=int(rng.integers(2**31)), options=options)
        winner = -2
        for t in range(env.max_steps):
            pair = np.stack([np.stack(env.get_observation_pair(i)) for i in range(2)])
            action = np.zeros((2, 2))
            modes = np.zeros(2, np.int8)
            for i in range(2):
                action[i], modes[i] = biases[i].act(policies[i].act(obs[i]), info, i)
            next_obs, _, term, trunc, next_info = env.step(action)
            buf["obs"].append(obs)
            buf["action"].append(action.astype(np.float32))
            buf["next_obs"].append(next_obs)
            buf["obs_pair"].append(pair)
            buf["terminated"].append(term)
            buf["truncated"].append(trunc)
            buf["episode_id"].append(ep)
            buf["step"].append(t)
            buf["actor_mode"].append(modes)
            for k in GT_KEYS:
                buf["gt0_" + k].append(np.asarray(info[k]))
                buf["gt_" + k].append(np.asarray(next_info[k]))
            obs, info = next_obs, next_info
            if term or trunc:
                if term:
                    winner = next_info["winner"]
                break
        ep_winner.append(winner)
        ep_policies.append(chosen)
        ep_params.append(json.dumps(env.episode_params, default=float))
        if args.verbose:
            print(f"ep {ep:4d} {chosen[0]:>10s} vs {chosen[1]:<10s} steps={t + 1:4d} winner={winner}")

    out = {}
    for k, v in buf.items():
        arr = np.asarray(v)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        out[k] = arr
    out["episode_id"] = out["episode_id"].astype(np.int32)
    out["step"] = out["step"].astype(np.int32)
    out["done"] = out["terminated"] | out["truncated"]
    out["ep_winner"] = np.array(ep_winner, np.int8)
    out["ep_policies"] = np.array(ep_policies)
    out["ep_params"] = np.array(ep_params)
    out["config"] = np.array(json.dumps(cfg.to_dict()))
    out["obs_fields"] = np.array(OBS_FIELDS)
    out["actor_modes"] = np.array(ACTOR_MODES)
    assert out["obs"].shape[-1] == OBS_DIM
    return out


def summarize(d: dict) -> str:
    n = len(d["done"])
    w = d["ep_winner"]
    line = d["gt_line_truth"].reshape(n, -1).any(1)
    pushing = d["gt_contact"] & (np.abs(d["gt_vel_world"]).max(axis=(1, 2)) > 0.05)
    return (
        f"steps={n} episodes={len(w)}  push-outs={int((w >= 0).sum())} draws={int((w == -1).sum())} "
        f"timeouts={int((w == -2).sum())}\n"
        f"contact={d['gt_contact'].mean():.1%}  pushing(contact+moving)={pushing.mean():.1%}  "
        f"line-sensor-on-border={line.mean():.1%}  mean-ep-len={n / len(w) * 0.02:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choices = sorted(POLICIES) + ["mix"]
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--p0", default="mix", choices=choices, help="policy for robot 0")
    ap.add_argument("--p1", default="mix", choices=choices, help="policy for robot 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sumo.npz")
    ap.add_argument("--randomize", action="store_true", help="per-episode domain randomization")
    ap.add_argument("--start-mode", default="mixed", choices=["standard", "random", "mixed"])
    ap.add_argument("--edge-start", type=float, default=0.3,
                    help="probability of a random start near the border (random/mixed modes)")
    ap.add_argument("--bias", type=float, default=0.3,
                    help="fraction of control segments overridden by event-seeking behaviours (0 disables)")
    ap.add_argument("--action-noise", type=float, default=0.1, help="Gaussian std added to every action")
    ap.add_argument("--timeout", type=float, help="episode timeout in seconds (default from config)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    data = collect(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"saved {args.out}\n{summarize(data)}")


if __name__ == "__main__":
    main()
