"""Interactive MuJoCo viewer: watch two policies fight in real time.

    python -m sumo_sim.viewer --p0 rule_based --p1 searcher --seed 0
"""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from .config import EnvConfig
from .env import SumoEnv
from .policies import POLICIES, make_policy


def run_interactive(env: SumoEnv, policies, seed: int = 0, episodes: int = 0, start_mode: str | None = None):
    """Run episodes in the passive viewer until it is closed (episodes=0 -> forever)."""
    options = {"start_mode": start_mode} if start_mode else None
    obs, _ = env.reset(seed=seed, options=options)
    for k, p in enumerate(policies):
        p.reset(seed + 1000 * (k + 1))
    ep = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.opt.geomgroup[:5] = 1
        v.cam.lookat[:] = (0, 0, 0)
        v.cam.distance, v.cam.elevation = 2.4, -60
        while v.is_running():
            t0 = time.perf_counter()
            action = np.stack([p.act(o) for p, o in zip(policies, obs)])
            obs, _, term, trunc, info = env.step(action)
            v.sync()
            if term or trunc:
                ep += 1
                print(f"episode {ep}: winner={info['winner']} t={info['time']:.2f}s")
                if episodes and ep >= episodes:
                    break
                time.sleep(0.5)
                obs, _ = env.reset(options=options)
                for p in policies:
                    p.reset()
            dt = env.cfg.sim.control_dt - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", default="rule_based", choices=sorted(POLICIES))
    ap.add_argument("--p1", default="searcher", choices=sorted(POLICIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-mode", choices=["standard", "random", "mixed"])
    ap.add_argument("--randomize", action="store_true", help="enable domain randomization")
    args = ap.parse_args()
    cfg = EnvConfig()
    cfg.randomization.enabled = args.randomize
    env = SumoEnv(cfg)
    run_interactive(env, [make_policy(args.p0), make_policy(args.p1)], args.seed, start_mode=args.start_mode)


if __name__ == "__main__":
    main()
