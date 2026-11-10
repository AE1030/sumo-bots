"""Render a top-down MP4 of one episode, or open the interactive viewer.

    python render.py --p0 rule_based --p1 charger --seed 3 --out episode.mp4
    python render.py --interactive --p0 rule_based --p1 searcher
"""

from __future__ import annotations

import argparse
import os
import sys

# Offscreen rendering needs a headless GL backend, chosen before mujoco is imported.
if "--interactive" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from sumo_sim import POLICIES, EnvConfig, SumoEnv, make_policy  # noqa: E402


def render_episode(env: SumoEnv, policies, path: str, seed: int, camera: str = "top",
                   start_mode: str | None = None, max_seconds: float | None = None) -> dict:
    import imageio.v2 as imageio

    options = {"start_mode": start_mode} if start_mode else None
    obs, info = env.reset(seed=seed, options=options)
    for k, p in enumerate(policies):
        p.reset(seed + 1000 * (k + 1))
    fps = int(round(1.0 / env.cfg.sim.control_dt))
    max_steps = int(max_seconds / env.cfg.sim.control_dt) if max_seconds else None
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8) as w:
        w.append_data(env.render(camera))
        t = 0
        while True:
            action = np.stack([p.act(o) for p, o in zip(policies, obs)])
            obs, _, term, trunc, info = env.step(action)
            w.append_data(env.render(camera))
            t += 1
            if term or trunc or (max_steps and t >= max_steps):
                break
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", default="rule_based", choices=sorted(POLICIES))
    ap.add_argument("--p1", default="charger", choices=sorted(POLICIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="episode.mp4")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--camera", default="top", choices=["top", "angled"])
    ap.add_argument("--start-mode", choices=["standard", "random", "mixed"])
    ap.add_argument("--max-seconds", type=float, help="stop the video early")
    ap.add_argument("--randomize", action="store_true", help="enable domain randomization")
    ap.add_argument("--interactive", action="store_true", help="open the MuJoCo viewer instead")
    args = ap.parse_args()

    cfg = EnvConfig()
    cfg.randomization.enabled = args.randomize
    policies = [make_policy(args.p0), make_policy(args.p1)]
    if args.interactive:
        from sumo_sim.viewer import run_interactive

        run_interactive(SumoEnv(cfg), policies, args.seed, start_mode=args.start_mode)
        return

    env = SumoEnv(cfg, render_mode="rgb_array", render_size=args.size)
    info = render_episode(env, policies, args.out, args.seed, args.camera, args.start_mode, args.max_seconds)
    env.close()
    print(f"wrote {args.out}: winner={info['winner']} duration={info['time']:.2f}s")


if __name__ == "__main__":
    main()
