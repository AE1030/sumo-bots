"""Two-robot sumo environment (Gymnasium-style)."""

from __future__ import annotations

import copy
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np

from .config import EnvConfig
from .model import build_model, validate_robots
from .sensors import OBS_DIM, ImuAccumulator, SensorSuite, clean_readings, line_sensor_offsets, over_border

DRAW = -1


def visual_options() -> mujoco.MjvOption:
    """Scene options that show the robot chassis (group 3) and markers (group 4)."""
    opt = mujoco.MjvOption()
    opt.geomgroup[:5] = 1
    return opt


class SumoEnv(gym.Env):
    """Two differential-drive robots on a sumo ring.

    action: float array (2, 2) -> [robot][left_motor, right_motor], each in [-1, 1].
    observation: float32 array (2, OBS_DIM), one row per robot (layout in sensors.py).
    reward: +1 / -1 / 0 from robot 0's point of view; per-robot rewards are in
        info["rewards"].
    info: ground truth for readout heads (see `_ground_truth`); never feed it to a model.

    reset(options=...) accepts:
        "start_mode": "standard" | "random" (overrides config for this episode)
        "poses": [(x, y, yaw), (x, y, yaw)] explicit start poses
        "random_start_radius": float, override for the random start
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None,
                 render_size: int = 480):
        self.cfg = config.copy() if config is not None else EnvConfig()
        sim = self.cfg.sim
        self.n_substeps = int(round(sim.control_dt / sim.physics_dt))
        if self.n_substeps < 1 or abs(self.n_substeps * sim.physics_dt - sim.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        self.max_steps = int(round(sim.timeout / sim.control_dt))

        self.model, self.data, self.handles = build_model(self.cfg)
        self._defaults = dict(
            geom_friction=self.model.geom_friction.copy(),
            actuator_gainprm=self.model.actuator_gainprm.copy(),
            actuator_biasprm=self.model.actuator_biasprm.copy(),
            actuator_forcerange=self.model.actuator_forcerange.copy(),
            body_pos=self.model.body_pos.copy(),
        )
        self._floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._robot_geoms = [
            {g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) in h.subtree_bodies}
            for h in self.handles
        ]

        self.sensors = [SensorSuite(self.cfg.sensors) for _ in self.handles]
        self._imu = [ImuAccumulator() for _ in self.handles]
        self.control_latency = [sim.control_latency] * 2
        self._action_queues: list = [deque() for _ in self.handles]
        self.applied_action = np.zeros((2, 2))
        self.episode_params: dict = {}

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)
        hi = np.full((2, OBS_DIM), np.inf, np.float32)
        self.observation_space = gym.spaces.Box(-hi, hi, dtype=np.float32)

        self.render_mode = render_mode
        self.render_size = render_size
        self._renderer = None
        self._noise_rng = np.random.default_rng(0)
        self.step_count = 0

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        seeds = self.np_random.integers(0, 2**63 - 1, size=3)
        init_rng = np.random.default_rng(seeds[0])
        dr_rng = np.random.default_rng(seeds[1])
        self._noise_rng = np.random.default_rng(seeds[2])

        noise_scales = self._apply_randomization(dr_rng)

        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        poses = options.get("poses") or self._sample_poses(init_rng, options)
        for h, (x, y, yaw) in zip(self.handles, poses):
            a = h.qpos_adr
            d.qpos[a : a + 3] = (x, y, self.cfg.robots[0].wheel_radius)
            d.qpos[a + 3 : a + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(m, d)
        validate_robots(m, d, self.handles)

        self.step_count = 0
        self.applied_action[:] = 0.0
        self._action_queues = [deque([np.zeros(2)] * lat) for lat in self.control_latency]
        for i, (h, imu, s) in enumerate(zip(self.handles, self._imu, self.sensors)):
            imu.reset()
            s.reset(self._clean(i), noise_scales[i])
        self._last_obs = self._observe()
        return self._last_obs, self._info(terminated=False, winner=None)

    def _apply_randomization(self, rng: np.random.Generator) -> list:
        m, rc_cfg = self.model, self.cfg.randomization
        for k, v in self._defaults.items():
            getattr(m, k)[:] = v
        self.control_latency = [self.cfg.sim.control_latency] * 2
        noise_scales = [1.0, 1.0]
        floor = self.cfg.arena.floor_friction
        params = dict(floor_friction=floor, wheel_friction=[1.0, 1.0], motor_strength=[1.0, 1.0],
                      noise_scale=noise_scales, control_latency=self.control_latency,
                      ballast_offset=[list(rc.ballast_offset) for rc in self.cfg.robots])
        if rc_cfg.enabled:
            floor = rng.uniform(*rc_cfg.floor_friction)
            params["floor_friction"] = floor
            m.geom_friction[self._floor_geom, 0] = floor
            shared = None
            for i, (h, rc) in enumerate(zip(self.handles, self.cfg.robots)):
                if rc_cfg.per_robot or shared is None:
                    shared = dict(
                        wheel=rng.uniform(*rc_cfg.wheel_friction),
                        motor=rng.uniform(*rc_cfg.motor_strength),
                        noise=rng.uniform(*rc_cfg.sensor_noise_scale),
                        latency=int(rng.integers(rc_cfg.control_latency[0], rc_cfg.control_latency[1] + 1)),
                        ballast=rng.uniform(-1, 1, 3) * np.asarray(rc_cfg.ballast_offset),
                    )
                s = shared
                for g in h.wheel_geoms:
                    m.geom_friction[g, 0] = rc.wheel_friction * s["wheel"] * floor
                for a in h.actuators:
                    m.actuator_gainprm[a, 0] *= s["motor"]
                    m.actuator_biasprm[a, 2] *= s["motor"]
                    m.actuator_forcerange[a] *= s["motor"]
                offset = np.asarray(rc.ballast_offset) + s["ballast"]
                m.body_pos[h.ballast] = self._defaults["body_pos"][h.ballast] + s["ballast"]
                noise_scales[i] = s["noise"]
                self.control_latency[i] = s["latency"]
                params["wheel_friction"][i] = s["wheel"]
                params["motor_strength"][i] = s["motor"]
                params["ballast_offset"][i] = offset.tolist()
        self.episode_params = params
        return noise_scales

    def _sample_poses(self, rng: np.random.Generator, options: dict) -> list:
        sim = self.cfg.sim
        mode = options.get("start_mode", sim.start_mode)
        if mode == "mixed":
            mode = "standard" if rng.random() < 0.5 else "random"
        if mode == "standard":
            s = sim.standard_separation / 2
            j = sim.start_jitter
            p = [(-s, 0.0, 0.0), (s, 0.0, np.pi)]
            if j > 0:
                p = [(x + rng.normal() * j, y + rng.normal() * j, yaw + rng.normal() * j * 10)
                     for x, y, yaw in p]
            return p
        if mode == "random":
            rmax = options.get("random_start_radius", sim.random_start_radius)
            for _ in range(1000):
                r = rmax * np.sqrt(rng.random(2))
                th = rng.uniform(0, 2 * np.pi, 2)
                xy = np.stack([r * np.cos(th), r * np.sin(th)], 1)
                if np.linalg.norm(xy[0] - xy[1]) >= sim.min_start_separation:
                    yaw = rng.uniform(-np.pi, np.pi, 2)
                    return [(xy[i, 0], xy[i, 1], yaw[i]) for i in range(2)]
            raise RuntimeError("could not sample non-overlapping start poses")
        raise ValueError(f"unknown start_mode {mode!r}")

    # ------------------------------------------------------------------- step
    def step(self, action):
        action = np.clip(np.asarray(action, np.float64).reshape(2, 2), -1.0, 1.0)
        m, d = self.model, self.data
        for i, h in enumerate(self.handles):
            q = self._action_queues[i]
            q.append(action[i].copy())
            self.applied_action[i] = q.popleft()
            d.ctrl[list(h.actuators)] = self.applied_action[i]

        for imu in self._imu:
            imu.reset()
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
            for imu, h in zip(self._imu, self.handles):
                imu.add(d, h)
        self.step_count += 1

        for i, s in enumerate(self.sensors):
            s.push(self._clean(i))
        self._last_obs = self._observe()

        out = [self._is_out(i) for i in range(2)]
        terminated = any(out)
        winner = None
        if terminated:
            winner = DRAW if all(out) else (1 if out[0] else 0)
        truncated = (not terminated) and self.step_count >= self.max_steps
        rewards = np.zeros(2)
        if winner is not None and winner != DRAW:
            rewards[winner], rewards[1 - winner] = 1.0, -1.0
        info = self._info(terminated, winner)
        info["rewards"] = rewards
        return self._last_obs, float(rewards[0]), terminated, truncated, info

    # ------------------------------------------------------------ observation
    def _clean(self, i: int) -> np.ndarray:
        a = self.cfg.arena
        return clean_readings(self.model, self.data, self.handles[i], self.cfg.robots[i],
                              self.cfg.sensors, a.radius, a.border_width, self._imu[i])

    def _observe(self) -> np.ndarray:
        return np.stack([s.sample(self._noise_rng) for s in self.sensors])

    def get_observation_pair(self, robot_id: int) -> tuple:
        """Two independently noised observations of the exact same physical state.

        Noise and dropout are resampled for each view; the physics state, the
        latency-delayed clean readings and the episode's noise scale are shared.
        """
        s = self.sensors[robot_id]
        clean = s.delayed()
        return s.apply_noise(clean, self._noise_rng), s.apply_noise(clean, self._noise_rng)

    def get_clean_observation(self, robot_id: int) -> np.ndarray:
        """Noise-free (but latency-delayed) observation. For debugging and tests only."""
        obs = self.sensors[robot_id].delayed()
        obs[5:9] = obs[5:9] > 0.5
        return obs.astype(np.float32)

    # ----------------------------------------------------------- ground truth
    def _is_out(self, i: int) -> bool:
        a = self.cfg.arena
        h, rc = self.handles[i], self.cfg.robots[i]
        pos = self.data.xpos[h.chassis, :2]
        crit = a.out_criterion
        if crit == "center":
            return bool(np.linalg.norm(pos) > a.radius)
        R = self.data.xmat[h.chassis].reshape(3, 3)[:2, :2]
        hx, hy = rc.length / 2, rc.width / 2
        corners = pos + np.array([[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]]) @ R.T
        outside = np.linalg.norm(corners, axis=1) > a.radius
        if crit == "any_corner":
            return bool(outside.any())
        if crit == "all_corners":
            return bool(outside.all())
        raise ValueError(f"unknown out_criterion {crit!r}")

    def robots_in_contact(self) -> bool:
        d = self.data
        g0, g1 = self._robot_geoms
        for k in range(d.ncon):
            a, b = int(d.contact[k].geom1), int(d.contact[k].geom2)
            if (a in g0 and b in g1) or (a in g1 and b in g0):
                return True
        return False

    def _ground_truth(self) -> dict:
        d, a = self.data, self.cfg.arena
        pos = np.zeros((2, 2))
        heading = np.zeros(2)
        vel_w = np.zeros((2, 2))
        vel_b = np.zeros((2, 2))
        yaw_rate = np.zeros(2)
        line = np.zeros((2, 4))
        rots = []
        for i, h in enumerate(self.handles):
            R = d.xmat[h.chassis].reshape(3, 3)
            rots.append(R[:2, :2])
            pos[i] = d.xpos[h.chassis, :2]
            heading[i] = np.arctan2(R[1, 0], R[0, 0])
            vel_w[i] = d.qvel[h.qvel_adr : h.qvel_adr + 2]
            vel_b[i] = R[:2, :2].T @ vel_w[i]
            yaw_rate[i] = (R @ d.qvel[h.qvel_adr + 3 : h.qvel_adr + 6])[2]
            corners = pos[i] + line_sensor_offsets(self.cfg.robots[i], self.cfg.sensors) @ R[:2, :2].T
            line[i] = over_border(corners, a.radius, a.border_width)
        rel_w = np.stack([pos[1] - pos[0], pos[0] - pos[1]])
        rel_b = np.stack([rots[i].T @ rel_w[i] for i in range(2)])
        return dict(
            time=self.step_count * self.cfg.sim.control_dt,
            pos=pos,
            heading=heading,
            vel_world=vel_w,
            vel_body=vel_b,
            yaw_rate=yaw_rate,
            dist_to_edge=a.radius - np.linalg.norm(pos, axis=1),
            opp_rel_pos_world=rel_w,
            opp_rel_pos_body=rel_b,
            opp_distance=np.linalg.norm(rel_w, axis=1),
            opp_bearing=np.arctan2(rel_b[:, 1], rel_b[:, 0]),
            contact=self.robots_in_contact(),
            line_truth=line,
            wheel_speed=np.array([[d.qvel[j] for j in h.wheel_qvel] for h in self.handles]),
            applied_action=self.applied_action.copy(),
        )

    def _info(self, terminated: bool, winner) -> dict:
        info = self._ground_truth()
        info["winner"] = winner  # None while running, 0/1, or DRAW (-1)
        info["episode_params"] = copy.deepcopy(self.episode_params)
        return info

    # ----------------------------------------------------------------- render
    def render(self, camera: str = "top"):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.render_size, self.render_size)
            self._scene_option = visual_options()
        self._renderer.update_scene(self.data, camera=camera, scene_option=self._scene_option)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class SingleAgentWrapper(gym.Env):
    """Controls one robot; the other is driven by a fixed policy (obs -> action)."""

    def __init__(self, env: SumoEnv, opponent, agent_id: int = 0):
        self.env, self.opponent, self.agent_id = env, opponent, agent_id
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        hi = np.full(OBS_DIM, np.inf, np.float32)
        self.observation_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
        self._obs = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._obs, info = self.env.reset(seed=seed, options=options)
        if hasattr(self.opponent, "reset"):
            self.opponent.reset(self.env.np_random.integers(2**31))
        return self._obs[self.agent_id], info

    def step(self, action):
        a = np.zeros((2, 2))
        a[self.agent_id] = action
        a[1 - self.agent_id] = self.opponent.act(self._obs[1 - self.agent_id])
        self._obs, _, term, trunc, info = self.env.step(a)
        return self._obs[self.agent_id], float(info["rewards"][self.agent_id]), term, trunc, info
