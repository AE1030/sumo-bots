import dataclasses

import mujoco
import numpy as np
import pytest

from sumo_sim import (
    OBS_DIM,
    OBS_INDEX,
    EnvConfig,
    RobotConfig,
    RobotSpecError,
    SumoEnv,
    make_policy,
)
from sumo_sim.model import robot_footprint, robot_mass
from sumo_sim.sensors import _TOF_MASK

R = 0.77
FAR = (5.0, 5.0, 0.0)  # park a robot well away from everything (only valid before stepping)


@pytest.fixture
def env():
    e = SumoEnv()
    yield e
    e.close()


def geom_size(env, name):
    return env.model.geom_size[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)]


def test_ring_dimensions(env):
    assert env.cfg.arena.diameter == pytest.approx(1.54)
    assert env.cfg.arena.border_width == pytest.approx(0.0254)
    assert geom_size(env, "ring_border")[0] == pytest.approx(0.77)
    assert geom_size(env, "ring_black")[0] == pytest.approx(0.77 - 0.0254)


@pytest.mark.parametrize("randomize", [False, True])
def test_robot_footprint_and_mass(randomize):
    cfg = EnvConfig()
    cfg.randomization.enabled = randomize
    env = SumoEnv(cfg)
    for seed in range(3):
        env.reset(seed=seed)
        for h in env.handles:
            length, width = robot_footprint(env.model, env.data, h)
            assert length == pytest.approx(0.20, abs=1e-9)
            assert width == pytest.approx(0.20, abs=1e-9)
            assert robot_mass(env.model, h) == pytest.approx(3.0, abs=1e-9)


@pytest.mark.parametrize(
    "change",
    [dict(length=0.22), dict(width=0.21), dict(total_mass=3.2), dict(wheel_y_offset=0.1)],
)
def test_invalid_robot_raises(change):
    cfg = EnvConfig()
    cfg.robots[1] = dataclasses.replace(RobotConfig(), **change)
    with pytest.raises(RobotSpecError):
        SumoEnv(cfg)


@pytest.mark.parametrize("yaw", [0.0, np.pi / 4, -2.0])
def test_equal_commands_drive_straight(env, yaw):
    start = np.array([-0.5 * np.cos(yaw), -0.5 * np.sin(yaw)])
    # Opponent parked 0.5 m to the side of the path.
    side = start + 0.5 * np.array([-np.sin(yaw), np.cos(yaw)])
    env.reset(seed=0, options={"poses": [(*start, yaw), (*side, 0.0)]})
    for _ in range(50):  # 1 s
        _, _, term, _, info = env.step([[1.0, 1.0], [0.0, 0.0]])
        assert not term
    disp = info["pos"][0] - start
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    lateral = disp - fwd * (disp @ fwd)
    assert disp @ fwd > 0.7  # reaches ~1 m/s
    assert np.linalg.norm(lateral) < 1e-3
    assert abs(np.angle(np.exp(1j * (info["heading"][0] - yaw)))) < np.deg2rad(0.2)


def test_tof_distance_to_opponent(env):
    # Robot 0 at the origin facing +x; ToF origin is its front face at x=0.1.
    # Opponent center at x=0.6 -> its near face at x=0.5 -> expected 0.4 m.
    env.reset(seed=0, options={"poses": [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0)]})
    obs = env.get_clean_observation(0)
    assert obs[OBS_INDEX["tof_0"]] == pytest.approx(0.4, abs=1e-6)
    assert obs[OBS_INDEX["tof_p60"]] == pytest.approx(1.6)
    assert obs[OBS_INDEX["tof_m60"]] == pytest.approx(1.6)

    # Opponent face perpendicular to the +30 deg ray at a known 0.5 m.
    a, d = np.deg2rad(30), 0.5
    c = np.array([0.1, 0.0]) + (d + 0.1) * np.array([np.cos(a), np.sin(a)])
    env.reset(seed=0, options={"poses": [(0.0, 0.0, 0.0), (*c, a)]})
    assert env.get_clean_observation(0)[OBS_INDEX["tof_p30"]] == pytest.approx(d, abs=1e-6)


def test_tof_ignores_floor_and_reports_max_range(env):
    env.reset(seed=0, options={"poses": [(0.0, 0.0, 0.0), FAR]})
    obs = env.get_clean_observation(0)
    np.testing.assert_allclose(obs[:5], 1.6)
    # A ray pointing straight down never hits the floor or ring under the ToF mask.
    h = env.handles[0]
    dist = mujoco.mj_ray(env.model, env.data, np.array([0, 0, 0.04]), np.array([0, 0, -1.0]),
                         _TOF_MASK, 1, h.chassis, np.zeros(1, np.int32))
    assert dist == -1


def test_line_sensor_fires_over_border(env):
    # Put the front-left sensor (body offset +0.09, +0.09) at radius R - 1.27 cm (mid-band).
    x, y = R - 0.0127 - 0.09, -0.09
    env.reset(seed=0, options={"poses": [(x, y, 0.0), (-0.3, 0.0, 0.0)]})
    obs = env.get_clean_observation(0)
    assert obs[OBS_INDEX["line_fl"]] == 1
    assert obs[OBS_INDEX["line_rl"]] == 0
    assert obs[OBS_INDEX["line_rr"]] == 0
    # Fully on the black surface: nothing fires.
    np.testing.assert_array_equal(env.get_clean_observation(1)[5:9], 0)


def test_pushing_opponent_out_ends_episode(env):
    env.reset(seed=0, options={"start_mode": "standard"})
    for _ in range(env.max_steps):
        _, reward, term, trunc, info = env.step([[1.0, 1.0], [0.0, 0.0]])
        if term or trunc:
            break
    assert term and not trunc
    assert info["winner"] == 0
    assert reward == 1.0
    np.testing.assert_array_equal(info["rewards"], [1.0, -1.0])
    assert np.linalg.norm(info["pos"][1]) > R
    assert np.linalg.norm(info["pos"][0]) < R


def test_timeout_truncates():
    cfg = EnvConfig()
    cfg.sim.timeout = 0.2
    env = SumoEnv(cfg)
    env.reset(seed=0)
    for _ in range(10):
        _, _, term, trunc, _ = env.step(np.zeros((2, 2)))
    assert trunc and not term


def _rollout(seed, randomize):
    cfg = EnvConfig()
    cfg.randomization.enabled = randomize
    cfg.sim.start_mode = "random"
    env = SumoEnv(cfg)
    pols = [make_policy("random", seed=seed), make_policy("rule_based", seed=seed + 1)]
    obs, _ = env.reset(seed=seed)
    for p in pols:
        p.reset()
    out_obs, out_pos = [obs], []
    for _ in range(150):
        obs, _, term, trunc, info = env.step(np.stack([p.act(o) for p, o in zip(pols, obs)]))
        out_obs.append(obs)
        out_pos.append(info["pos"])
        if term or trunc:
            break
    return np.array(out_obs), np.array(out_pos), env.episode_params


@pytest.mark.parametrize("randomize", [False, True])
def test_reproducible_with_seed(randomize):
    a = _rollout(7, randomize)
    b = _rollout(7, randomize)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]
    c = _rollout(8, randomize)
    assert c[0].shape != a[0].shape or not np.array_equal(a[0], c[0])


def test_observation_pair_same_state_small_difference(env):
    env.reset(seed=0)
    for _ in range(20):
        env.step([[1.0, 0.5], [0.3, 1.0]])
    qpos = env.data.qpos.copy()
    clean = env.get_clean_observation(0)
    diffs = []
    for _ in range(200):
        a, b = env.get_observation_pair(0)
        assert a.shape == b.shape == (OBS_DIM,)
        diffs.append(np.abs(a - b))
    np.testing.assert_array_equal(env.data.qpos, qpos)  # physics untouched
    diffs = np.array(diffs)
    assert diffs.sum() > 0  # independently noised
    med = np.median(diffs, axis=0)
    assert np.all(med[:5] < 0.05)  # ToF: a few cm
    assert np.all(med[9:] < 0.5)  # IMU / encoders
    assert np.all(diffs.mean(0)[5:9] < 0.05)  # line flips are rare
    assert np.all(np.abs(a - clean)[12:] < 2.0)


def test_motor_response_and_latency():
    cfg = EnvConfig()
    cfg.sim.control_latency = 2
    env = SumoEnv(cfg)
    env.reset(seed=0)
    _, _, _, _, info = env.step([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(info["applied_action"], 0.0)
    env.step(np.ones((2, 2)))
    _, _, _, _, info = env.step(np.ones((2, 2)))
    np.testing.assert_array_equal(info["applied_action"], 1.0)
    # First-order lag: one step after the command arrives the wheels are far from top speed.
    assert np.all(info["wheel_speed"] < 0.5 * cfg.robots[0].motor.no_load_speed)


def test_policies_produce_valid_actions(env):
    obs, _ = env.reset(seed=0, options={"start_mode": "random"})
    for name in ("random", "charger", "searcher", "rule_based", "idle"):
        p = make_policy(name, seed=0)
        p.reset()
        for _ in range(30):
            a = p.act(obs[0])
            assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)
            obs, *_ = env.step(np.stack([a, a]))
