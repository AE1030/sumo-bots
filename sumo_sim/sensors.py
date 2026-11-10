"""Sensor models: clean readings from the physics state, plus noise/dropout/latency.

Observation vector layout (per robot, float32, length OBS_DIM = 14):

    idx  name          units   notes
    0    tof_m60       m       ToF ray at -60 deg (right of front); max range if nothing hit
    1    tof_m30       m       -30 deg
    2    tof_0         m       straight ahead
    3    tof_p30       m       +30 deg (left)
    4    tof_p60       m       +60 deg
    5    line_fl       {0,1}   front-left corner line sensor, 1 = over white border
    6    line_fr       {0,1}   front-right
    7    line_rl       {0,1}   rear-left
    8    line_rr       {0,1}   rear-right
    9    gyro_yaw      rad/s   yaw rate (body z, CCW positive)
    10   accel_fwd     m/s^2   forward (body x) acceleration, averaged over the control step
    11   accel_lat     m/s^2   lateral (body y, left positive) acceleration
    12   enc_left      rad/s   left wheel angular velocity (positive = drives forward)
    13   enc_right     rad/s   right wheel angular velocity

Dropout fill values: ToF -> max range, everything else -> 0.
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from .config import NoiseConfig, RobotConfig, SensorConfig
from .model import TOF_GEOM_GROUP, RobotHandles

OBS_FIELDS = (
    "tof_m60", "tof_m30", "tof_0", "tof_p30", "tof_p60",
    "line_fl", "line_fr", "line_rl", "line_rr",
    "gyro_yaw", "accel_fwd", "accel_lat",
    "enc_left", "enc_right",
)  # fmt: skip
OBS_DIM = len(OBS_FIELDS)
OBS_INDEX = {name: i for i, name in enumerate(OBS_FIELDS)}

# Sensor groups in observation order: (name, slice).
GROUPS = (
    ("tof", slice(0, 5)),
    ("line", slice(5, 9)),
    ("gyro", slice(9, 10)),
    ("accel", slice(10, 12)),
    ("encoder", slice(12, 14)),
)

_TOF_INSET = 0.02
_TOF_MASK = np.zeros(6, np.uint8)
_TOF_MASK[TOF_GEOM_GROUP] = 1


def line_sensor_offsets(rc: RobotConfig, sc: SensorConfig) -> np.ndarray:
    """Body-frame xy of the line sensors in order FL, FR, RL, RR."""
    ax = rc.length / 2 - sc.line_inset
    ay = rc.width / 2 - sc.line_inset
    return np.array([[ax, ay], [ax, -ay], [-ax, ay], [-ax, -ay]])


def over_border(xy: np.ndarray, radius: float, border_width: float) -> np.ndarray:
    r = np.linalg.norm(xy, axis=-1)
    return ((r >= radius - border_width) & (r <= radius)).astype(np.float64)


class ImuAccumulator:
    """Averages gyro and accelerometer readings across the physics substeps."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.gyro = 0.0
        self.accel = np.zeros(2)
        self.n = 0

    def add(self, data: mujoco.MjData, h: RobotHandles):
        self.gyro += data.sensordata[h.gyro_adr + 2]
        self.accel += data.sensordata[h.accel_adr : h.accel_adr + 2]
        self.n += 1

    def mean(self) -> tuple:
        n = max(self.n, 1)
        return self.gyro / n, self.accel / n


def clean_readings(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    h: RobotHandles,
    rc: RobotConfig,
    sc: SensorConfig,
    radius: float,
    border_width: float,
    imu: ImuAccumulator,
) -> np.ndarray:
    """Noise-free observation vector (line sensors as analog reflectance 0/1)."""
    out = np.empty(OBS_DIM)
    R = data.xmat[h.chassis].reshape(3, 3)
    pos = data.xpos[h.chassis]

    # Time-of-flight rays from the front-center of the chassis, horizontal in the body frame.
    # Distances are measured from the sensor point on the front face, but each ray is
    # cast from _TOF_INSET behind it along its own direction (own body is excluded) so
    # that a slightly penetrating opponent registers ~0 instead of its far side.
    sensor = pos + R @ np.array([rc.length / 2, 0.0, sc.tof_height - rc.wheel_radius])
    geomid = np.zeros(1, np.int32)
    for k, a in enumerate(np.deg2rad(sc.tof_angles_deg)):
        vec = R @ np.array([np.cos(a), np.sin(a), 0.0])
        d = mujoco.mj_ray(model, data, sensor - _TOF_INSET * vec, vec, _TOF_MASK, 0, h.chassis, geomid)
        d -= _TOF_INSET
        out[k] = sc.tof_max_range if (d < -_TOF_INSET or d > sc.tof_max_range) else max(d, 0.0)

    corners = pos[:2] + line_sensor_offsets(rc, sc) @ R[:2, :2].T
    out[5:9] = over_border(corners, radius, border_width)

    gyro, accel = imu.mean() if imu.n else (
        data.sensordata[h.gyro_adr + 2],
        data.sensordata[h.accel_adr : h.accel_adr + 2],
    )
    out[9] = gyro
    out[10:12] = accel
    out[12] = data.qvel[h.wheel_qvel[0]]
    out[13] = data.qvel[h.wheel_qvel[1]]
    return out


class SensorSuite:
    """Per-robot sensor history (for latency) and noise model."""

    def __init__(self, sc: SensorConfig):
        self.sc = sc
        self.noise_scale = 1.0
        self.max_latency = max(getattr(sc, g).latency for g, _ in GROUPS)
        self.history: deque = deque(maxlen=self.max_latency + 1)

    def reset(self, first: np.ndarray, noise_scale: float = 1.0):
        self.noise_scale = noise_scale
        self.history.clear()
        for _ in range(self.max_latency + 1):
            self.history.append(first.copy())

    def push(self, clean: np.ndarray):
        self.history.append(clean.copy())

    def delayed(self) -> np.ndarray:
        """Clean readings with each group's latency applied."""
        out = np.empty(OBS_DIM)
        for g, sl in GROUPS:
            lat = getattr(self.sc, g).latency
            out[sl] = self.history[-1 - lat][sl]
        return out

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One noisy observation of the current (delayed) clean state."""
        return self.apply_noise(self.delayed(), rng)

    def apply_noise(self, clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sc, s = self.sc, self.noise_scale
        obs = clean.copy()
        for g, sl in GROUPS:
            nc: NoiseConfig = getattr(sc, g)
            x = obs[sl]
            std = s * (nc.std + nc.rel_std * np.abs(x))
            x = x + rng.normal(size=x.shape) * std
            if g == "tof":
                # A ray with no return keeps reporting exactly max range.
                x = np.where(obs[sl] >= sc.tof_max_range, sc.tof_max_range, np.clip(x, 0.0, sc.tof_max_range))
                fill = sc.tof_max_range
            elif g == "line":
                x = (x > 0.5).astype(np.float64)
                fill = 0.0
            else:
                fill = 0.0
            drop = rng.random(x.shape) < nc.dropout
            x[drop] = fill
            obs[sl] = x
        return obs.astype(np.float32)
