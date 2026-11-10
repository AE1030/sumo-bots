"""Configuration dataclasses for the sumo simulation.

All lengths are in metres, masses in kilograms, angles in degrees (where noted)
or radians, and times in seconds.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

# Hard limits from the competition rules. These are not part of the config on
# purpose: the config may describe a robot, the validator checks it against these.
MAX_FOOTPRINT_LENGTH = 0.20
MAX_FOOTPRINT_WIDTH = 0.20
MAX_ROBOT_MASS = 3.0
RING_DIAMETER = 1.54
BORDER_WIDTH = 0.0254


@dataclass
class ArenaConfig:
    diameter: float = RING_DIAMETER
    border_width: float = BORDER_WIDTH
    floor_friction: float = 0.9
    # Loss criterion: "center" (chassis center leaves the ring), "any_corner"
    # (any footprint corner leaves), or "all_corners" (whole footprint out).
    out_criterion: str = "center"

    @property
    def radius(self) -> float:
        return self.diameter / 2.0


@dataclass
class MotorConfig:
    # DC-motor model per wheel: tau = stall_torque * act - (stall_torque / no_load_speed) * omega,
    # clipped to +-torque_limit, where act is the first-order filtered command.
    stall_torque: float = 0.40  # N*m at the wheel (after gearbox)
    torque_limit: float = 0.40  # N*m current limit
    no_load_speed: float = 40.0  # rad/s  (~1.2 m/s with 3 cm wheels)
    time_constant: float = 0.05  # s, first-order command response
    armature: float = 1e-4  # reflected rotor inertia, kg*m^2
    damping: float = 1e-4  # joint viscous damping, N*m*s/rad


@dataclass
class RobotConfig:
    length: float = 0.20  # x extent of the chassis (forward)
    width: float = 0.20  # y extent
    height: float = 0.08  # top of chassis above ground
    ground_clearance: float = 0.005
    total_mass: float = 3.0
    wheel_radius: float = 0.03
    wheel_width: float = 0.02
    wheel_mass: float = 0.08
    wheel_y_offset: float = 0.08  # wheel center |y|; outer face must stay inside width/2
    wheel_friction: float = 1.0
    skid_radius: float = 0.008
    skid_x_offset: float = 0.08
    skid_mass: float = 0.02
    skid_friction: float = 0.05
    shell_mass: float = 1.4  # chassis shell; ballast = total - shell - wheels - skids
    chassis_friction: float = 0.5  # robot-robot contact friction
    # Ballast (center-of-mass) offset from the chassis center, (x, y, z) in metres.
    ballast_offset: tuple = (0.0, 0.0, 0.0)
    ballast_half_size: float = 0.02
    motor: MotorConfig = field(default_factory=MotorConfig)

    @property
    def ballast_mass(self) -> float:
        return self.total_mass - self.shell_mass - 2 * self.wheel_mass - 2 * self.skid_mass


@dataclass
class NoiseConfig:
    """Noise model for one sensor group."""

    std: float = 0.0  # additive Gaussian std (sensor units)
    rel_std: float = 0.0  # extra std proportional to the reading (ToF only)
    dropout: float = 0.0  # per-element probability the reading is replaced by its fill value
    latency: int = 0  # delay in control steps


@dataclass
class SensorConfig:
    tof_angles_deg: tuple = (-60.0, -30.0, 0.0, 30.0, 60.0)
    tof_max_range: float = 1.6
    tof_height: float = 0.04  # above ground
    # Line sensors at the corners, inset from the footprint edge.
    line_inset: float = 0.01
    tof: NoiseConfig = field(default_factory=lambda: NoiseConfig(std=0.005, rel_std=0.01, dropout=0.01))
    # Line sensors are modelled as an analog reflectance in [0, 1] plus noise,
    # thresholded at 0.5; std is in reflectance units.
    line: NoiseConfig = field(default_factory=lambda: NoiseConfig(std=0.1, dropout=0.005))
    gyro: NoiseConfig = field(default_factory=lambda: NoiseConfig(std=0.02, dropout=0.0))
    accel: NoiseConfig = field(default_factory=lambda: NoiseConfig(std=0.1, dropout=0.0))
    encoder: NoiseConfig = field(default_factory=lambda: NoiseConfig(std=0.2, dropout=0.0))


@dataclass
class RandomizationConfig:
    enabled: bool = False
    floor_friction: tuple = (0.6, 1.1)
    wheel_friction: tuple = (0.8, 1.2)  # multiplier on RobotConfig.wheel_friction
    motor_strength: tuple = (0.8, 1.2)  # multiplier on stall torque & torque limit
    sensor_noise_scale: tuple = (0.5, 2.0)  # multiplier on all noise stds
    control_latency: tuple = (0, 2)  # inclusive range, control steps
    # Ballast offset sampled uniformly within +-these bounds (x, y, z).
    ballast_offset: tuple = (0.05, 0.05, 0.01)
    per_robot: bool = True  # sample robot-level parameters independently per robot


@dataclass
class SimConfig:
    physics_dt: float = 0.002  # 500 Hz
    control_dt: float = 0.02  # 50 Hz
    timeout: float = 30.0
    control_latency: int = 0  # action delay in control steps
    # "standard" (facing each other), "random", or "mixed" (choose per episode).
    start_mode: str = "standard"
    standard_separation: float = 0.6  # distance between chassis centers
    start_jitter: float = 0.0  # position/heading jitter for the standard start (m, rad*10)
    random_start_radius: float = 0.55  # max radial position of a random start
    min_start_separation: float = 0.35


@dataclass
class EnvConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    robots: list = field(default_factory=lambda: [RobotConfig(), RobotConfig()])
    sensors: SensorConfig = field(default_factory=SensorConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        return _from_dict(cls, d)

    def copy(self) -> "EnvConfig":
        return copy.deepcopy(self)


def _from_dict(cls, d: Any):
    if not is_dataclass(cls):
        return d
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if f.name == "robots":
            kwargs[f.name] = [_from_dict(RobotConfig, r) for r in v]
        elif isinstance(f.default_factory, type) and is_dataclass(f.default_factory):  # type: ignore[arg-type]
            kwargs[f.name] = _from_dict(f.default_factory, v)
        elif f.name in ("tof", "line", "gyro", "accel", "encoder"):
            kwargs[f.name] = _from_dict(NoiseConfig, v)
        elif isinstance(v, list):
            kwargs[f.name] = tuple(v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)
