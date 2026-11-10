"""Builds the MuJoCo model from config and validates robot constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

import mujoco
import numpy as np

from .config import (
    MAX_FOOTPRINT_LENGTH,
    MAX_FOOTPRINT_WIDTH,
    MAX_ROBOT_MASS,
    EnvConfig,
    RobotConfig,
)

ASSETS = Path(__file__).parent / "assets"
ROBOT_COLORS = ("0.85 0.15 0.15 1", "0.15 0.35 0.9 1")
TOF_GEOM_GROUP = 3

# Tolerances for the startup validation.
_LEN_TOL = 1e-9
_MASS_TOL = 1e-6


class RobotSpecError(ValueError):
    """Raised when a robot violates the footprint or mass constraints."""


def prefix(i: int) -> str:
    return f"r{i}_"


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def _robot_xml(i: int, rc: RobotConfig, floor_friction: float) -> str:
    wr = rc.wheel_radius
    hz = (rc.height - rc.ground_clearance) / 2.0
    cz = (rc.height + rc.ground_clearance) / 2.0 - wr
    bs = rc.ballast_half_size
    bm = rc.ballast_mass
    bx, by, bz = rc.ballast_offset
    subs = dict(
        p=prefix(i),
        x=_fmt(-0.3 if i == 0 else 0.3),
        y="0",
        z=_fmt(wr),
        yaw=_fmt(0.0 if i == 0 else np.pi),
        hx=_fmt(rc.length / 2),
        hy=_fmt(rc.width / 2),
        hz=_fmt(hz),
        cz=_fmt(cz),
        shell_mass=_fmt(rc.shell_mass),
        rgba=ROBOT_COLORS[i % 2],
        chassis_friction=_fmt(rc.chassis_friction),
        skid_r=_fmt(rc.skid_radius),
        skid_x=_fmt(rc.skid_x_offset),
        skid_z=_fmt(rc.skid_radius - wr),
        skid_mass=_fmt(rc.skid_mass),
        skid_friction=_fmt(rc.skid_friction),
        marker_x=_fmt(rc.length / 2 - 0.02),
        marker_z=_fmt(rc.height - wr + 0.002),
        bx=_fmt(bx),
        by=_fmt(by),
        bz=_fmt(cz + bz),
        bs=_fmt(bs),
        ballast_mass=_fmt(bm),
        bI=_fmt(max(bm, 1e-6) * (2 * bs) ** 2 / 6.0),
        wy=_fmt(rc.wheel_y_offset),
        wr=_fmt(wr),
        whw=_fmt(rc.wheel_width / 2),
        wheel_mass=_fmt(rc.wheel_mass),
        # Wheels have contact priority over the floor, so their friction value
        # is the effective wheel/floor coefficient.
        wheel_friction=_fmt(rc.wheel_friction * floor_friction),
        armature=_fmt(rc.motor.armature),
        damping=_fmt(rc.motor.damping),
    )
    tmpl = Template((ASSETS / "robot.xml").read_text())
    body = tmpl.substitute(subs)
    # Drop the template comment and indent.
    body = body.split("-->", 1)[1].strip()
    return "\n".join("    " + line for line in body.splitlines())


def _actuator_xml(i: int, rc: RobotConfig) -> str:
    p = prefix(i)
    m = rc.motor
    lines = []
    for side in ("left", "right"):
        lines.append(
            f'    <general name="{p}motor_{side}" joint="{p}wheel_{side}" dyntype="filterexact" '
            f'dynprm="{_fmt(m.time_constant)}" gaintype="fixed" gainprm="{_fmt(m.stall_torque)}" '
            f'biastype="affine" biasprm="0 0 {_fmt(-m.stall_torque / m.no_load_speed)}" '
            f'ctrlrange="-1 1" forcerange="{_fmt(-m.torque_limit)} {_fmt(m.torque_limit)}"/>'
        )
    return "\n".join(lines)


def _sensor_xml(i: int) -> str:
    p = prefix(i)
    return (
        f'    <gyro name="{p}gyro" site="{p}imu"/>\n'
        f'    <accelerometer name="{p}accel" site="{p}imu"/>'
    )


def _check_config(rc: RobotConfig, i: int) -> None:
    if rc.ballast_mass < 0:
        raise RobotSpecError(
            f"robot {i}: component masses exceed total_mass ({rc.total_mass} kg); ballast would be negative"
        )
    if rc.wheel_y_offset + rc.wheel_width / 2 > rc.width / 2 + _LEN_TOL:
        raise RobotSpecError(f"robot {i}: wheels stick out of the chassis width")
    if rc.ground_clearance >= rc.height:
        raise RobotSpecError(f"robot {i}: ground clearance must be below chassis height")


def build_xml(cfg: EnvConfig) -> str:
    if len(cfg.robots) != 2:
        raise ValueError("exactly two robots are supported")
    for i, rc in enumerate(cfg.robots):
        _check_config(rc, i)
    ff = cfg.arena.floor_friction
    subs = dict(
        timestep=_fmt(cfg.sim.physics_dt),
        floor_friction=_fmt(ff),
        ring_radius=_fmt(cfg.arena.radius),
        inner_radius=_fmt(cfg.arena.radius - cfg.arena.border_width),
        robots="\n".join(_robot_xml(i, rc, ff) for i, rc in enumerate(cfg.robots)),
        actuators="\n".join(_actuator_xml(i, rc) for i, rc in enumerate(cfg.robots)),
        sensors="\n".join(_sensor_xml(i) for i in range(len(cfg.robots))),
    )
    return Template((ASSETS / "arena.xml").read_text()).substitute(subs)


@dataclass
class RobotHandles:
    """Cached MuJoCo indices for one robot."""

    chassis: int
    ballast: int
    shell_geom: int
    wheel_geoms: tuple
    qpos_adr: int  # freejoint qpos start (7 values)
    qvel_adr: int  # freejoint qvel start (6 values)
    wheel_qvel: tuple  # (left, right) dof addresses
    actuators: tuple  # (left, right)
    gyro_adr: int
    accel_adr: int
    subtree_bodies: tuple


def robot_handles(model: mujoco.MjModel, i: int) -> RobotHandles:
    p = prefix(i)

    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p + n)

    def gid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p + n)

    def jid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p + n)

    def aid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, p + n)

    def sadr(n):
        return int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, p + n)])

    chassis = bid("chassis")
    root = jid("root")
    subtree = tuple(
        b for b in range(model.nbody) if b == chassis or _is_descendant(model, b, chassis)
    )
    return RobotHandles(
        chassis=chassis,
        ballast=bid("ballast"),
        shell_geom=gid("shell"),
        wheel_geoms=(gid("wheel_left"), gid("wheel_right")),
        qpos_adr=int(model.jnt_qposadr[root]),
        qvel_adr=int(model.jnt_dofadr[root]),
        wheel_qvel=(
            int(model.jnt_dofadr[jid("wheel_left")]),
            int(model.jnt_dofadr[jid("wheel_right")]),
        ),
        actuators=(aid("motor_left"), aid("motor_right")),
        gyro_adr=sadr("gyro"),
        accel_adr=sadr("accel"),
        subtree_bodies=subtree,
    )


def _is_descendant(model, b: int, ancestor: int) -> bool:
    while b != 0:
        b = int(model.body_parentid[b])
        if b == ancestor:
            return True
    return False


def robot_mass(model: mujoco.MjModel, h: RobotHandles) -> float:
    return float(model.body_subtreemass[h.chassis])


def robot_footprint(model: mujoco.MjModel, data: mujoco.MjData, h: RobotHandles) -> tuple:
    """Return (length, width) of the xy bounding box of every robot geom,
    measured in the chassis frame. Requires kinematics to be up to date."""
    R = data.xmat[h.chassis].reshape(3, 3)
    origin = data.xpos[h.chassis]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    pts = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in h.subtree_bodies:
            continue
        aabb = model.geom_aabb[g]
        corners_local = aabb[:3] + signs * aabb[3:]
        corners_world = data.geom_xpos[g] + corners_local @ data.geom_xmat[g].reshape(3, 3).T
        pts.append((corners_world - origin) @ R)
    pts = np.concatenate(pts)
    return float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))


def validate_robots(model: mujoco.MjModel, data: mujoco.MjData, handles: list) -> None:
    """Raise RobotSpecError if any robot violates the footprint or mass limits."""
    for i, h in enumerate(handles):
        mass = robot_mass(model, h)
        if mass > MAX_ROBOT_MASS + _MASS_TOL:
            raise RobotSpecError(f"robot {i}: mass {mass * 1000:.1f} g exceeds {MAX_ROBOT_MASS * 1000:.0f} g")
        length, width = robot_footprint(model, data, h)
        if length > MAX_FOOTPRINT_LENGTH + _LEN_TOL or width > MAX_FOOTPRINT_WIDTH + _LEN_TOL:
            raise RobotSpecError(
                f"robot {i}: footprint {length * 100:.2f} x {width * 100:.2f} cm exceeds "
                f"{MAX_FOOTPRINT_LENGTH * 100:.0f} x {MAX_FOOTPRINT_WIDTH * 100:.0f} cm"
            )


def build_model(cfg: EnvConfig) -> tuple:
    """Compile the model and return (model, data, handles). Validates robots."""
    model = mujoco.MjModel.from_xml_string(build_xml(cfg))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    handles = [robot_handles(model, i) for i in range(len(cfg.robots))]
    validate_robots(model, data, handles)
    return model, data, handles
