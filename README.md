# sumo-bots

A MuJoCo simulation of two differential-drive sumo robots on a 154 cm ring.
It is built as a clean, configurable data source for training a JEPA-style
world model later on. There is no ML training code here: just the
environment, scripted opponent policies, data collection, rendering and tests.

## Requirements

- **Python 3.10 or newer**
- **Python packages** (installed in step 2 below):
  - `mujoco` (≥ 3.1; developed against 4.0.0): the physics engine and viewer
  - `numpy`: arrays
  - `gymnasium`: the environment API
  - `imageio[ffmpeg]`: writing MP4 videos
  - `pytest`: optional, only for running the tests
- **A display** for the interactive viewer. This works on Linux, macOS,
  Windows, and WSL2 on Windows 11 (through WSLg). Recording MP4 videos needs
  no display.

## Getting started

### 1. Download the project

**Clone the repository and move into its folder:**
```bash
git clone https://github.com/AE1030/sumo-bots.git
cd sumo-bots
```

### 2. Create the virtual environment and install dependencies

**Create an isolated Python environment in a `.venv` folder:**
```bash
python3 -m venv .venv
```

**Activate ("source") the environment so `python3` uses it:**
```bash
source .venv/bin/activate
```
Your prompt now starts with `(.venv)`. On Windows PowerShell, run
`.venv\Scripts\Activate.ps1` instead.

**Install the required packages into the environment:**
```bash
pip install -r requirements.txt pytest
```

You only need to do step 2 once. In each new terminal afterwards, just `cd`
into the folder and run `source .venv/bin/activate` again. Run `deactivate`
to leave the environment.

### 3. Check that everything works

**Run the sanity tests (about 1 second):**
```bash
python3 -m pytest
```

## Running the simulation

All commands assume you are in the project folder with the environment
activated (step 2).

### Interactive viewer

**Open a live 3D window with the default matchup (red rule_based vs blue
searcher, standard face-off start):**
```bash
python3 render.py --interactive
```
A new match starts automatically when one robot is pushed out or after the
30-second timeout. Close the window to stop.

In the window:
- Left-drag rotates the camera, right-drag moves it, and scrolling zooms.
- Space pauses.
- Double-click a robot to select it, then Ctrl + right-drag to shove it.

### Starting positions

**Standard start: robots 60 cm apart, facing each other (the default):**
```bash
python3 render.py --interactive --start-mode standard
```

**Random start: random positions within 55 cm of the center, random headings,
at least 35 cm apart:**
```bash
python3 render.py --interactive --start-mode random
```

**Mixed start: each match picks standard or random with 50/50 odds:**
```bash
python3 render.py --interactive --start-mode mixed
```

### Choosing the robots' behaviour

The available policies are:

| policy | behaviour |
|---|---|
| `rule_based` | charges when it sees the opponent, backs off the white border, otherwise rotates to search |
| `charger` | rotates until its center sensor sees the opponent, then drives at full speed |
| `searcher` | spins or sweeps in arcs, then charges whatever its sensors see |
| `random` | random wheel commands, each held for a short time |
| `idle` | doesn't move (useful as a target) |

**Pick the policy for robot 0 (red, `--p0`) and robot 1 (blue, `--p1`):**
```bash
python3 render.py --interactive --p0 charger --p1 searcher
```

**Combine a matchup with a starting-position mode:**
```bash
python3 render.py --interactive --p0 rule_based --p1 random --start-mode random
```

### Domain randomization

**Vary floor and wheel friction, motor strength, sensor noise, control latency
and weight distribution every match:**
```bash
python3 render.py --interactive --randomize
```

### Reproducible runs

**Fix the random seed so the same command replays the same match:**
```bash
python3 render.py --interactive --start-mode random --seed 42
```

### Recording a video instead of opening a window

**Save a top-down MP4 of one match (works without a display):**
```bash
python3 render.py --p0 rule_based --p1 charger --start-mode random --out episode.mp4
```

**Record from an angled camera instead of top-down, stopping after 10 seconds:**
```bash
python3 render.py --camera angled --max-seconds 10 --out angled.mp4
```

### Collecting training data

**Run 200 matches with mixed policies and starts, then save them to a `.npz` file:**
```bash
python3 collect.py --episodes 200 --out data/sumo.npz -v
```

**Collect with domain randomization and random starts only:**
```bash
python3 collect.py --episodes 200 --randomize --start-mode random --out data/sumo_rand.npz
```

**Collect unbiased scripted play with a fixed matchup (no event-seeking
overrides, no action noise):**
```bash
python3 collect.py --episodes 100 --p0 rule_based --p1 charger --bias 0 --action-noise 0 --out data/plain.npz
```

**List every option for a script:**
```bash
python3 render.py --help
python3 collect.py --help
```

### Changing defaults in code

Everything else is set in `sumo_sim/config.py`. That includes:
- the ring size and the rule for when a robot is out
- motor torque and speed
- sensor noise, dropout and latency
- physics and control rates
- the match timeout
- how far apart the robots start, plus `start_jitter` to add small variations
  to the standard start

You can also set these from Python:

```python
import numpy as np
from sumo_sim import SumoEnv, EnvConfig, make_policy

cfg = EnvConfig()
cfg.sim.start_mode = "random"
cfg.randomization.enabled = True
env = SumoEnv(cfg)
obs, info = env.reset(seed=0)                 # obs: (2, 14) float32
pols = [make_policy("rule_based"), make_policy("searcher")]
for p in pols:
    p.reset(seed=1)
action = np.stack([p.act(o) for p, o in zip(pols, obs)])   # (2, 2) in [-1, 1]
obs, reward, terminated, truncated, info = env.step(action)
view_a, view_b = env.get_observation_pair(0)  # two noisy views of the same state
```

`SingleAgentWrapper(env, opponent_policy, agent_id=0)` turns it into a
standard single-agent Gymnasium env where the other robot follows a fixed policy.

## Layout

```
sumo_sim/
  config.py        dataclass configs (arena, robot, motor, sensors, sim, randomization)
  assets/arena.xml arena template (ring, floor, cameras)
  assets/robot.xml robot template (chassis, skids, wheels, ballast)
  model.py         builds the MJCF from config; validates footprint and mass
  sensors.py       ToF / line / IMU / encoder models, noise, dropout, latency
  env.py           SumoEnv (2 robots) and SingleAgentWrapper
  policies.py      idle, random, charger, searcher, rule_based
  viewer.py        interactive MuJoCo viewer
collect.py         data collection -> .npz
render.py          top-down MP4 (or --interactive)
tests/             sanity tests
```

## Arena

- Flat ring, diameter **154 cm**, black surface, with a **2.54 cm** white border
  band inside that diameter. The ring is drawn with visual-only geoms. The
  physical floor is an infinite plane, so a robot that leaves the ring keeps
  driving on the floor instead of falling off.
- Loss rule (`ArenaConfig.out_criterion`): `"center"` (default; the chassis
  center leaves the 77 cm radius), `"any_corner"`, or `"all_corners"`.
  If both robots go out in the same step, the episode is a draw (`winner == -1`).
- Episodes end on a loss (`terminated`) or after `SimConfig.timeout`
  (default 30 s; `truncated`).
- Physics runs at 500 Hz (`physics_dt = 0.002`) and control at 50 Hz
  (`control_dt = 0.02`).

## Robots

| property | default | |
|---|---|---|
| footprint | 20 × 20 cm | checked at startup and on every reset: max ≤ 20 × 20 cm |
| mass | 3000 g | checked on every reset: ≤ 3000 g |
| chassis height | 8 cm | `RobotConfig.height` |
| wheels | 2 × Ø6 cm, on the center axle at y = ±8 cm | |
| skids | front and rear spheres, μ = 0.05 | |
| ballast | the rest of the mass (1.40 kg) | its position sets the center of mass |

`model.validate_robots` builds an xy bounding box from every geom in each
robot's body tree (in the chassis frame) and sums its subtree mass. If a robot
is over either limit it raises `RobotSpecError`, so building or resetting the env
fails with that error.

**Motor model.** Each wheel uses a MuJoCo `general` actuator that models a DC motor:

```
act' = (cmd - act) / time_constant              # first-order response (50 ms)
tau  = stall_torque * act - (stall_torque / no_load_speed) * omega
tau  = clip(tau, -torque_limit, torque_limit)
```

The defaults are 0.4 N·m stall torque at the wheel and a 40 rad/s no-load speed.
With these the robot reaches about 1 m/s in 0.3 s and peaks at about
4 m/s² of acceleration, so wheel traction is the limit. A pushing robot shoves
an idle robot out from the standard start in about 1.5 s. Two identical robots
pushing head-on stalemate, as they would in reality.

**Friction.** The wheels and skids have contact priority over the floor, so
their friction value is the one MuJoCo uses. The wheel/floor coefficient is
`floor_friction × wheel_friction`. Chassis contacts (robot vs robot, and chassis
vs floor if a robot tips) use `chassis_friction`.

## Observation layout (per robot, `float32[14]`)

| idx | field | units | notes |
|---|---|---|---|
| 0 | `tof_m60` | m | ToF at −60° (right of front) |
| 1 | `tof_m30` | m | −30° |
| 2 | `tof_0` | m | straight ahead |
| 3 | `tof_p30` | m | +30° (left) |
| 4 | `tof_p60` | m | +60° |
| 5 | `line_fl` | {0,1} | front-left corner; 1 = over the white band |
| 6 | `line_fr` | {0,1} | front-right |
| 7 | `line_rl` | {0,1} | rear-left |
| 8 | `line_rr` | {0,1} | rear-right |
| 9 | `gyro_yaw` | rad/s | yaw rate, counter-clockwise positive |
| 10 | `accel_fwd` | m/s² | body-x acceleration, averaged over the control step |
| 11 | `accel_lat` | m/s² | body-y (left) acceleration |
| 12 | `enc_left` | rad/s | left wheel speed (positive = forward) |
| 13 | `enc_right` | rad/s | right wheel speed |

The same layout is available in code as `sumo_sim.OBS_FIELDS` and `sumo_sim.OBS_INDEX`.
The env returns a `(2, 14)` array, where row `i` is robot `i`'s own sensors.

- **ToF:** five horizontal rays from the center of the front face, 4 cm above
  the ground, with a maximum range of 1.6 m. A ray that hits nothing returns
  exactly 1.6. The rays only see chassis geoms (geom group 3), never the floor,
  the ring, wheels or their own robot. The cast starts 2 cm behind the face so a
  touching opponent reads about 0.
- **Line sensors:** placed at the corners, 1 cm in from the footprint edge.
  The sensor reads a reflectance of 1 (white) or 0 (black/outside), adds
  Gaussian noise, and thresholds at 0.5.
- **IMU:** the MuJoCo gyro and accelerometer, averaged over the substeps of each
  control step.
- **Noise** (`SensorConfig.<group>` is a `NoiseConfig` with fields `std`,
  `rel_std`, `dropout`, `latency`). The groups are `tof`, `line`, `gyro`,
  `accel` and `encoder`. A reading that drops out is replaced by max range (ToF)
  or 0 (everything else). Latency is counted in control steps, per group.
- **`get_observation_pair(robot_id)`** returns two observations of the same
  latency-delayed clean reading. Each view draws its own noise and dropout, and
  the physics is not touched. With the default noise, the median gap between
  the two views is a few mm on the ToF, about 0.02 rad/s on the gyro, about
  0.1 m/s² on the accelerometer and about 0.2 rad/s on the encoders. Line
  sensors almost never differ.
- `get_clean_observation(robot_id)` returns the observation without noise. It is
  for debugging and tests.

## Ground truth (`info`)

Every `reset`/`step` returns ground truth in `info`. Use it as targets for
readout heads, never as model input:

`time`, `pos (2,2)`, `heading (2,)`, `vel_world (2,2)`, `vel_body (2,2)`,
`yaw_rate (2,)`, `dist_to_edge (2,)` (negative once outside),
`opp_rel_pos_world (2,2)`, `opp_rel_pos_body (2,2)` (opponent in each robot's
frame), `opp_distance`, `opp_bearing`, `contact` (robots touching),
`line_truth (2,4)`, `wheel_speed (2,2)`, `applied_action (2,2)` (after the
latency delay), `winner` (None while running, 0/1, −1 for a draw), `rewards (2,)`,
and `episode_params` (the domain-randomization draw for the episode).

## Policies

Every policy has the interface `reset(seed)` followed by `act(obs) -> [left, right]`.
Policies only see the noisy observation, so any of them can drive either robot.

- `random`: picks a uniform random wheel command and holds it for 5–25 steps.
- `charger`: rotates until the center ToF sees something, then drives at full speed.
- `searcher`: spins or sweeps in arcs (chosen per episode) and backs away from
  the border while searching. When any ToF sees something it charges, steering
  toward the closest return.
- `rule_based`: if a line sensor fires, it backs up (or drives forward, if the
  rear sensor fired) and turns away. Otherwise it charges with steering when it
  sees the opponent, and rotates slowly to search when it doesn't.
- `idle`: always outputs zero.

Start positions (`SimConfig.start_mode`, or `reset(options={"start_mode": ...})`):
- `standard`: facing each other 60 cm apart, with optional jitter.
- `random`: non-overlapping positions within a 55 cm radius and random headings.
- `mixed`: picks one of the two per episode.

You can also pass explicit poses with `options={"poses": [(x, y, yaw), (x, y, yaw)]}`.

## Domain randomization

Set `RandomizationConfig.enabled = True`. The following are sampled at each
reset, independently per robot when `per_robot` is set:

- floor friction
- wheel friction multiplier
- motor strength (scales stall torque and torque limit)
- sensor-noise scale
- control latency (0–2 steps)
- ballast position (±5 cm in x and y, ±1 cm in z), which moves the center of mass

Total mass and footprint never change, and they are re-validated on every reset.

## Data collection

`collect.py` writes one compressed `.npz` file. Axis 1 is always the robot index:

| key | shape | |
|---|---|---|
| `obs`, `next_obs` | (T, 2, 14) | what the policies saw, before and after the step |
| `obs_pair` | (T, 2, 2, 14) | two more independently noised views of the state in `obs` |
| `action` | (T, 2, 2) | commanded action |
| `done`, `terminated`, `truncated` | (T,) | |
| `episode_id`, `step` | (T,) | |
| `actor_mode` | (T, 2) | index into `actor_modes`: which action source drove the step |
| `gt_<key>` / `gt0_<key>` | (T, …) | ground truth after / before the step |
| `ep_winner`, `ep_policies`, `ep_params` | (E, …) | per episode (winner −2 = timeout) |
| `config`, `obs_fields`, `actor_modes` | | metadata (config as JSON) |

**Event bias.** Pure scripted play spends a lot of time searching. To get more
contact, pushing and border events into the data:

- **Policy mixture:** with `--p0 mix` and `--p1 mix` (the default), each episode
  draws its policies from a mixture weighted toward engaging policies.
- **Action overrides:** for a fraction `--bias` of 10–40-step segments, a
  robot's policy is replaced by one of three behaviors (0.3 by default):
  - `seek_opponent`: steers at the opponent using ground truth.
  - `seek_edge`: drives toward the ring border.
  - `random_hold`: holds a random command.
- **Action noise:** Gaussian noise (`--action-noise`) is added to every action.
- **Near-edge starts:** a fraction `--edge-start` of episodes start near the
  border.

Each run prints a summary of contact, pushing and line-sensor frequencies.
On a 20-episode run with the defaults:

- contact: about 50% of steps
- pushing (in contact and moving): about 40% of steps
- every episode ended in a push-out

Pass `--bias 0 --action-noise 0` for unbiased scripted play.

## Tests

The tests in `tests/test_sanity.py` check that:

- the ring is 154 cm across with a 2.54 cm border
- each robot is 20 × 20 cm and 3000 g, including under randomization
- oversized or overweight robots raise `RobotSpecError`
- equal motor commands drive straight at several headings
- the ToF reads the correct distance to an opponent at a known position (0° and 30°)
- the ToF ignores the floor
- a line sensor fires over the white band
- pushing an idle opponent out ends the episode with the right winner and rewards
- the timeout truncates the episode
- runs with a fixed seed are reproducible (with and without randomization)
- the observation pair keeps physics untouched and differs only slightly
- motor lag and control latency behave as configured
- every policy outputs valid actions
