Build a sumo robot simulation in Python using MuJoCo (official `mujoco` bindings) with a Gymnasium-style environment. This is a research testbed for later training a JEPA world model, so prioritize clean data and configurability over graphics. Do NOT implement any ML training yet.

ARENA (hard constraints)
- Flat circular ring, diameter exactly 154 cm, black surface.
- White border band 2.54 cm wide at the outer edge (included within the 154 cm diameter).
- A robot loses when its chassis center leaves the ring (configurable). Episode ends on a loss or a timeout (default 30 s).

ROBOTS (two, identical by default; hard constraints)
- Footprint: 20 cm long x 20 cm wide. Mass: 3000 g total. No height limit; use 8 cm chassis height as a configurable default.
- Validate at startup that every robot satisfies the footprint and mass constraints, and raise an error if not.
- Differential drive: two driven wheels, low-friction front/rear skids. This means turning happens by adjusting the speed of each wheel.
- Action per robot: [left_motor, right_motor], each in [-1, 1]. Model the motor with a torque limit and a simple first-order response so commands aren't instant. Choose default torque so a 3 kg bot accelerates and pushes realistically (keep it configurable).
- Physics at ~500 Hz, control at 50 Hz (configurable).

SENSORS (each with configurable Gaussian noise, dropout and latency)
- 5 time-of-flight rays at -60, -30, 0, 30, 60 degrees from the front, max range 1.6 m, returning distance (max range if nothing hit). They should detect the opponent but not the floor.
- 4 line sensors at the chassis corners returning 1 if over the white band, else 0.
- IMU: yaw rate and forward/lateral acceleration.
- Wheel encoders: left and right wheel angular velocity.
- Return one flat observation vector per robot and document the index of every field.

GOOD-VIEW SUPPORT
- Add env.get_observation_pair(robot_id) that returns TWO independently noised observations of the exact same physical state (noise and dropout resampled, physics untouched). This will allow for better positive pair inputs. Ensure the noise is a small error difference.

GROUND TRUTH
- Every step, put true state in `info`: both robots' positions, headings, velocities, each robot's distance to the ring edge, and the opponent's position relative to each robot. This will be used later to train readout heads, never as model input.

OPPONENT POLICIES (pluggable interface: obs -> action)
- random: random wheel commands held for random short durations
- charger: rotate until the center ToF sees the opponent, then drive forward full speed
- searcher: spin or sweep, then charge when any ToF sees something
- rule_based baseline: charge if the opponent is seen, back up and turn if a line sensor fires. Usable for either robot.
- Starting positions: random, plus the standard facing-each-other start.

DOMAIN RANDOMIZATION (config flag, per episode)
- Floor and wheel friction, motor strength, sensor noise levels, control latency.
- Mass and footprint stay fixed at the constraints above; only randomize how mass is distributed within the chassis.

DATA COLLECTION
- Script collect.py: run N episodes with chosen policies for both robots and save to .npz: obs, action, next_obs, a second noisy obs of the same state (from get_observation_pair), ground-truth info, done flags, episode ids.
- Bias action choices so the data includes contact, pushing and edge events, not just driving around.

VISUALIZATION AND TESTS
- Interactive MuJoCo viewer mode, plus a script that renders a top-down MP4 of an episode.
- Sanity tests: ring is 154 cm with a 2.54 cm border; each robot is 20x20 cm and 3000 g; equal motor commands drive straight; a ToF reports the correct distance to an opponent placed at a known position; a line sensor fires over the white band; pushing an opponent out triggers the loss; runs are reproducible with a fixed seed.

PROJECT LAYOUT
- sumo_sim/ (env, robot model XML, sensors, policies, config), collect.py, render.py, tests/, README with setup and the observation layout.
- Keep dependencies minimal: mujoco, numpy, gymnasium, imageio.
