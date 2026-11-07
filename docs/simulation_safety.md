# RoboTwin safety inference

`SimulationSafetyHead` loads the original **06_offline/safety-offline.pt**.
Its only trained profile is `robotwin_legacy_target14`; the legacy checkpoint
schema happens to be named `ur5_safety_delivery/v1`. That schema name does not
make these weights a UR robot model. The public interface accepts RoboTwin
targets only and rejects the 07 or later experimental delivery formats.

```python
from safe_wam.contracts import native_to_observation, freeze_observation
from safe_wam.simulation_safety import SimulationSafetyHead

head = SimulationSafetyHead("/path/to/06_offline/safety-offline.pt", device="cpu")
raw_observation = task_env.get_obs()
observation = freeze_observation(native_to_observation(raw_observation))
risk = head.score(observation, candidate_targets)  # [16, 14], physical units
```

The policy must receive this same observation and the simulator must execute
the exact candidate that was scored, subject to the application's model gate.
This module itself neither steps the simulator nor sends robot commands.

## Exact trained contract

| Input | Meaning |
|---|---|
| Images | **One current time point**, three RGB cameras ordered head, left, right |
| Original camera keys | head_camera, left_camera, right_camera |
| State | 14 current **drive-target proxy** values; not independently measured qpos |
| State/action order | left arm six joint targets, left gripper, right arm six joint targets, right gripper |
| Arm units | Absolute revolute joint targets in radians; not TCP XYZ, joint deltas, or speedL velocities |
| Grippers | Normalized commands in [0,1]; rejected outside this range, never silently clipped here |
| Candidate | 16 consecutive absolute target vectors, shape 16×14; no padding/repetition |
| Observation times | Actual seconds relative to this decision, ≤ 0; current synchronous get_obs() is time zero |
| Model action timing | `unknown_duration`, `control_dt=None`; action timestamps are not supplied |

The simulator source establishes the state identity: `envs/_base_task.py`
`get_obs()` builds `joint_action.vector` by concatenating
`robot.get_left_arm_jointState()` and `get_right_arm_jointState()`.
`envs/robot/robot.py` implements their six arm entries with
`joint.get_drive_target()[0]`, then appends the stored gripper command.
The separate `get_*_arm_real_jointState()` methods use `entity.get_qpos()`.
Do not substitute those measured values for the trained proxy. Physical
diagnostics may record measured state separately with its correct name.

Each original camera image is independently converted to RGB and resized to
128×128 with Pillow BILINEAR, exactly as original `scripts/full_data.py:image`.
The original `CanonicalSafetyAdapter` converts uint8 to [0,1] and applies the
checkpoint's train-only means/std once. State/action statistics are shared in
this original position-target profile; they are not the separate velocity
statistics from a UR adapter. Inputs must remain in physical units.

## Decision and limitations

`RiskScores` exposes 16 step scores, post-chunk score, their original overall
maximum, checkpoint threshold (0.5), and unchanged model status. The maximum
is computed by the original temporal core over step and post log-odds; the
post term cannot be discarded. `step_scores_validated` remains false.

Training used recorded consecutive target samples whose fixed-chunk execution
was not verified. Calling RoboTwin's official `take_action(..., action_type="qpos")`
provides real simulator execution, but does not establish that its timing is
the original training distribution. Keep actual execution timing in logs;
do not invent a 0.06 s/0.1 s schedule or replace this profile with newer failed
linear-waypoint candidates. Runtime equivalence and task success do not measure
danger recall or certify this model as a safety system.

The unchanged numerical import closure and original Apache-2.0 license are in
`src/safety_verify_wam/`. No training, asset download, CUDA, or robot driver is
required for CPU scoring. NumPy is the base dependency; PyTorch and Pillow are
imported lazily when the head is used. Weights are supplied externally.

## CPU equivalence check

On 2026-11-08, one saved real RoboTwin training input was passed through the
original 06 runtime and this adapter in separate CPU processes. All input
arrays and state/action normalization matched exactly; all 18 probabilities
(16 step, post, overall) had maximum absolute difference **0.0**. Unknown
action timing was preserved. Nine numerical source modules were compared
directly and were byte-equal to the 06 delivery. No labels were used for this
check, and no simulator, training, robot actuation, or file hashing was run.

The portable receipt is `docs/simulation_safety_cpu_check.json`. The optional
reproduction entry point is:

```bash
python tests/check_simulation_safety_equivalence.py \
  --original-package /path/to/06_offline \
  --prepared /path/to/existing-trusted-robotwin-prepared.pt \
  --output /path/to/new-cpu-equivalence.json
```

The prepared archive is an existing trusted input cache. It is not downloaded
or bundled. The check selects a recorded training input by identity/provenance,
never by its label or model score. This is an interface equivalence check, not
a performance benchmark or additional accuracy measurement.
