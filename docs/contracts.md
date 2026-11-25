# Inputs, outputs and checkpoints

## Observation and action

| Item | Definition |
|---|---|
| Cameras | Current `head_camera`, `left_camera`, `right_camera` uint8 RGB from RoboTwin `get_obs()` |
| State | `get_obs()["joint_action"]["vector"]`, 14 values: the current **drive targets** (RoboTwin reads `joint.get_drive_target()`), not measured qpos |
| Order | left arm 6 joints, left gripper, right arm 6 joints, right gripper |
| Units | joints in rad; grippers in [0, 1], **1 = open, 0 = closed** |
| Candidate | `[16, 14]` absolute joint targets |
| Execution | `task.take_action(target, action_type="qpos")` per row; RoboTwin's TOPP planner decides the duration, so action timing is unknown to the models |

`--gripper-mode clip` (default) clips the two gripper columns to [0, 1] **before** scoring. RoboTwin's
`set_gripper` clips anyway, so the scored and executed chunks are identical. `strict` rejects out-of-range values.

## Policy: Efficient-WAM-RT (`watcher/policy.py`)

The official 14-D RoboTwin model, unchanged. The head camera (240x320) sits above the two wrist cameras
(resized to 160x120); the 360x320 mosaic is padded to 384x320. State and actions are normalized with the
qpos mean/std stored in the checkpoint. The model runs 10 action denoising steps, refreshes the video branch
at steps 0 and 1, and predicts 192x160 future latents. `EfficientWAMPolicy` checks these settings and refuses
other checkpoints. Instructions are encoded once, offline, by `scripts/make_text_cache.py`; the cache stores
the full prompt (scene prefix + instruction), and a different instruction at run time is an error.

Asset JSON (paths relative to the JSON's folder):

| Key | Value |
|---|---|
| `checkpoint` | stage-3 export, e.g. the official `Efficient-WAM-RT_stage3.pt` |
| `wan_directory` | Wan2.2-TI2V-5B folder with `config.json` and `Wan2.2_VAE.pth` (the 5B DiT weights are not needed) |
| `text_embeddings` | cache from `scripts/make_text_cache.py` |
| `deploy_config` | optional; default: the bundled official `deploy_policy.yml` |

Each chunk is sampled with seed `(policy_seed + chunk_index * 104729) mod (2^63 - 1)`, so an episode is
reproducible for fixed scene and policy seeds.

## Safety head (`watcher/safety.py`)

Inputs: the three current views, each resized to 128x128 (PIL bilinear); the state and the candidate in
physical units, normalized once with the checkpoint's statistics. Outputs: 16 step risks, a post-chunk risk,
and `overall = max(step, post)`. A chunk is rejected when `overall >= threshold` (0.5 for the 06 checkpoint).

Checkpoint format (a `torch.save` dict, loaded with `weights_only=True`):

| Key | Content |
|---|---|
| `profiles["robotwin_legacy_target14"]` | `state_dim = action_dim = 14`, `camera_names = [head, left, right]`, `image_size = [128, 128]`, `action_timing = "unknown_duration"`, optional `state_mean/std`, `action_mean/std` |
| `trained_profiles` | must contain `robotwin_legacy_target14` |
| `action_horizon` | 16 |
| `model_config` | `RiskModel` options; `optional_action_times` must be true, `sensor_timing` false |
| `model` | state dict, loaded with `strict=True` |
| `threshold` | in (0, 1) |

`RiskModel` re-implements the original 06 `TemporalRiskCore` (4x4 spatial grid) for a single current
observation. It keeps every parameter name, so `06_offline/safety-offline.pt` loads unchanged, and it
reproduces the original scores exactly (see [VALIDATION.md](VALIDATION.md)).
`scripts/make_random_safety_head.py` writes a random head in this format for smoke tests.
