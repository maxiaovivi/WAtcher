# Official RoboTwin Efficient-WAM assets and interface

WAtcher's simulation policy uses the unchanged official **14-D Efficient-WAM-RT base**. It does not replace IO layers or load real-robot task adapters. `SimulationEfficientWAMPolicy(config_path, device="cuda:0")` is lazy: importing or constructing it does not load torch, CUDA, weights, a simulator or network resources.

## Policy contract

`predict(observation, instruction=None)` accepts either WAtcher's `SimulationObservation` or an official `TASK_ENV.get_obs()` dictionary:

```python
{
    "observation": {
        "head_camera": {"rgb": head_rgb},
        "left_camera": {"rgb": left_rgb},
        "right_camera": {"rgb": right_rgb},
    },
    "joint_action": {"vector": current_drive_target14},
}
```

WAtcher's `SimulationObservation.images` uses `head`, `left`, `right`; the adapter maps those names to the native camera keys above. All images are native uint8 RGB; the head camera must be240×320. The official preprocessor resizes each wrist camera to160×120, concatenates it beneath the head camera, then pads the360×320 mosaic to384×320. This is the vendored official preprocessing, not a safety-head resize. The safety head separately uses its own trained128×128 per-camera preprocessing.

`joint_action.vector` is **current joint drive-target proxy**, not measured qpos: the pinned environment's `get_obs()` calls `robot.get_left_arm_jointState()` and `get_right_arm_jointState()`, which read `joint.get_drive_target()[0]`. Ordering is `[left6 joints, left_gripper, right6 joints, right_gripper]`. Joint targets are radians. RoboTwin grippers use **1=open, 0=closed**. Real qpos is available through `get_left_arm_real_jointState()` and `get_right_arm_real_jointState()` for separate logs; substituting it into the policy or original safety head changes the input contract.

The policy returns raw finite float32 **[16,14] absolute targets**. The official checkpoint-embedded qpos mean/std normalizes state and denormalizes actions inside the official runner; no extra minmax, target subtraction, or manual denormalization is applied. There is no gripper clipping in the policy. Native `robot.set_gripper()` clips to[0,1], so an explicit pipeline materialization mode must handle indices6/13 **before** scoring if clipping is desired, and execute that exact scored candidate afterward.

`reset(seed=None)` clears the official runner's observation history and action buffer, resets the chunk index, and rebinds the configured cached instruction. `close()` releases optional video output handles. `predict` never calls `take_action`, `setup_demo`, or any hardware interface. Scoped inference RNG uses `(seed + chunk_index*104729) mod (2**63-1)` and restores the surrounding torch RNG; it is an explicit reproducibility rule, not a claim that outputs match a different historical RNG schedule.

## External assets

Copy `examples/wam_assets.example.json`. Paths are local and resolve relative to that JSON:

| Key | Required asset |
|---|---|
| `schema` | `safe_wam_robotwin_assets/v1` |
| `checkpoint` | Complete official `Efficient-WAM-RT_stage3.pt` model/config export |
| `deploy_config` | Official RoboTwin `deploy_policy.yml` retaining14-D architecture,16-action chunk,384×320 current mosaic,192×160 future latents,10 action steps and video refresh[0,1] |
| `wan_directory` | Existing `config.json` and `Wan2.2_VAE.pth` |
| `text_embeddings` | Existing official CPU UMT5 cache containing `instruction`, `scene_prefix`, `prompt`, and one finite `[L,4096]` embedding |
| `instruction` | Exact instruction in that cache; full prompt must equal official scene prefix plus this instruction |
| `seed` | Optional integer episode seed, default0 |

The adapter uses the original compact architecture allocation and strict full checkpoint load, omitting only the redundant5B teacher initialization that would immediately be overwritten. It does not allocate/download the teacher or text encoder. An empty normalization override inherits the official checkpoint's embedded mean/std; placeholder stats paths in the example upstream YAML are not used. The loaded model must remain14-D, have normalization enabled and match the specified official inference settings. Unmatched instructions are rejected; there is no automatic text encoding, cache search or hashing.

The [vendored runtime](../src/safe_wam/vendor/efficient_wam_runtime) is from [Efficient-WAM commit2bd75a8c56acfcd5754b98c7ed313176911ccae0](https://github.com/jiajun613/Efficient-WAM/tree/2bd75a8c56acfcd5754b98c7ed313176911ccae0/inference/robotwin/EfficientWAM), with Apache-2.0 and WAN license files retained. Its package initializer omits benchmark execution entry points. Model assets retain their own terms and remain outside Git. No fresh hashes are computed; strict tensor/config loading is not fresh cryptographic file-integrity verification. Only use trusted local model files.

## Native simulator integration

The existing audited installation uses RoboTwin revision `c3ddfa8b97d5519efa828b075999bd0006778e5e`. Launch one isolated process per environment/policy, with working directory set to the RoboTwin root; its native code uses relative paths and global RNG. The official API is:

```python
task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_config)
task.set_instruction(exact_cached_instruction)
observation = task.get_obs()
candidate = policy.predict(observation, instruction=task.get_instruction())
# Score/materialize the complete candidate first. Only on an allowing model gate:
for target14 in approved_actions[:execute_steps]:
    task.take_action(target14, action_type="qpos")
task.close_env(clear_cache=False)
policy.reset()
```

The native `take_action(...,"qpos")` executes TOPP interpolation independently for each endpoint/arm at physics stepping1/250s. Endpoint duration is determined by TOPP; it is **not** a fixed`.06s` interval. The original safety profile therefore retains unknown-duration semantics. Never change its timing features merely because this is simulation. Stop on native success/horizon. In `guard` mode, do not execute a rejected chunk; `monitor` deliberately executes valid rejected chunks for simulation comparison and records that fact. Replan from the next real observation after the chosen execution prefix; policy output does not itself establish that a task is safely completed.

The `watcher-robotwin` CLI uses `safe_wam.robotwin_env` to load the installed task configuration, resolve embodiment and camera settings, and select the renderer in an isolated process. For manual integration, the native root must be on `sys.path`, together with `policy` and `description/utils`; import the selected task and use the API above. Model/text paths must be provided by the local asset JSON.

This document describes model and simulator interfaces. A model `ALLOW` remains a prediction, not proof of safety, task success, or authorization for a real robot.
