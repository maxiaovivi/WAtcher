# Efficient-WAM UR assets

WAtcher supports the audited **UR frozen-video A** task checkpoint schema, `full_action_frozen_video_v1`. The reference deployment is `pp26_wam_A_pick_sampling1x`, task/prompt `pick`, checkpoint `best_step014000`. This is the user-retained robot-tested version; it is not interchangeable with a new training best, the official 14-D RoboTwin model, video-LoRA B, goal-push, tower-phase, or OpenWAM exports. Those formats are rejected rather than guessed.

## Input and output

`EfficientWAMPolicy(config_path, device="cuda:0").predict(observation)` uses the last real frame's three RGB cameras (`external`, `hand`, `hand2`) and measured state `[q0..q5, gripper_feedback]`: six joint angles in radians plus feedback in `[0,1]`. The frame timestamp must be finite and `<=0` relative to decision time. Different native camera resolutions are supported through the asset contract; they must match the recorded training/deployment shapes. WAM consumes only the current frame; the safety head independently consumes the four real history frames.

The returned float32 `[16,7]` candidate is **robot-base TCP speedL**, columns `[vx,vy,vz,wx,wy,wz,gripper]`, with m/s, rad/s, and `0=open, 1=closed`. This is not a joint-target or delta-position protocol. Recorded WAM sampling is 15Hz; the safety model's historical speedL profile retains unknown action-duration semantics. Do not silently substitute the `.06s` RoboTwin linear-waypoint contract.

The adapter preserves PyAV AREA resizing into an RGB224 letterbox, removal of its known padding, then OpenCV AREA three-view mosaic `[384,320,3]`. State min/max normalization and action denormalization occur once; constant state dimensions map to zero. No normalization clipping, speed clipping, gripper thresholding, or action repetition is applied. The downstream candidate validator can reject out-of-range gripper predictions.

`hand_4_3_to_16_9_canvas=true` is only for the historically recorded pick camera conversion: raw hand images `[480,640,3]` are placed inside the trained `[720,1280,3]` canvas using a `[720,960]` resize and 160-pixel side padding. Do not enable it for the reference profile whose hand shapes are already `[480,640,3]`.

## Required local files

Copy `examples/wam_assets.example.json` and supply explicit local paths (relative paths resolve against that JSON):

| Key | Asset |
|---|---|
| `checkpoint` | The matching trusted UR task checkpoint with `manifest`, `step`, and complete `trainable_state` |
| `checkpoint_step` | Exact selected checkpoint step, e.g. 14000; not the run's target step |
| `base_checkpoint` | Existing official `Efficient-WAM-RT_stage3.pt` complete model/config export |
| `base_config` | Original task `base_config.json`, retaining official model architecture and identity metadata |
| `wan_directory` | Existing WAN `config.json` and `Wan2.2_VAE.pth`; the 5B teacher is not loaded |
| `text_embeddings` | Matching cached task/prompt UMT5 `text-embeddings.pt`, one finite `[L,4096]` embedding |
| `normalization` | Original `train_only_normalization.json`, exactly matching the task checkpoint manifest |
| `camera_shapes` | Original native RGB shapes, keyed by all three cameras |

Also required: `schema=safe_wam_ur_assets/v1`, `format=full_action_frozen_video_v1`, exact `task` and `prompt`. The official base starts at 14-D state/actions. The adapter allocates its compact model, strict-loads that complete export, replaces the three UR IO linears with 7-D FP32 layers, then requires all 130 action-expert tensors (138,987,271 parameters), exact shapes and finite FP32 values. The pretrained compact video branch remains frozen. The task checkpoint is **not** standalone: all listed dependencies must match. Existing digest strings are retained as historical identity metadata, without computing fresh hashes; this does not constitute new cryptographic integrity verification. Only load checkpoints from a trusted source.

All loading is lazy at first `predict`; importing/constructing the policy does not import torch, allocate CUDA, download assets, or contact a robot. Cached text avoids constructing a text encoder. The vendored loader still imports tokenizer definitions, so its Python dependencies include transformers/ftfy/regex. Install the `wam` extra in an appropriate existing CUDA PyTorch environment. FlashAttention is optional and disabled in this adapter in favor of the original SDPA fallback; `decord` is not required for current-frame inference.

For externally supplied physical actions, use WAtcher's evaluate-only path. It does not need a WAM policy or any WAM assets and never substitutes generated actions for the supplied candidate. No API here sends robot commands.

## Observed smoke boundary

One real-record CUDA smoke loaded the reference pick A step14000 export and generated a finite float32 `[16,7]` candidate. It used four consecutive native video frames from one recorded episode, its measured joint/gripper state and measured relative source timestamps, with LeRobot-v3 episode video offsets checked. The raw gripper range was `[0.9796699, 1.0052298]`; the default strict pipeline correctly returned `POLICY_ERROR` before safety scoring because the final value exceeds `[0,1]`.

The saved raw candidate was then evaluated with the separately explicit `gripper_mode="clip"` contract, using the same source frame mapping and measured times. Exactly one gripper value changed to1; all six speed columns remained bitwise unchanged. The original07 safety head returned finite16 step probabilities plus post/overall, with step maximum `0.00024346`, post/overall `0.06958139`, and threshold `.5`. The model gate returned `ALLOW`; its approved actions exactly matched the materialized candidate that was scored. This second stage used **zero new WAM generations**. It does not authorize actuation or establish safety/generalization. The raw WAM model is unchanged, strict remains the default, and the chosen materialization mode/count is reported in the decision.

The opt-in `tests/smoke_wam_record.py` and `tests/rescore_wam_record.py` reproduce these two distinct stages with existing local assets; record extraction additionally needs PyArrow. The small public summary omits private paths, images and action values. These are functional smoke observations rather than robot success statistics.

## Source provenance and licenses

Vendored inference code comes from [Efficient-WAM commit 2bd75a8c56acfcd5754b98c7ed313176911ccae0](https://github.com/jiajun613/Efficient-WAM/tree/2bd75a8c56acfcd5754b98c7ed313176911ccae0/inference/robotwin/EfficientWAM), whose [Apache-2.0 license](https://github.com/jiajun613/Efficient-WAM/blob/2bd75a8c56acfcd5754b98c7ed313176911ccae0/LICENSE) is included. The 23 retained Python files were byte-compared with the existing audited runtime; no hashes were generated. Only the package `__init__.py` is changed to remove robot benchmark entry imports; `deploy_policy.py` and benchmark task/launch files are omitted.

WAN component copyright headers remain intact, and its [Apache-2.0 license](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt) is included separately. Third-party model assets retain their own terms; no model weights, data, robot logs, credentials, or private experiment records are part of this repository. The adapter's IO replacement, task restoration, mosaic and runner settings follow the existing `wam_model.py` and verified UR deployment wrapper. Real-robot success records are provenance for the chosen artifact, not a safety guarantee for this combined system.
