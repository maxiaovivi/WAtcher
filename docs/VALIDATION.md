# Validation

## 2026-11-24: refactored `watcher` package, NVIDIA H100 (Slurm node)

Environment: RoboTwin 2.0 at `c3ddfa8`, SAPIEN 3.0.3, torch 2.10.0+cu128, official
`Efficient-WAM-RT_stage3.pt` (sha256 `209ab4d6...40f`), Wan2.2-TI2V-5B VAE and UMT5 encoder.

- **Unit tests:** 49 passed (`pytest -q`, CPU and GPU node).
- **Safety head equals the original 06 implementation.** Random weights in the 06 checkpoint format, scored by
  the original `safety_verify_wam` code and by `watcher.safety`: maximum absolute difference **0** over
  3 model configurations (single profile; three profiles with profile-specific heads and coarse/chunk visual
  context; no motion encoder) x 6 observations (16 step risks, post and overall each). Two small configurations
  are kept as a regression fixture in `tests/data/safety_reference.pt`.
- **Closed loop in RoboTwin.** `hanging_mug`, randomized, scene seed 100033, policy seed 310100033, the validated
  instruction, gripper clipping on. The safety head had **random weights** (the trained 06 checkpoint is not
  on this server), so these runs check execution, scoring and rejection, not hazard detection.

| Run | Threshold | Result | Actions | Scored chunks | Time |
|---|---:|---|---:|---:|---:|
| guard | 0.5 | all chunks allowed (risk 0.383-0.475), task horizon reached | 900 | 57 | 141 s |
| monitor | 0.5 | identical risks to the guard run (reproducible seeds) | 900 | 57 | 123 s |
| guard | 0.42 | chunk 4 scored 0.433 and was rejected; none of its actions ran | 64 | 5 | 64 s |

The task was not completed in either full-length run (the same outcome as the 2026-11-08 runs below).

## 2026-11-08: original 06 checkpoint, NVIDIA L40S

With the trained `06_offline/safety-offline.pt` (threshold 0.5) on the same scene: `guard` executed 96 actions
and rejected the 7th chunk (risk 0.9998); `monitor` ran all 900 actions with 26 ALLOW and 31 REJECT decisions.
Neither episode completed the task. The new adapter matched the original 06 scores exactly on a recorded
RoboTwin input. Receipts: [validation/](validation/).
