# Safety head interface

`SafetyHead(checkpoint, device="cpu").score(observation, actions)` scores a
proposed action chunk before execution. It sends no robot commands and does
not need policy internals. The policy output and an external action array use
the same physical-unit interface.

The supported asset is the original **07_offline_plus_real**
`safety-offline-plus-real.pt`, schema `ur5_safety_delivery/v1`, trained profile
`ur5_speedl7_velocity_v1`. Loading is strict and uses `weights_only=True`.
The 06 offline checkpoint has no trained UR velocity profile and is rejected.
Experimental RoboTwin/LIBERO adapters are not replacements for this asset.
Weights must be supplied locally; nothing is downloaded automatically.

## Inputs

| Field | Contract |
|---|---|
| `observation.frames` | Exactly four real observations, oldest to newest |
| `frame.images` | `external`, `hand`, `hand2`, each original H×W×3 uint8 RGB |
| `frame.timestamp_s` | Actual seconds relative to the decision, strictly increasing, all ≤ 0 |
| `observation.state` | Seven measured values: six joint positions in radians, then gripper feedback in the trained [0,1] convention |
| `observation.state_timestamp_s` | Actual measurement time relative to the decision, ≤ 0 |
| `actions` | Exactly 16×7 physical values: base-frame TCP `vx,vy,vz` m/s, `wx,wy,wz` rad/s, then gripper command in [0,1] |

The gripper's physical open/close polarity must match the training recording;
the numeric range alone does not establish this mapping. State is **joint
position**, not TCP pose, velocity, or a manufactured state history.
Images may have different resolutions. Each is independently converted to RGB
and resized to 128×128 with Pillow BILINEAR, matching the original 07 training
dataset (`portable/data.py:268–275`). The unchanged tensor adapter then divides
uint8 by 255 and applies the checkpoint's separate train-only state/action
means and standard deviations exactly once. Do not pre-normalize inputs.

No padding, repeating history, or repeating short action chunks is provided.
The caller is responsible for obtaining real observations; identical pixels
from a stationary camera are not evidence that a frame was fabricated.
Action duration remains `unknown_duration`, `control_dt=None`. This trained
profile does not accept guessed execution timestamps or a fabricated 0.06 s
schedule. It is different from the newer RoboTwin linear-waypoint protocol.

## Outputs

`RiskScores` contains `step[16]`, `post`, `overall`, `threshold`,
`step_scores_validated=False`, and the original `model_status`.
The unchanged `TemporalRiskCore` takes the maximum of the 16 step log-odds and
the post-chunk log-odds, then maps it to risk probability. Consequently overall
is the maximum of the step and post probabilities, up to floating-point
rounding. The post score must not be discarded. The original threshold is
loaded from the checkpoint (0.5 for the supplied 07 asset).

Step/post scores are inspectable model outputs, **not validated collision
times or a calibrated probability of independent events**. The checkpoint was
trained with chunk labels. An `ALLOW` result is a model decision, not physical
execution authorization or a certified safety guarantee.

## Actual validation scope

The supplied 07 evidence is historical UR5 development data: 196 training
examples and 61 development examples, split by recorded video group. At the
fixed 0.5 threshold, its selected development result is TP 17, FN 9, FP 16,
TN 19 (36/61 accuracy). Label timing limitations remain. It is not an
independent blind test or a robot task-success test, and it does not establish
UR7e deployment validity. The repository preserves this status instead of
presenting later RT/LIBERO results as UR performance.

## Dependencies and provenance

The base API uses NumPy; scoring lazily imports PyTorch and Pillow. The supplied
delivery documents Python 3.10 and PyTorch 2.7.1. The focused CPU equivalence
check additionally records its actual runtime versions. No Efficient-WAM,
simulator, CUDA, robot-driver, training dataset, or network service is required
for safety-only inference.

The minimal original numerical modules and their Apache-2.0 license are in
`src/safety_verify_wam/`; see that directory's `PROVENANCE.md`. The package
initializers were replaced to remove eager imports; numerical modules remain
unchanged. Weights and recorded images are external assets.

Run the optional real-asset equivalence check with:

```bash
python tests/check_safety_equivalence.py \
  --original-package /path/to/07_offline_plus_real \
  --output /path/to/new-equivalence-receipt.json
```

It compares the original runtime and this adapter on the archived example,
checks all step/post/overall values on CPU, and exercises different-resolution
camera preprocessing. It does not command hardware, train, or compute file
hashes. The original adapter's internal profile-configuration fingerprint is
retained unchanged.

The 2026-11-08 check passed with PyTorch 2.4.1, NumPy 1.26.4 and Pillow 12.3.0
on CPU: all 18 probabilities (16 step, post, overall) matched exactly. The
example's overall/post score was 0.7217370272 while its maximum step score was
0.0033148949. Nine copied numerical modules were byte-equal to the original
delivery; this is runtime-equivalence evidence, not a new safety accuracy test.
