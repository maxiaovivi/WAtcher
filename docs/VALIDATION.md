# RoboTwin validation

This record concerns the RoboTwin-only implementation. It separates numerical equivalence, interface tests, actual simulator execution, task success, and safety-model quality.

- Original 06 safety-head CPU equivalence: all 18 probabilities match exactly on one existing real RoboTwin input. See [receipt](simulation_safety_cpu_check.json).
- RoboTwin contracts, policy adapter, risk gate and execution-loop tests: 45 passed.
- Installed `watcher-robotwin` console entry point: `--help` passed.

## Actual simulator integration (2026-11-08, Asia/Singapore)

Two episodes used the same `hanging_mug` randomized scene (seed 100033), instruction and policy seed (310100033). The official task horizon remained 900 actions. Each candidate contained 16 absolute 14D targets; gripper clipping was explicitly enabled before scoring. Policy inference ran on one NVIDIA L40S, with the original 06 safety head on CPU at its 0.5 threshold.

| Mode | Executed actions | Risk evaluations | Outcome | Native task success |
|---|---:|---:|---|---|
| `guard` | 96 | 7 | Six ALLOW chunks executed. The seventh scored 0.999756 and was rejected; none of its actions executed. | false |
| `monitor` | 900 | 57 | Reached the official horizon; 26 ALLOW and 31 REJECT decisions. The comparison mode executed 484 actions from rejected chunks, explicitly logged as high risk. | false |

Both processes exited, and all risk values were finite. The integration establishes actual policy execution, repeated observation and scoring, and pre-execution rejection in guard mode. **Neither episode completed the task.** This is a functional integration check on one scene, not a task-success benchmark or hazard-detection accuracy measurement. The risk classifications were not independently labeled as correct or incorrect.

Seeds, dependency versions, settings and compact outcomes are in the [integration receipt](robotwin_integration_check.json). Full local logs remain outside the repository. No threshold tuning, scene replacement or success-condition changes were used to obtain these results.

No physical robot commands are sent. An explicit `clip` setting changes only out-of-range gripper targets before scoring; the default strict mode rejects them. Monitor-mode execution of a high-risk candidate is logged as such and is never described as an allowing safety decision.
