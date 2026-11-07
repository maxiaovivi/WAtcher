"""Run one actual RoboTwin episode with risk scoring before each candidate chunk."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", default="hanging_mug")
    parser.add_argument("--setting", choices=("clean", "randomized"), default="clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=17)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--safety-checkpoint", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--safety-device", default="cpu")
    parser.add_argument("--execute-steps", type=int, default=16)
    parser.add_argument("--max-actions", type=int, help="Optional diagnostic limit; otherwise the official task horizon")
    parser.add_argument("--mode", choices=("guard", "monitor"), default="guard")
    parser.add_argument("--gripper-mode", choices=("strict", "clip"), default="strict")
    parser.add_argument("--output-dir", type=Path, required=True)
    a = parser.parse_args()
    root = a.robotwin_root.resolve(strict=True)
    assets = a.policy_config.resolve(strict=True)
    weight = a.safety_checkpoint.resolve(strict=True)
    output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = policy = None
    started = time.monotonic()
    try:
        import torch
        from .robotwin_env import create_task
        from .simulation_policy import SimulationEfficientWAMPolicy
        from .simulation_safety import SimulationSafetyHead
        from .simulation_pipeline import GuardedSimulationPolicy
        from .simulation_loop import run_episode
        torch.set_num_threads(2)
        task = create_task(root, a.task, a.setting, a.seed, a.instruction, render_device=a.device)
        random.seed(a.policy_seed); np.random.seed(a.policy_seed); torch.manual_seed(a.policy_seed)
        policy = SimulationEfficientWAMPolicy(assets, device=a.device)
        policy.reset(seed=a.policy_seed)
        head = SimulationSafetyHead(weight, device=a.safety_device)
        controller = GuardedSimulationPolicy(head, policy, gripper_mode=a.gripper_mode)
        with (output / "risk_events.jsonl").open("x", encoding="utf-8") as log:
            def record(event):
                log.write(json.dumps(event, allow_nan=False) + "\n"); log.flush()
                print(json.dumps({"chunk": event["chunk"], "status": event["status"],
                                  "risk": None if event["risk"] is None else event["risk"]["overall"],
                                  "executed_actions": event["executed_actions"]}), flush=True)
            result = run_episode(task, controller, a.instruction, execute_steps=a.execute_steps,
                                 max_actions=a.max_actions, mode=a.mode, on_decision=record)
        result.update(task=a.task, setting=a.setting, seed=a.seed, policy_seed=a.policy_seed,
                      instruction=a.instruction, elapsed_seconds=time.monotonic()-started,
                      state_source="native joint_action.vector: drive targets plus normalized gripper commands",
                      task_success_is_not_safety_accuracy=True)
        with (output / "summary.json").open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
        print(json.dumps({k: v for k, v in result.items() if k != "events"}, allow_nan=False), flush=True)
        return 0 if result["status"] in ("TASK_SUCCESS", "TASK_HORIZON", "ACTION_LIMIT", "RISK_REJECTED") else 2
    except Exception as exc:
        failure = dict(status="FAILED", error_type=type(exc).__name__, error=str(exc),
                       elapsed_seconds=time.monotonic()-started, physical_robot_commands_sent=False)
        with (output / "failure.json").open("x", encoding="utf-8") as file:
            json.dump(failure, file, indent=2)
        print(json.dumps(failure), file=sys.stderr)
        return 2
    finally:
        if policy is not None:
            policy.close()
        if task is not None:
            task.close_env(clear_cache=False)


if __name__ == "__main__":
    raise SystemExit(main())
