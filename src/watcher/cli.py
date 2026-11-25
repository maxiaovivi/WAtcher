"""Run one RoboTwin episode in which every Efficient-WAM chunk is risk-scored before execution."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback


def main(argv=None):
    p = argparse.ArgumentParser(prog="watcher-robotwin", description=__doc__)
    p.add_argument("--robotwin-root", type=Path, required=True, help="RoboTwin 2.0 checkout with assets")
    p.add_argument("--assets", type=Path, required=True, help="policy asset JSON (examples/assets.example.json)")
    p.add_argument("--safety-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="new directory for the episode logs")
    p.add_argument("--task", default="hanging_mug")
    p.add_argument("--setting", choices=("clean", "randomized"), default="clean")
    p.add_argument("--seed", type=int, default=0, help="scene seed")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--instruction", help="default: the instruction stored in the text cache")
    p.add_argument("--mode", choices=("guard", "monitor"), default="guard",
                   help="guard: stop at the first rejected chunk; monitor: execute it anyway and log it")
    p.add_argument("--execute-steps", type=int, default=16, help="actions executed per chunk before replanning")
    p.add_argument("--max-actions", type=int, help="short diagnostic runs; default: the task's step limit")
    p.add_argument("--gripper-mode", choices=("clip", "strict"), default="clip",
                   help="clip: clip grippers to [0,1] before scoring (RoboTwin clips anyway); strict: reject")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--safety-device", default="cpu")
    a = p.parse_args(argv)
    for name in ("robotwin_root", "assets", "safety_checkpoint", "output_dir"):  # RoboTwin changes the cwd
        setattr(a, name, getattr(a, name).expanduser().resolve())
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started, task, policy = time.monotonic(), None, None
    try:
        import random

        import numpy as np
        import torch

        from .gate import Gate
        from .loop import run_episode
        from .policy import EfficientWAMPolicy
        from .robotwin import create_task
        from .safety import SafetyHead

        policy = EfficientWAMPolicy(a.assets, device=a.device, seed=a.policy_seed)
        instruction = a.instruction or policy.instruction
        head = SafetyHead(a.safety_checkpoint, device=a.safety_device)
        task = create_task(a.robotwin_root, a.task, setting=a.setting, seed=a.seed, instruction=instruction,
                           render_device=a.device)
        random.seed(a.policy_seed)
        np.random.seed(a.policy_seed)
        torch.manual_seed(a.policy_seed)
        with (a.output_dir / "risk_events.jsonl").open("x") as log:
            def record(event):
                log.write(json.dumps(event) + "\n")
                log.flush()
                risk = event["risk"] and round(event["risk"]["overall"], 4)
                print(json.dumps(dict(chunk=event["chunk"], status=event["status"], risk=risk,
                                      executed=event["executed"])), flush=True)
            result = run_episode(task, Gate(head, policy, gripper_mode=a.gripper_mode), instruction,
                                 execute_steps=a.execute_steps, max_actions=a.max_actions, mode=a.mode,
                                 on_event=record)
        result.pop("events")
        result.update(task=a.task, setting=a.setting, seed=a.seed, policy_seed=a.policy_seed,
                      instruction=instruction, safety_checkpoint=str(a.safety_checkpoint),
                      safety_checkpoint_status=head.status, seconds=round(time.monotonic() - started, 1))
        (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return 2 if result["status"] == "ERROR" else 0
    except Exception as exc:
        failure = dict(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                       seconds=round(time.monotonic() - started, 1))
        (a.output_dir / "failure.json").write_text(json.dumps(failure, indent=2))
        print(failure["traceback"], file=sys.stderr)
        return 2
    finally:
        if policy is not None:
            policy.close()
        if task is not None:
            task.close_env(clear_cache=False)


if __name__ == "__main__":
    raise SystemExit(main())
