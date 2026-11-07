"""Offline JSON interface. No robot driver or network listener is started."""
import argparse
import json
from pathlib import Path
import sys

from .contracts import Decision, validate_observation
from .io import load_request
from .pipeline import GuardedPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="Local safe_wam_request/v1 JSON file")
    parser.add_argument("--safety-checkpoint", type=Path, required=True)
    parser.add_argument("--safety-device", default="cpu")
    parser.add_argument("--wam-config", type=Path, help="Required only when the request has no external actions")
    parser.add_argument("--wam-device", default="cuda:0")
    parser.add_argument("--gripper-mode", choices=("strict", "clip"), default="strict",
                        help="Explicitly clip gripper values to [0,1] before scoring, or reject out-of-range values")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    decision = None
    try:
        observation, actions = load_request(args.request)
        validate_observation(observation)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        decision = Decision("INVALID_INPUT", f"Request could not be loaded: {type(exc).__name__}: {exc}")
    if decision is None:
        try:
            from .safety_head import SafetyHead
            safety = SafetyHead(args.safety_checkpoint, device=args.safety_device)
            policy = None
            if actions is None and args.wam_config is not None:
                from .efficient_wam import EfficientWAMPolicy
                policy = EfficientWAMPolicy(args.wam_config, device=args.wam_device)
        except Exception as exc:
            decision = Decision("INITIALIZATION_ERROR", f"Model initialization failed: {type(exc).__name__}: {exc}")
        else:
            decision = GuardedPolicy(safety, policy, gripper_mode=args.gripper_mode).evaluate(observation, actions)
    output = json.dumps(decision.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        # A new decision must never silently replace an existing experimental record.
        try:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(output + "\n")
        except OSError as exc:
            print(f"Cannot create output file: {exc}", file=sys.stderr)
            return 2
    print(output)
    return 0 if decision.status in ("ALLOW", "REJECT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
