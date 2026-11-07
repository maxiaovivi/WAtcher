"""Score a RoboTwin action chunk before any action reaches the simulator."""
import numpy as np

from .contracts import (Decision, GRIPPERS, freeze_observation, native_to_observation,
                        observation_to_native, validate_actions, validate_risk)


class GuardedSimulationPolicy:
    def __init__(self, safety_head, policy=None, *, gripper_mode="strict"):
        if gripper_mode not in ("strict", "clip"):
            raise ValueError("gripper_mode must be strict or clip")
        self.safety_head, self.policy, self.gripper_mode = safety_head, policy, gripper_mode

    def evaluate(self, native_observation, instruction, actions=None):
        processing = dict(gripper_mode=self.gripper_mode, gripper_values_clipped=0)
        try:
            snapshot = freeze_observation(native_to_observation(native_observation))
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("A nonempty task instruction is required")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return Decision("INVALID_INPUT", str(exc), **processing)
        generated = actions is None
        if generated:
            if self.policy is None:
                return Decision("INVALID_INPUT", "Supply a policy or an external candidate", **processing)
            try:
                actions = self.policy.predict(observation_to_native(snapshot), instruction=instruction)
            except Exception as exc:
                return Decision("POLICY_ERROR", f"{type(exc).__name__}: {exc}", **processing)
        try:
            candidate = validate_actions(actions, self.gripper_mode)
            processing["gripper_values_clipped"] = int(np.count_nonzero(
                np.asarray(actions, dtype=np.float32)[:, GRIPPERS] != candidate[:, GRIPPERS]))
        except (ValueError, TypeError, OverflowError) as exc:
            return Decision("POLICY_ERROR" if generated else "INVALID_INPUT", str(exc), **processing)
        try:
            scoring_input = candidate.copy()
            scores = self.safety_head.score(snapshot, scoring_input)
            if not np.array_equal(scoring_input, candidate):
                raise ValueError("Scorer modified the physical candidate")
            scores = validate_risk(scores)
        except Exception as exc:
            return Decision("SCORING_ERROR", f"{type(exc).__name__}: {exc}", candidate, **processing)
        if scores.overall >= scores.threshold:
            return Decision("REJECT", "Risk meets or exceeds the model threshold", candidate, scores, **processing)
        return Decision("ALLOW", "Candidate passes the model threshold", candidate, scores, **processing)
