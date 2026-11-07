"""One decision snapshot, one physical candidate and one pre-execution risk check."""
from typing import Protocol

import numpy as np

from .contracts import (
    Decision, Observation, RiskScores, freeze_observation, validate_actions,
    validate_observation, validate_risk,
)


class Policy(Protocol):
    def predict(self, observation: Observation) -> np.ndarray: ...


class Scorer(Protocol):
    def score(self, observation: Observation, actions: np.ndarray) -> RiskScores: ...


class GuardedPolicy:
    def __init__(self, safety_head: Scorer, policy: Policy | None = None, *, gripper_mode: str = "strict"):
        if gripper_mode not in ("strict", "clip"):
            raise ValueError("gripper_mode must be strict or clip")
        self.safety_head = safety_head
        self.policy = policy
        self.gripper_mode = gripper_mode

    def _candidate(self, actions):
        candidate = validate_actions(actions, gripper_mode=self.gripper_mode)
        changed = int(np.count_nonzero(np.asarray(actions, dtype=np.float32)[:, 6] != candidate[:, 6]))
        return candidate, changed

    def evaluate(self, observation: Observation, actions=None) -> Decision:
        """Generate or accept physical actions; return a decision without executing."""
        processing = {"gripper_mode": self.gripper_mode, "gripper_values_clipped": 0}
        try:
            validate_observation(observation)
            snapshot = freeze_observation(observation)
            candidate = None
            if actions is not None:
                candidate, processing["gripper_values_clipped"] = self._candidate(actions)
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            return Decision("INVALID_INPUT", str(exc), **processing)
        if candidate is None:
            if self.policy is None:
                return Decision("INVALID_INPUT", "Supply candidate actions or configure an Efficient-WAM policy", **processing)
            try:
                candidate, processing["gripper_values_clipped"] = self._candidate(self.policy.predict(snapshot))
            except Exception as exc:
                return Decision("POLICY_ERROR", f"Policy inference failed: {type(exc).__name__}: {exc}", **processing)
        try:
            # Scorers may adapt tensors, but may not change the physical candidate returned to the caller.
            scoring_input = candidate.copy()
            scores = self.safety_head.score(snapshot, scoring_input)
            if not np.array_equal(scoring_input, candidate):
                raise ValueError("Safety scorer modified the candidate")
            scores = validate_risk(scores)
        except Exception as exc:
            return Decision("SCORING_ERROR", f"Risk evaluation failed: {type(exc).__name__}: {exc}", candidate, **processing)
        if scores.overall >= scores.threshold:
            return Decision("REJECT", "Predicted risk meets or exceeds the model threshold", candidate, scores, **processing)
        return Decision("ALLOW", "Candidate passes the model threshold; no robot execution is authorized", candidate, scores, **processing)
