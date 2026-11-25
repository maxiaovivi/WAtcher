"""Risk gate: score one candidate chunk for one observation before anything is executed."""
from dataclasses import dataclass

import numpy as np

from .contracts import GRIPPERS, HORIZON, as_native, check_actions, read_observation


@dataclass(frozen=True)
class RiskScores:
    step: np.ndarray  # [16] risk of each action
    post: float  # risk of the state after the chunk
    overall: float  # max(step, post); this is compared with the threshold
    threshold: float


@dataclass(frozen=True)
class Decision:
    status: str  # ALLOW, REJECT or ERROR
    reason: str
    actions: np.ndarray | None = None  # the exact candidate that was scored (after gripper clipping)
    risk: RiskScores | None = None
    clipped: int = 0  # gripper values changed by gripper_mode='clip'

    @property
    def allowed(self):
        return self.status == "ALLOW"

    def to_dict(self):
        risk = None if self.risk is None else dict(
            step=self.risk.step.tolist(), post=self.risk.post, overall=self.risk.overall,
            threshold=self.risk.threshold)
        return dict(status=self.status, reason=self.reason, risk=risk, gripper_values_clipped=self.clipped)


def check_risk(scores):
    """Validate scores from any safety head; the overall risk must equal max(step, post)."""
    step = np.array(scores.step, dtype=np.float64)
    values = np.r_[step.ravel(), scores.post, scores.overall, scores.threshold].astype(np.float64)
    if step.shape != (HORIZON,) or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Risk scores must be 16 step values plus post, overall and threshold, all in [0, 1]")
    overall = max(step.max(), float(scores.post))
    if abs(overall - float(scores.overall)) > 1e-6:
        raise ValueError("Overall risk must equal max(step risks, post risk)")
    step.flags.writeable = False
    return RiskScores(step, float(scores.post), overall, float(scores.threshold))


class Gate:
    """Get a candidate (from the policy, or passed in), score it, and decide ALLOW or REJECT.

    Every failure (bad observation, policy error, invalid candidate, scoring error) becomes an
    ERROR decision, which the episode loop treats as a stop.
    """

    def __init__(self, safety_head, policy=None, *, gripper_mode="strict"):
        if gripper_mode not in ("strict", "clip"):
            raise ValueError("gripper_mode must be 'strict' or 'clip'")
        self.safety_head, self.policy, self.gripper_mode = safety_head, policy, gripper_mode

    def evaluate(self, obs, instruction, actions=None):
        try:
            images, state = read_observation(obs)
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("A non-empty task instruction is required")
            if actions is None:
                if self.policy is None:
                    raise ValueError("Pass candidate actions or build the gate with a policy")
                actions = self.policy.predict(as_native(images, state), instruction=instruction)
            candidate = check_actions(actions, self.gripper_mode)
            clipped = int(np.count_nonzero(np.asarray(actions, np.float32)[:, GRIPPERS] != candidate[:, GRIPPERS]))
        except Exception as exc:
            return Decision("ERROR", f"{type(exc).__name__}: {exc}")
        try:  # the head gets copies, so it cannot change what will be executed
            risk = check_risk(self.safety_head.score(images, state.copy(), candidate.copy()))
        except Exception as exc:
            return Decision("ERROR", f"scoring: {type(exc).__name__}: {exc}", candidate, clipped=clipped)
        if risk.overall >= risk.threshold:
            return Decision("REJECT", "risk >= threshold", candidate, risk, clipped)
        return Decision("ALLOW", "risk < threshold", candidate, risk, clipped)
