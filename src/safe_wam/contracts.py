"""Physical-unit interfaces shared by the policy and safety head."""
from dataclasses import dataclass
from typing import Mapping

import numpy as np

CAMERAS = ("external", "hand", "hand2")
PROFILE = "ur5_speedl7_velocity_v1"
HORIZON = 16


@dataclass(frozen=True)
class Frame:
    """A real synchronized camera observation; time is relative to the decision."""

    images: Mapping[str, np.ndarray]
    timestamp_s: float


@dataclass(frozen=True)
class Observation:
    frames: tuple[Frame, ...]
    state: np.ndarray
    state_timestamp_s: float


@dataclass(frozen=True)
class RiskScores:
    step: np.ndarray
    post: float
    overall: float
    threshold: float
    step_scores_validated: bool = False
    model_status: str = "research"


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    candidate_actions: np.ndarray | None = None
    risk: RiskScores | None = None
    gripper_mode: str = "strict"
    gripper_values_clipped: int = 0

    @property
    def allowed(self) -> bool:
        return self.status == "ALLOW"

    @property
    def approved_actions(self) -> np.ndarray | None:
        """The exact scored candidate, only when the model gate passes."""
        return self.candidate_actions if self.allowed else None

    def to_dict(self) -> dict:
        risk = self.risk
        return {
            "status": self.status,
            "allowed": self.allowed,
            "reason": self.reason,
            "profile": PROFILE,
            "candidate_actions": None if self.candidate_actions is None else self.candidate_actions.tolist(),
            "approved_actions": None if self.approved_actions is None else self.approved_actions.tolist(),
            "risk": None if risk is None else {
                "step": risk.step.tolist(), "post": risk.post, "overall": risk.overall,
                "threshold": risk.threshold, "step_scores_validated": risk.step_scores_validated,
                "model_status": risk.model_status,
            },
            "action_timing": "unknown_duration",
            "action_processing": {"gripper_mode": self.gripper_mode,
                                  "gripper_values_clipped": self.gripper_values_clipped},
            "robot_commands_sent": False,
            "execution_authorized": False,
        }


def validate_observation(observation: Observation) -> None:
    if len(observation.frames) != 4:
        raise ValueError("Four real camera observations are required; do not repeat or pad frames")
    timestamps = np.asarray([f.timestamp_s for f in observation.frames], dtype=np.float64)
    if not np.isfinite(timestamps).all() or (timestamps > 0).any() or not (np.diff(timestamps) > 0).all():
        raise ValueError("Frame timestamps must be finite, strictly increasing and <= 0")
    if not np.isfinite(observation.state_timestamp_s) or observation.state_timestamp_s > 0:
        raise ValueError("Measured state timestamp must be finite and <= 0")
    state = np.asarray(observation.state)
    if state.shape != (7,) or not np.issubdtype(state.dtype, np.number) or not np.isrealobj(state) or not np.isfinite(state).all():
        raise ValueError("Expected seven finite measured state values: six joints in rad and gripper feedback")
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.isfinite(state.astype(np.float32)).all():
            raise ValueError("Measured state must remain finite in float32")
    if not 0 <= state[6] <= 1:
        raise ValueError("Measured gripper feedback must use the trained [0, 1] range")
    for frame in observation.frames:
        if set(frame.images) != set(CAMERAS):
            raise ValueError("Each observation must contain external, hand and hand2 cameras")
        for name in CAMERAS:
            image = np.asarray(frame.images[name])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
                raise ValueError(f"{name} must be an H x W x 3 uint8 RGB image")


def freeze_observation(observation: Observation) -> Observation:
    """Take a decision-local snapshot so concurrent camera updates cannot change it."""
    def frozen_array(value):
        array = np.array(value, copy=True)
        array.flags.writeable = False
        return array
    return Observation(
        tuple(Frame({name: frozen_array(f.images[name]) for name in CAMERAS}, float(f.timestamp_s))
              for f in observation.frames),
        frozen_array(observation.state), float(observation.state_timestamp_s),
    )


def validate_actions(actions, gripper_mode: str = "strict") -> np.ndarray:
    if gripper_mode not in ("strict", "clip"):
        raise ValueError("gripper_mode must be strict or clip")
    raw = np.asarray(actions)
    if raw.shape != (HORIZON, 7) or not np.issubdtype(raw.dtype, np.number) or not np.isrealobj(raw):
        raise ValueError("Expected exactly 16 x 7 real physical speedL actions; no padding or repetition")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(raw, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("Actions must remain finite in float32")
    if gripper_mode == "clip":
        array[:, 6] = np.clip(array[:, 6], 0, 1)
    elif ((array[:, 6] < 0) | (array[:, 6] > 1)).any():
        raise ValueError("Gripper commands must use the trained [0, 1] range")
    array.flags.writeable = False
    return array


def validate_risk(scores: RiskScores) -> RiskScores:
    raw_step = np.asarray(scores.step)
    if not np.isrealobj(raw_step) or not np.issubdtype(raw_step.dtype, np.number):
        raise ValueError("Risk scores must be real numbers")
    for value in (scores.post, scores.overall, scores.threshold):
        array = np.asarray(value)
        if array.ndim != 0 or not np.isrealobj(array) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("Post risk, overall risk and threshold must be real scalar numbers")
    step = np.asarray(raw_step, dtype=np.float64)
    if step.shape != (HORIZON,):
        raise ValueError("Safety head must return one risk score per candidate action")
    values = np.r_[step, scores.post, scores.overall, scores.threshold]
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Risk probabilities and threshold must be finite and in [0, 1]")
    aggregate = max(float(step.max()), float(scores.post))
    if not np.isclose(scores.overall, aggregate, rtol=0, atol=1e-6):
        raise ValueError("Overall risk must preserve the trained max(step, post) aggregation")
    step = step.copy()
    step.flags.writeable = False
    return RiskScores(step, float(scores.post), aggregate, float(scores.threshold),
                      scores.step_scores_validated, str(scores.model_status))
