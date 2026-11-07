"""RoboTwin absolute joint-target contracts, in native physical units."""
from dataclasses import dataclass
from typing import Mapping

import numpy as np

CAMERAS = ("head", "left", "right")
NATIVE_CAMERAS = ("head_camera", "left_camera", "right_camera")
PROFILE = "robotwin_legacy_target14"
HORIZON = 16
GRIPPERS = (6, 13)


@dataclass(frozen=True)
class SimulationObservation:
    images: Mapping[str, np.ndarray]
    current_target14: np.ndarray
    image_timestamp_s: float = 0.0
    state_timestamp_s: float = 0.0


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
    def allowed(self):
        return self.status == "ALLOW"

    @property
    def approved_actions(self):
        return self.candidate_actions if self.allowed else None

    def to_dict(self, *, include_actions=True):
        risk = self.risk
        result = {
            "status": self.status, "allowed": self.allowed, "reason": self.reason,
            "profile": PROFILE,
            "risk": None if risk is None else {
                "step": risk.step.tolist(), "post": risk.post, "overall": risk.overall,
                "threshold": risk.threshold, "step_scores_validated": risk.step_scores_validated,
                "model_status": risk.model_status,
            },
            "action_processing": {"gripper_mode": self.gripper_mode,
                                  "gripper_values_clipped": self.gripper_values_clipped},
            "action_semantics": "absolute joint targets: left6, left_gripper, right6, right_gripper",
            "action_timing": "unknown_duration",
            "physical_robot_commands_sent": False,
        }
        if include_actions:
            result.update(candidate_actions=None if self.candidate_actions is None else self.candidate_actions.tolist(),
                          approved_actions=None if self.approved_actions is None else self.approved_actions.tolist())
        return result


def _finite_real(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.isrealobj(array) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be real numeric {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must remain finite in float32")
    return array


def validate_simulation_observation(observation):
    if set(observation.images) != set(CAMERAS):
        raise ValueError("Expected exactly head, left and right RGB views")
    for name in CAMERAS:
        image = np.asarray(observation.images[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
            raise ValueError(f"{name} must be H x W x 3 uint8 RGB")
    target = _finite_real(observation.current_target14, (14,), "Current drive-target proxy")
    if ((target[list(GRIPPERS)] < 0) | (target[list(GRIPPERS)] > 1)).any():
        raise ValueError("Current gripper commands must lie in [0,1]")
    for timestamp in (observation.image_timestamp_s, observation.state_timestamp_s):
        value = np.asarray(timestamp)
        if value.ndim != 0 or not np.isrealobj(value) or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value) or value > 0:
            raise ValueError("Observation timestamps must be finite real seconds <= 0 relative to this decision")


def native_to_observation(native):
    observation = SimulationObservation(
        {name: native["observation"][camera]["rgb"] for name, camera in zip(CAMERAS, NATIVE_CAMERAS)},
        np.asarray(native["joint_action"]["vector"]),
    )
    validate_simulation_observation(observation)
    return observation


def freeze_observation(observation):
    def frozen(value):
        array = np.array(value, copy=True)
        array.flags.writeable = False
        return array
    return SimulationObservation({k: frozen(observation.images[k]) for k in CAMERAS},
                                 frozen(observation.current_target14),
                                 float(observation.image_timestamp_s), float(observation.state_timestamp_s))


def observation_to_native(observation):
    return {"observation": {camera: {"rgb": observation.images[name].copy()}
                            for name, camera in zip(CAMERAS, NATIVE_CAMERAS)},
            "joint_action": {"vector": observation.current_target14.copy()}}


def validate_actions(actions, gripper_mode="strict"):
    if gripper_mode not in ("strict", "clip"):
        raise ValueError("gripper_mode must be strict or clip")
    array = _finite_real(actions, (HORIZON, 14), "Candidate absolute joint targets")
    if gripper_mode == "clip":
        array[:, GRIPPERS] = np.clip(array[:, GRIPPERS], 0, 1)
    elif ((array[:, GRIPPERS] < 0) | (array[:, GRIPPERS] > 1)).any():
        raise ValueError("Gripper targets must lie in [0,1]; no silent simulator-side clipping")
    array.flags.writeable = False
    return array


def validate_risk(scores):
    raw_step = np.asarray(scores.step)
    if not np.isrealobj(raw_step) or not np.issubdtype(raw_step.dtype, np.number):
        raise ValueError("Risk scores must be real numbers")
    for value in (scores.post, scores.overall, scores.threshold):
        array = np.asarray(value)
        if array.ndim != 0 or not np.isrealobj(array) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("Post risk, overall risk and threshold must be real scalars")
    step = np.asarray(raw_step, dtype=np.float64)
    if step.shape != (HORIZON,):
        raise ValueError("Expected 16 step risks")
    values = np.r_[step, scores.post, scores.overall, scores.threshold]
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Risk probabilities and threshold must be finite and in [0,1]")
    aggregate = max(float(step.max()), float(scores.post))
    if not np.isclose(scores.overall, aggregate, rtol=0, atol=1e-6):
        raise ValueError("Overall risk must preserve max(step, post)")
    step = step.copy(); step.flags.writeable = False
    return RiskScores(step, float(scores.post), aggregate, float(scores.threshold),
                      scores.step_scores_validated, str(scores.model_status))
