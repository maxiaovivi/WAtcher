"""Original 07 safety head, consuming physical-unit candidate actions only."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .contracts import RiskScores, validate_actions, validate_observation

if TYPE_CHECKING:
    from .contracts import Observation

PROFILE = "ur5_speedl7_velocity_v1"
CAMERAS = ("external", "hand", "hand2")


def _prepare_inputs(observation: Observation, actions: np.ndarray) -> dict:
    """Match 07's dataset PIL resize, then its original tensor adapter."""
    validate_observation(observation)
    frames = observation.frames
    times = np.asarray([frame.timestamp_s for frame in frames], dtype=np.float64)
    state_time = float(observation.state_timestamp_s)
    state = np.asarray(observation.state)
    action = validate_actions(actions)
    # Import only when scoring: the base package does not require Pillow or torch.
    from PIL import Image

    history = []
    for frame in frames:
        views = []
        for camera in CAMERAS:
            rgb = np.asarray(frame.images[camera])
            image = Image.fromarray(rgb).convert("RGB")
            image = image.resize((128, 128), resample=Image.Resampling.BILINEAR)
            views.append(np.asarray(image, dtype=np.uint8).copy().transpose(2, 0, 1))
        history.append(np.stack(views))
    return {
        "video": np.stack(history)[None],
        "state": state[None, None],
        "action": action[None],
        "video_timestamps": times[None],
        "state_timestamps": np.asarray([[state_time]], dtype=np.float64),
    }


def _validate_payload(payload: dict) -> None:
    if payload.get("schema") != "ur5_safety_delivery/v1":
        raise ValueError("Expected the original ur5_safety_delivery/v1 checkpoint")
    if PROFILE not in payload.get("trained_profiles", ()):
        raise ValueError("Checkpoint has no trained UR speedL profile; 06/RT/LIBERO weights are unsupported")
    if payload.get("action_horizon") != 16:
        raise ValueError("This adapter requires the trained 16-step horizon")
    profile = payload["profiles"][PROFILE]
    expected = {
        "name": PROFILE, "state_dim": 7, "action_dim": 7,
        "control_dt": None, "action_timing": "unknown_duration",
        "state_semantics": "measured joint q0..q5 rad plus gripper position",
        "action_semantics": "base-frame TCP speedL vx vy vz m/s wx wy wz rad/s plus gripper command",
        "pixel_semantics": "rgb",
    }
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint physical state/action semantics differ from the supported 07 profile")
    if tuple(profile["camera_names"]) != CAMERAS or tuple(profile["image_size"]) != (128, 128):
        raise ValueError("Checkpoint camera order or image size differs from the supported 07 profile")
    threshold = float(payload["threshold"])
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid checkpoint threshold")
    if not isinstance(payload.get("status"), str):
        raise ValueError("Checkpoint must retain its original model status")


class SafetyHead:
    """Score proposed actions before execution; this class sends no commands.

    The original profile is named UR5. Loading it does not establish UR7e
    validation. Step/post outputs are exposed for inspection, not validated
    collision timing; use the unchanged overall max score for the decision.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        import torch
        from safety_verify_wam.portable.adapters import CanonicalSafetyAdapter
        from safety_verify_wam.portable.contracts import RobotProfile
        from safety_verify_wam.portable.multidomain import MultiProfileSafetyConfig
        from safety_verify_wam.portable.temporal_risk import TemporalRiskCore

        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        _validate_payload(payload)
        self._torch = torch
        self.device = torch.device(device)
        self.profile = RobotProfile.from_dict(payload["profiles"][PROFILE])
        self._adapter = CanonicalSafetyAdapter(self.profile)
        self.model = TemporalRiskCore(
            MultiProfileSafetyConfig.from_dict(payload["model_config"]), spatial_grid=4
        ).to(self.device).eval()
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.requires_grad_(False)
        self.threshold = float(payload["threshold"])
        self.model_status = payload["status"]

    def score(self, observation: Observation, actions: np.ndarray) -> RiskScores:
        inputs = _prepare_inputs(observation, actions)
        with self._torch.inference_mode():
            batch = self._adapter.from_tensors(
                **inputs, pixel_range="uint8", action_timestamps=None
            ).batch.to(device=self.device)
            out = self.model(PROFILE, batch)
            step = out["step_risk_probability"][0].cpu().numpy().copy()
            post = float(out["post_risk_probability"][0].cpu())
            overall = float(out["risk_probability"][0].cpu())
        if step.shape != (16,) or not np.isfinite(step).all() or not np.isfinite([post, overall]).all():
            raise RuntimeError("Safety model returned invalid scores")
        return RiskScores(
            step=step, post=post, overall=overall, threshold=self.threshold,
            step_scores_validated=False, model_status=self.model_status,
        )
