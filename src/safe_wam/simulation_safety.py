"""Original 06 RoboTwin target14 inference, separate from robot actuation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import (
    CAMERAS, NATIVE_CAMERAS as ENV_CAMERAS, PROFILE, RiskScores,
    SimulationObservation, validate_actions, validate_simulation_observation,
)

STATE_SEMANTICS = "legacy recorded drive-target state proxy; NOT independently measured joint state"
ACTION_SEMANTICS = "recorded consecutive absolute targets; unverified fixed-chunk replay; execution duration unknown"


def prepare_simulation_inputs(observation: SimulationObservation, actions: np.ndarray) -> dict:
    """Original single-frame PIL preprocessing and physical-unit tensor layout."""
    validate_simulation_observation(observation)
    times = np.asarray([observation.image_timestamp_s, observation.state_timestamp_s], dtype=np.float64)
    state = np.array(observation.current_target14, dtype=np.float32, copy=True)
    action = validate_actions(actions)
    from PIL import Image

    views = []
    for name in CAMERAS:
        rgb = np.asarray(observation.images[name])
        image = Image.fromarray(rgb).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        views.append(np.asarray(image, dtype=np.uint8).copy().transpose(2, 0, 1))
    return {
        "video": np.stack(views)[None, None], "state": state[None, None],
        "action": action[None],
        "video_timestamps": np.asarray([[times[0]]]),
        "state_timestamps": np.asarray([[times[1]]]),
    }


def _validate_checkpoint(payload):
    if payload.get("schema") != "ur5_safety_delivery/v1":
        raise ValueError("Expected the original 06 delivery checkpoint schema")
    if set(payload.get("trained_profiles", ())) != {PROFILE}:
        raise ValueError("Use the original 06 checkpoint trained on RoboTwin target14")
    if payload.get("status") != "OFFLINE_SIMULATION_ONLY_NO_UR5_PROFILE" or payload.get("action_horizon") != 16:
        raise ValueError("Checkpoint is not the supported 06 simulation-only delivery")
    profile = payload["profiles"][PROFILE]
    expected = {"name": PROFILE, "state_dim": 14, "action_dim": 14,
                "control_dt": None, "action_timing": "unknown_duration",
                "state_semantics": STATE_SEMANTICS, "action_semantics": ACTION_SEMANTICS,
                "pixel_semantics": "rgb"}
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint RoboTwin input/execution semantics do not match")
    if tuple(profile["camera_names"]) != CAMERAS or tuple(profile["image_size"]) != (128, 128):
        raise ValueError("Checkpoint camera order/image size mismatch")
    threshold = float(payload["threshold"])
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid checkpoint threshold")


class SimulationSafetyHead:
    """Score 16 candidate absolute joint targets using the untouched 06 model.

    Action timing is unknown to this model. Actual simulator cadence belongs
    in execution records; do not invent timestamps or claim timing validation.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        import torch
        from safety_verify_wam.portable.adapters import CanonicalSafetyAdapter
        from safety_verify_wam.portable.contracts import RobotProfile
        from safety_verify_wam.portable.multidomain import MultiProfileSafetyConfig
        from safety_verify_wam.portable.temporal_risk import TemporalRiskCore

        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        _validate_checkpoint(payload)
        self._torch = torch
        self.device = torch.device(device)
        self.profile = RobotProfile.from_dict(payload["profiles"][PROFILE])
        self._adapter = CanonicalSafetyAdapter(self.profile)
        self.model = TemporalRiskCore(MultiProfileSafetyConfig.from_dict(payload["model_config"]),
                                      spatial_grid=4).to(self.device).eval()
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.requires_grad_(False)
        self.threshold = float(payload["threshold"])
        self.model_status = payload["status"]

    def score(self, observation: SimulationObservation, actions: np.ndarray) -> RiskScores:
        inputs = prepare_simulation_inputs(observation, actions)
        with self._torch.inference_mode():
            batch = self._adapter.from_tensors(**inputs, pixel_range="uint8",
                                               action_timestamps=None).batch.to(device=self.device)
            out = self.model(PROFILE, batch)
            step = out["step_risk_probability"][0].cpu().numpy().copy()
            post = float(out["post_risk_probability"][0].cpu())
            overall = float(out["risk_probability"][0].cpu())
        if step.shape != (16,) or not np.isfinite(np.r_[step, post, overall]).all():
            raise RuntimeError("Simulation safety model returned invalid scores")
        return RiskScores(step=step, post=post, overall=overall, threshold=self.threshold,
                          step_scores_validated=False, model_status=self.model_status)
