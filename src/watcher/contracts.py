"""RoboTwin observation and action conventions shared by the policy, the safety head and the gate."""
import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")
HORIZON = 16  # actions per candidate chunk
ACTION_DIM = 14  # absolute targets: left arm 6 (rad), left gripper, right arm 6 (rad), right gripper
GRIPPERS = [6, 13]  # 1 = open, 0 = closed


def read_observation(obs):
    """Copy the three RGB views and the 14-D drive-target state out of a RoboTwin `get_obs()` dict."""
    images = {}
    for name in CAMERAS:
        image = np.array(obs["observation"][name]["rgb"], copy=True)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape:
            raise ValueError(f"{name} must be an HxWx3 uint8 RGB image")
        images[name] = image
    state = _finite(obs["joint_action"]["vector"], (ACTION_DIM,), "joint_action.vector")
    if ((state[GRIPPERS] < 0) | (state[GRIPPERS] > 1)).any():
        raise ValueError("Current gripper values must lie in [0, 1]")
    return images, state


def as_native(images, state):
    """Rebuild a fresh `get_obs()`-style dict, so a consumer cannot modify the caller's copy."""
    return {"observation": {name: {"rgb": images[name].copy()} for name in CAMERAS},
            "joint_action": {"vector": state.copy()}}


def check_actions(actions, gripper_mode="strict"):
    """Return a read-only float32 [16, 14] copy; 'clip' clips the two gripper columns to [0, 1] first."""
    actions = _finite(actions, (HORIZON, ACTION_DIM), "Candidate actions")
    if gripper_mode == "clip":
        actions[:, GRIPPERS] = np.clip(actions[:, GRIPPERS], 0, 1)
    elif gripper_mode != "strict":
        raise ValueError("gripper_mode must be 'strict' or 'clip'")
    elif ((actions[:, GRIPPERS] < 0) | (actions[:, GRIPPERS] > 1)).any():
        raise ValueError("Gripper targets must lie in [0, 1]; use gripper_mode='clip' to clip them")
    actions.flags.writeable = False
    return actions


def _finite(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real-valued array of shape {shape}")
    with np.errstate(over="ignore"):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array
