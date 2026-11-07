"""Small interface checks; real weights are covered by the optional CPU check."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from safe_wam.contracts import Frame, Observation
from safe_wam.safety_head import CAMERAS, PROFILE, _prepare_inputs, _validate_payload


def observation():
    return Observation(tuple(
        Frame({name: np.full((19 + i, 31 + j, 3), i * 10 + j, np.uint8)
               for j, name in enumerate(CAMERAS)}, (-0.3, -0.2, -0.1, 0.0)[i])
        for i in range(4)
    ), np.zeros(7, np.float32), 0.0)


class SafetyInputsTest(unittest.TestCase):
    def test_separate_camera_resize_matches_training(self):
        from PIL import Image
        obs = observation()
        raw = _prepare_inputs(obs, np.zeros((16, 7), np.float32))
        self.assertEqual(raw["video"].shape, (1, 4, 3, 3, 128, 128))
        for i, frame in enumerate(obs.frames):
            for j, name in enumerate(CAMERAS):
                expected = np.asarray(Image.fromarray(frame.images[name]).convert("RGB").resize(
                    (128, 128), resample=Image.Resampling.BILINEAR)).transpose(2, 0, 1)
                np.testing.assert_array_equal(raw["video"][0, i, j], expected)
        self.assertNotIn("action_timestamps", raw)
        np.testing.assert_array_equal(raw["state"], obs.state[None, None])

    def test_bad_history_and_short_actions_fail(self):
        obs = observation()
        with self.assertRaises(ValueError):
            _prepare_inputs(Observation(obs.frames[:3], obs.state, 0.0), np.zeros((16, 7)))
        with self.assertRaises(ValueError):
            _prepare_inputs(Observation((obs.frames[0],) * 4, obs.state, 0.0), np.zeros((16, 7)))
        with self.assertRaises(ValueError):
            _prepare_inputs(obs, np.zeros((8, 7)))
        with self.assertRaises(ValueError):
            _prepare_inputs(Observation(obs.frames, obs.state, 0.1), np.zeros((16, 7)))

    def test_base_import_is_lazy(self):
        root = str(Path(__file__).resolve().parents[1] / "src")
        code = (f"import sys;sys.path.insert(0,{root!r});import safe_wam.safety_head;"
                "assert 'torch' not in sys.modules;assert 'PIL' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_untrained_or_wrong_semantics_rejected(self):
        p = {"schema": "ur5_safety_delivery/v1", "trained_profiles": [], "profiles": {}}
        with self.assertRaises(ValueError):
            _validate_payload(p)
        profile = {"name": PROFILE, "state_dim": 7, "action_dim": 7, "control_dt": None,
                   "action_timing": "unknown_duration", "camera_names": CAMERAS,
                   "image_size": (128, 128), "pixel_semantics": "rgb",
                   "state_semantics": "measured joint q0..q5 rad plus gripper position",
                   "action_semantics": "base-frame TCP speedL vx vy vz m/s wx wy wz rad/s plus gripper command"}
        p.update(trained_profiles=[PROFILE], profiles={PROFILE: profile}, action_horizon=16,
                 threshold=0.5, status="LEGACY_REAL_DEVELOPMENT_ONLY_NOT_APPROVED_FOR_ACTUATION")
        _validate_payload(p)
        wrong = copy.deepcopy(p)
        wrong["profiles"][PROFILE]["action_timing"] = "seconds"
        with self.assertRaises(ValueError):
            _validate_payload(wrong)


if __name__ == "__main__":
    unittest.main()
