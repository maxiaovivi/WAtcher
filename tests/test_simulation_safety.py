import copy
import unittest

import numpy as np

from safe_wam.contracts import freeze_observation, native_to_observation
from safe_wam.simulation_safety import (
    ACTION_SEMANTICS, CAMERAS, ENV_CAMERAS, PROFILE, STATE_SEMANTICS,
    SimulationObservation, _validate_checkpoint, prepare_simulation_inputs,
)


class SimulationSafetyTest(unittest.TestCase):
    def test_actual_observation_mapping_and_independent_resize(self):
        from PIL import Image
        images = {name: {"rgb": np.full((17 + i, 23 + i, 3), i * 41, np.uint8)}
                  for i, name in enumerate(ENV_CAMERAS)}
        target = np.zeros(14, np.float32)
        target[6], target[13] = 0.2, 0.8
        raw = {"observation": images, "joint_action": {"vector": target}}
        obs = freeze_observation(native_to_observation(raw))
        raw["joint_action"]["vector"][0] = 9
        self.assertEqual(obs.current_target14[0], 0)
        out = prepare_simulation_inputs(obs, np.zeros((16, 14), np.float32))
        self.assertEqual(out["video"].shape, (1, 1, 3, 3, 128, 128))
        self.assertEqual(out["state"].shape, (1, 1, 14))
        self.assertNotIn("action_timestamps", out)
        for i, camera in enumerate(CAMERAS):
            expected = np.asarray(Image.fromarray(obs.images[camera]).convert("RGB").resize(
                (128, 128), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
            np.testing.assert_array_equal(out["video"][0, 0, i], expected)

    def test_wrong_dimension_timing_and_clipping_rejected(self):
        obs = SimulationObservation({name: np.zeros((20, 30, 3), np.uint8) for name in CAMERAS},
                                    np.zeros(14, np.float32))
        for action in (np.zeros((16, 7)), np.zeros((8, 14)), np.full((16, 14), np.nan)):
            with self.assertRaises(ValueError):
                prepare_simulation_inputs(obs, action)
        bad = np.zeros((16, 14), np.float32)
        bad[0, 6] = 1.1
        with self.assertRaises(ValueError):
            prepare_simulation_inputs(obs, bad)
        with self.assertRaises(ValueError):
            prepare_simulation_inputs(SimulationObservation(obs.images, obs.current_target14, 0.1),
                                      np.zeros((16, 14)))

    def test_only_original_trained_profile_contract(self):
        profile = dict(name=PROFILE, state_dim=14, action_dim=14, control_dt=None,
                       action_timing="unknown_duration", state_semantics=STATE_SEMANTICS,
                       action_semantics=ACTION_SEMANTICS, pixel_semantics="rgb",
                       camera_names=CAMERAS, image_size=(128, 128))
        p = dict(schema="ur5_safety_delivery/v1", trained_profiles=[PROFILE],
                 status="OFFLINE_SIMULATION_ONLY_NO_UR5_PROFILE", action_horizon=16,
                 profiles={PROFILE: profile}, threshold=0.5)
        _validate_checkpoint(p)
        for field, value in (("state_semantics", "measured qpos"), ("control_dt", 0.06)):
            bad = copy.deepcopy(p)
            bad["profiles"][PROFILE][field] = value
            with self.assertRaises(ValueError):
                _validate_checkpoint(bad)
        p["trained_profiles"] = ["ur5_speedl7_velocity_v1"]
        with self.assertRaises(ValueError):
            _validate_checkpoint(p)


if __name__ == "__main__":
    unittest.main()
