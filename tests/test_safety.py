from pathlib import Path

import numpy as np
import pytest
import torch

from watcher.contracts import CAMERAS
from watcher.safety import PROFILE, RiskModel, SafetyHead, random_checkpoint

FIXTURE = Path(__file__).parent / "data" / "safety_reference.pt"  # recorded with the original 06 code


def images(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    return {name: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for name in CAMERAS}


@pytest.mark.parametrize("case", ["position_target", "multi_profile"])
def test_scores_match_the_original_implementation(case, tmp_path):
    fixture = torch.load(FIXTURE, weights_only=True)
    torch.save(fixture["cases"][case]["checkpoint"], tmp_path / "head.pt")
    head = SafetyHead(tmp_path / "head.pt")
    for x, expected in zip(fixture["inputs"], fixture["cases"][case]["expected"]):
        views = {f"{k}_camera": v.numpy() for k, v in x["images"].items()}
        risk = head.score(views, x["state"].numpy(), x["actions"].numpy())
        np.testing.assert_array_equal(risk.step, expected["step"].numpy())
        assert (risk.post, risk.overall) == (expected["post"], expected["overall"])


def test_random_head_scores_one_chunk(tmp_path):
    head = SafetyHead(random_checkpoint(tmp_path / "random.pt"))
    actions = np.full((16, 14), 0.1, np.float32)
    risk = head.score(images(), np.zeros(14, np.float32), actions)
    assert risk.step.shape == (16,) and head.status == "RANDOM_WEIGHTS" and risk.threshold == 0.5
    assert risk.overall == pytest.approx(max(risk.step.max(), risk.post), abs=1e-6)
    assert risk.overall == head.score(images(), np.zeros(14, np.float32), actions).overall  # deterministic


@pytest.mark.parametrize("field,value", [("action_timing", "seconds"), ("camera_names", ["left", "head", "right"]),
                                         ("image_size", [64, 64]), ("action_dim", 7)])
def test_rejects_checkpoints_with_another_input_contract(tmp_path, field, value):
    payload = torch.load(random_checkpoint(tmp_path / "head.pt"), weights_only=True)
    payload["profiles"][PROFILE][field] = value
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        SafetyHead(tmp_path / "bad.pt")


@pytest.mark.parametrize("option", [dict(sensor_timing=True), dict(optional_action_times=False), dict(unknown=1)])
def test_rejects_unsupported_model_options(option):
    config = dict(profiles=[dict(key=PROFILE, state_dim=14, action_dim=14, motion_mode="position_target")],
                  optional_action_times=True)
    with pytest.raises(ValueError):
        RiskModel({**config, **option})
