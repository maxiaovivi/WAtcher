import json

import numpy as np
import pytest
import torch

from watcher.contracts import CAMERAS
from watcher.policy import DEPLOY_CONFIG, SCENE_PREFIX, EfficientWAMPolicy

INSTRUCTION = "Grab the mug."


@pytest.fixture
def assets(tmp_path):
    (tmp_path / "wan").mkdir()
    (tmp_path / "stage3.pt").write_bytes(b"")
    torch.save(dict(instruction=INSTRUCTION, scene_prefix=SCENE_PREFIX, prompt=SCENE_PREFIX + INSTRUCTION,
                    embeddings=[torch.zeros(7, 4096)]), tmp_path / "text.pt")
    path = tmp_path / "config" / "assets.json"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(checkpoint="../stage3.pt", wan_directory=str(tmp_path / "wan"),
                                    text_embeddings="../text.pt")))
    return path


def native(head=(240, 320, 3)):
    return {"observation": {name: {"rgb": np.zeros(head if name == "head_camera" else (240, 320, 3), np.uint8)}
                            for name in CAMERAS}, "joint_action": {"vector": np.zeros(14, np.float32)}}


def test_assets_resolve_relative_to_the_json_and_weights_load_lazily(assets):
    policy = EfficientWAMPolicy(assets)
    assert policy.checkpoint == assets.parent.parent / "stage3.pt" and policy.wan_directory.name == "wan"
    assert policy.instruction == INSTRUCTION and policy.deploy_config == DEPLOY_CONFIG and policy._runner is None
    assert DEPLOY_CONFIG.is_file()


def test_text_cache_must_match_the_scene_prefix(assets):
    cache = torch.load(assets.parent.parent / "text.pt")
    cache["prompt"] = INSTRUCTION
    torch.save(cache, assets.parent.parent / "text.pt")
    with pytest.raises(ValueError):
        EfficientWAMPolicy(assets)


def test_predict_uses_a_seed_per_chunk_and_checks_inputs(assets):
    class Runner:
        runtime = type("Runtime", (), {"video_size": (384, 320)})

        def step(self, processed):
            return torch.rand(16, 14).numpy()

        def reset(self):
            pass

        def set_instruction(self, text):
            self.text = text
    policy = EfficientWAMPolicy(assets, seed=3)
    policy._runner, policy._device_index, policy._preprocess = Runner(), 0, lambda obs, target_size: obs
    first, second = policy.predict(native()), policy.predict(native(), instruction=INSTRUCTION)
    assert first.shape == (16, 14) and first.dtype == np.float32 and not np.array_equal(first, second)
    policy.reset()
    np.testing.assert_array_equal(policy.predict(native()), first)  # same seed, same chunk index
    assert policy._runner.text == INSTRUCTION
    with pytest.raises(ValueError):
        policy.predict(native(), instruction="Another task.")
    with pytest.raises(ValueError):
        policy.predict(native(head=(128, 128, 3)))
