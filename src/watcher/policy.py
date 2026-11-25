"""Efficient-WAM-RT policy: a RoboTwin observation -> 16 absolute 14-D joint targets."""
import atexit
import json
from pathlib import Path

import numpy as np

from .contracts import ACTION_DIM, HORIZON

SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)
DEPLOY_CONFIG = Path(__file__).parent / "vendor" / "efficient_wam" / "deploy_policy.yml"


def load_text_cache(path):
    """Return (instruction, [embedding [L, 4096]]) from a cache made by scripts/make_text_cache.py."""
    import torch
    cache = torch.load(path, map_location="cpu", weights_only=True)
    instruction, embeddings = cache.get("instruction"), cache.get("embeddings")
    if (not isinstance(instruction, str) or cache.get("prompt") != SCENE_PREFIX + instruction
            or len(embeddings) != 1 or embeddings[0].ndim != 2 or embeddings[0].shape[1] != 4096):
        raise ValueError(f"{path} is not a text cache for the Efficient-WAM scene prefix")
    return instruction, embeddings


class EfficientWAMPolicy:
    """The official 14-D Efficient-WAM-RT RoboTwin model with a precomputed instruction embedding.

    `assets` is a JSON file with `checkpoint`, `wan_directory` and `text_embeddings` (relative paths
    resolve from the JSON's folder), and optionally `deploy_config` to replace the bundled official
    one. Weights are loaded on the first `predict()`. Each chunk uses its own seed, so episodes are
    reproducible for a given `seed`.
    """

    def __init__(self, assets, device="cuda:0", seed=0):
        assets = Path(assets).resolve()
        spec = json.loads(assets.read_text())
        missing = {"checkpoint", "wan_directory", "text_embeddings"} - spec.keys()
        if missing:
            raise ValueError(f"{assets} is missing {sorted(missing)}")

        def path(key):
            return (assets.parent / Path(spec[key]).expanduser()).resolve(strict=True)
        self.checkpoint, self.wan_directory = path("checkpoint"), path("wan_directory")
        self.deploy_config = path("deploy_config") if "deploy_config" in spec else DEPLOY_CONFIG
        self.instruction, self._embeddings = load_text_cache(path("text_embeddings"))
        self.device, self.seed = str(device), int(seed)
        self._runner, self._chunk = None, 0

    def load(self):
        import torch
        import yaml
        from .vendor.efficient_wam import model_loader
        from .vendor.efficient_wam.preprocess import preprocess_robotwin_observation
        from .vendor.efficient_wam.runner import EfficientWAMRunner
        from .vendor.efficient_wam.third_party.wan.modules import attention

        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Efficient-WAM inference needs a CUDA device")
        self._device_index = torch.device(self.device).index or 0
        config = model_loader.apply_runtime_overrides(yaml.safe_load(self.deploy_config.read_text()), {
            "checkpoint_path": str(self.checkpoint), "wan_path": str(self.wan_directory),
            "save_predicted_video": False})
        config["action_normalization"] = {}  # use the qpos mean/std stored in the checkpoint
        config["inference"].update(scene_prefix=SCENE_PREFIX, torch_compile={"enabled": False})
        attention.FLASH_ATTN_2_AVAILABLE = attention.FLASH_ATTN_3_AVAILABLE = False  # PyTorch attention

        def text_encoder(texts, device):
            if texts != [SCENE_PREFIX + self.instruction]:
                raise ValueError("The episode instruction differs from the cached text embedding")
            return [e.to(device=device, dtype=torch.bfloat16) for e in self._embeddings]

        with torch.random.fork_rng(devices=[self._device_index]):  # building must not consume episode RNG
            runtime = model_loader.build_runtime_from_config(
                config, device=self.device, from_teacher=False, text_encoder=text_encoder)
        model = runtime.model
        if ((model.config.state_dim, model.config.action_dim, runtime.chunk_size) != (ACTION_DIM, ACTION_DIM, HORIZON)
                or runtime.video_size != (384, 320)
                or tuple(model.compact_wan.config.future_video_size or ()) != (192, 160)
                or runtime.num_inference_steps != 10 or runtime.video_refresh_steps != (0, 1)
                or not runtime.action_normalizer.enabled):
            raise ValueError(f"{self.checkpoint} does not match the official Efficient-WAM-RT RoboTwin settings")
        self._runner = EfficientWAMRunner(runtime)
        atexit.unregister(self._runner.print_task_similarity_summary)  # diagnostics printout, not needed here
        self._preprocess = preprocess_robotwin_observation
        self.reset()

    def reset(self, seed=None):
        """Start a new episode: clear the runner's history and restart the per-chunk seeds."""
        if seed is not None:
            self.seed = int(seed)
        self._chunk = 0
        if self._runner is not None:
            self._runner.reset()
            self._runner.set_instruction(self.instruction)

    def predict(self, obs, instruction=None):
        """Return float32 [16, 14] absolute joint targets; gripper values are not clipped."""
        if instruction is not None and instruction != self.instruction:
            raise ValueError("The episode instruction differs from the cached text embedding")
        if np.asarray(obs["observation"]["head_camera"]["rgb"]).shape != (240, 320, 3):
            raise ValueError("Efficient-WAM-RT expects a 240x320 RGB head camera")
        if self._runner is None:
            self.load()
        import torch
        seed = (self.seed + self._chunk * 104729) % (2**63 - 1)
        devices = [self._device_index] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            actions = self._runner.step(self._preprocess(obs, target_size=self._runner.runtime.video_size))
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise RuntimeError("Efficient-WAM returned an invalid action chunk")
        self._chunk += 1
        return actions

    def close(self):
        if self._runner is not None:
            self._runner.close_prediction_video()
