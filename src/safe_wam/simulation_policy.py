"""Official 14-D Efficient-WAM RoboTwin policy; no robot or simulator execution."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

CAMERAS = ('head_camera', 'left_camera', 'right_camera')
PUBLIC_CAMERAS = ('head', 'left', 'right')
SCENE_PREFIX = (
    'The whole scene is in a realistic, industrial art style with three views: '
    'a fixed rear camera, a movable left arm camera, and a movable right arm camera. '
    'The aloha robot is currently performing the following task: '
)


def native_observation(observation):
    """Preserve official RGB and current drive targets; do not substitute actual qpos."""
    if isinstance(observation, Mapping):
        native = observation
    else:
        for name in ('image_timestamp_s', 'state_timestamp_s'):
            value = getattr(observation, name)
            if not np.isfinite(value) or value > 0:
                raise ValueError('Observation times must be finite and <= decision time')
        if set(observation.images) != set(PUBLIC_CAMERAS):
            raise ValueError('SimulationObservation requires head, left and right RGB images')
        native = {'observation': {k: {'rgb': observation.images[p]} for k,p in zip(CAMERAS,PUBLIC_CAMERAS)},
                  'joint_action': {'vector': observation.current_target14}}
    state = np.asarray(native['joint_action']['vector'])
    if state.shape != (14,) or not np.isrealobj(state) or not np.isfinite(state).all():
        raise ValueError('Expected official finite current drive-target vector14, not 7D state')
    for name in CAMERAS:
        image = np.asarray(native['observation'][name]['rgb'])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError(f'{name}: expected native HWC uint8 RGB')
        if name == 'head_camera' and image.shape != (240, 320, 3):
            raise ValueError('Official RoboTwin mosaic requires head_camera RGB240x320')
    return native


def _path(root, value):
    path = Path(value).expanduser()
    return (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def _read_config(path):
    if path.suffix.lower() in ('.yml', '.yaml'):
        import yaml
        return yaml.safe_load(path.read_text(encoding='utf-8'))
    return json.loads(path.read_text(encoding='utf-8'))


def _validate_text(cache, instruction, torch):
    if (cache.get('instruction') != instruction or cache.get('scene_prefix') != SCENE_PREFIX
        or cache.get('prompt') != SCENE_PREFIX + instruction):
        raise ValueError('An exact existing official scene-prefix/instruction text cache is required')
    embeddings = cache['embeddings']
    if (len(embeddings) != 1 or embeddings[0].ndim != 2 or not 0 < embeddings[0].shape[0] <= 512
        or embeddings[0].shape[1] != 4096 or not bool(torch.isfinite(embeddings[0]).all())):
        raise ValueError('Expected one finite cached UMT5 [L,4096] embedding')
    return embeddings


@contextmanager
def _student_and_cached_text(loader, encoder):
    """Skip overwritten teacher initialization and text-encoder allocation only."""
    from .vendor.efficient_wam_runtime.models.compact_wan import CompactWANModel
    from .vendor.efficient_wam_runtime.models.wan_model import WanVideoModel, _load_wan_arch_config
    previous_student = CompactWANModel.__dict__['from_teacher_checkpoint']
    previous_text = loader._build_t5_encoder
    def allocate(cls, config, device='cuda'):
        architecture = dict(_load_wan_arch_config(config.config_path or config.checkpoint_path))
        architecture.update(config.to_wan_model_config())
        architecture['num_layers'] = int(config.num_layers)
        return cls(config=config, video_model=WanVideoModel(model_config=architecture, vae_path=config.vae_path,
                                                          device=device, precision=config.precision))
    CompactWANModel.from_teacher_checkpoint = classmethod(allocate)
    loader._build_t5_encoder = lambda config, device: encoder
    try:
        yield
    finally:
        CompactWANModel.from_teacher_checkpoint = previous_student
        loader._build_t5_encoder = previous_text


class SimulationEfficientWAMPolicy:
    """Lazy official base policy returning raw16x14 absolute targets for pre-execution scoring."""
    def __init__(self, config_path, device='cuda:0'):
        self.config_path = Path(config_path)
        self.device = str(device)
        self._runner = None
        self._chunk_index = 0
        self._seed_override = None

    def _load(self):
        import torch
        from .vendor.efficient_wam_runtime import model_loader
        from .vendor.efficient_wam_runtime.runner import EfficientWAMRunner
        from .vendor.efficient_wam_runtime.preprocess import preprocess_robotwin_observation
        from .vendor.efficient_wam_runtime.third_party.wan.modules import attention
        cfg = _read_config(self.config_path); root = self.config_path.resolve().parent
        required = {'schema', 'checkpoint', 'deploy_config', 'wan_directory', 'text_embeddings', 'instruction'}
        if not required <= cfg.keys() or cfg['schema'] != 'safe_wam_robotwin_assets/v1':
            raise ValueError('Expected safe_wam_robotwin_assets/v1 official14D assets')
        if not isinstance(cfg['instruction'], str) or not cfg['instruction'].strip():
            raise ValueError('A nonempty exact cached instruction is required')
        if not self.device.startswith('cuda:') or not torch.cuda.is_available():
            raise ValueError('Official WAM inference requires an explicitly selected available CUDA device')
        self._device_index = torch.device(self.device).index
        config = _read_config(_path(root, cfg['deploy_config']))
        action = config['model']['action_expert']
        if (action['state_dim'], action['action_dim'], action['chunk_size']) != (14, 14, 16):
            raise ValueError('Only official unchanged14D/16-action architecture is supported')
        config = model_loader.apply_runtime_overrides(config, {
            'checkpoint_path': str(_path(root, cfg['checkpoint'])),
            'wan_path': str(_path(root, cfg['wan_directory'])), 'save_predicted_video': False,
        })
        # Empty override deliberately inherits official checkpoint-embedded qpos mean/std.
        config['action_normalization'] = {}
        config['inference']['scene_prefix'] = SCENE_PREFIX
        config['inference']['torch_compile'] = {'enabled': False}
        cache = torch.load(_path(root, cfg['text_embeddings']), map_location='cpu', weights_only=True)
        embeddings = _validate_text(cache, cfg['instruction'], torch)
        def encoder(texts, device):
            if texts != [SCENE_PREFIX + cfg['instruction']]:
                raise ValueError('Instruction not covered by the explicit local text cache')
            return [e.to(device=device, dtype=torch.bfloat16) for e in embeddings]
        attention.FLASH_ATTN_2_AVAILABLE = attention.FLASH_ATTN_3_AVAILABLE = False
        with torch.random.fork_rng(devices=[self._device_index]), _student_and_cached_text(model_loader, encoder):
            runtime = model_loader.build_runtime_from_config(config, device=self.device)
        if (runtime.model.config.state_dim, runtime.model.config.action_dim, runtime.chunk_size) != (14, 14, 16):
            raise ValueError('Loaded official model dimensions differ')
        if (runtime.video_size != (384, 320) or tuple(runtime.model.compact_wan.config.future_video_size) != (192,160)
            or runtime.num_inference_steps != 10 or runtime.video_refresh_steps != (0,1)
            or not runtime.action_normalizer.enabled):
            raise ValueError('Official RT preprocessing/denoising/qpos normalization contract differs')
        runtime.model.eval().requires_grad_(False)
        self._runner = EfficientWAMRunner(runtime)
        self._preprocess = preprocess_robotwin_observation
        self._instruction = cfg['instruction']
        self._seed = int(cfg.get('seed', 0)) if self._seed_override is None else self._seed_override
        self.metadata = dict(state_semantics='official current drive-target proxy14; not measured qpos',
                             action_semantics='absolute left6joint+gripper/right6joint+gripper targets',
                             normalization=runtime.action_normalizer.config(),chunk_size=16,
                             action_timing='unknown_duration; native TOPP execution determines duration',
                             rng='scoped per chunk: (seed + chunk_index*104729) mod (2**63-1)',
                             official_checkpoint_loaded_strict=True,official_14d_io_unchanged=True)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self._seed_override = int(seed)
            self._seed = int(seed)
        self._chunk_index = 0
        if self._runner is not None:
            self._runner.reset()  # Clears observation history and all old action_buffer entries.
            self._runner.set_instruction(self._instruction)

    def _run(self, native):
        import torch
        processed = self._preprocess(native, target_size=self._runner.runtime.video_size)
        chunk_seed = (self._seed + self._chunk_index * 104729) % (2**63-1)
        with torch.random.fork_rng(devices=[self._device_index]), torch.inference_mode():
            torch.random.default_generator.manual_seed(chunk_seed)
            torch.cuda.default_generators[self._device_index].manual_seed(chunk_seed)
            actions = self._runner.step(processed)
        return actions

    def predict(self, observation, instruction=None):
        native = native_observation(observation)
        if self._runner is None:
            self._load()
        if instruction is not None and instruction != self._instruction:
            raise ValueError('The episode instruction differs from the configured exact cached text')
        actions = np.asarray(self._run(native), dtype=np.float32)
        if actions.shape != (16,14) or not np.isfinite(actions).all():
            raise ValueError('Official WAM must return finite16x14 absolute targets')
        self._chunk_index += 1
        return actions  # Raw grippers unchanged; materialize explicitly before scoring/executing.

    def close(self):
        if self._runner is not None:
            self._runner.close_prediction_video()
