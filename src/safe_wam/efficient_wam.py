"""Lazy, hardware-free adapter for the audited UR frozen-video A export."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CAMERAS = ('external', 'hand', 'hand2')
FORMAT = 'full_action_frozen_video_v1'


def _json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _path(root, value):
    path = Path(value).expanduser()
    return (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def _stats(value):
    for name in ('state', 'action'):
        lo = np.asarray(value[name]['min'], dtype=np.float64)
        hi = np.asarray(value[name]['max'], dtype=np.float64)
        if lo.shape != (7,) or hi.shape != (7,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError('Expected finite ordered seven-dimensional min/max statistics')
    if value.get('constant_threshold', 1e-6) != 1e-6:
        raise ValueError('This adapter preserves the trained 1e-6 constant threshold')
    return value


def normalize_state(state, stats):
    lo, hi = (np.asarray(stats['state'][k]) for k in ('min', 'max'))
    scale = hi - lo
    return np.where(scale > 1e-6, 2 * (np.asarray(state) - lo) / np.maximum(scale, 1e-6) - 1, 0).astype(np.float32)


def denormalize_actions(actions, stats):
    lo, hi = (np.asarray(stats['action'][k]) for k in ('min', 'max'))
    return ((np.asarray(actions) + 1) * .5 * (hi - lo) + lo).astype(np.float32)


def _letterbox(image, height, width):
    import cv2
    ratio = min(height / image.shape[0], width / image.shape[1])
    h, w = round(image.shape[0] * ratio), round(image.shape[1] * ratio)
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - h) // 2, (width - w) // 2
    output[y:y+h, x:x+w] = resized
    return output


def prepare_mosaic(images, camera_shapes, hand_canvas=False):
    """Original PyAV AREA224 cache, unpad, OpenCV mosaic384x320; RGB throughout."""
    import av
    import cv2
    if set(images) != set(CAMERAS):
        raise ValueError('Exactly external, hand and hand2 cameras are required')
    parts = []
    for index, name in enumerate(CAMERAS):
        image = np.asarray(images[name])
        expected = tuple(camera_shapes[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'{name}: expected native uint8 RGB image')
        if hand_canvas and name in ('hand', 'hand2') and image.shape == (480, 640, 3) and expected == (720, 1280, 3):
            canvas = np.zeros(expected, dtype=np.uint8)
            canvas[:, 160:1120] = cv2.resize(image, (960, 720), interpolation=cv2.INTER_LINEAR)
            image = canvas
        if image.shape != expected:
            raise ValueError(f'{name}: expected audited native shape {expected}, received {image.shape}')
        h, w = image.shape[:2]
        ratio = min(224 / h, 224 / w); nh, nw = round(h * ratio), round(w * ratio)
        resized = av.VideoFrame.from_ndarray(image, format='rgb24').reformat(
            width=nw, height=nh, format='rgb24', interpolation='AREA').to_ndarray()
        cached = np.zeros((224, 224, 3), np.uint8)
        y, x = (224 - nh) // 2, (224 - nw) // 2
        cached[y:y+nh, x:x+nw] = resized
        parts.append(_letterbox(cached[y:y+nh, x:x+nw], 240 if index == 0 else 120, 320 if index == 0 else 160))
    return _letterbox(np.concatenate([parts[0], np.concatenate(parts[1:], axis=1)], axis=0), 384, 320)


def _validate_manifest(payload, config):
    manifest = payload.get('manifest', {})
    if (payload.get('format', FORMAT) != FORMAT or manifest.get('variant') != 'A' or manifest.get('video_base_frozen') is not True
        or manifest.get('video_lora_parameters') != 0 or manifest.get('video_lora_rank') is not None
        or manifest.get('prompt') != config['prompt'] or manifest.get('task') != config['task']
        or payload.get('step') != config['checkpoint_step'] or manifest.get('fps') != 15
        or manifest.get('action_horizon') != 16 or manifest.get('cameras') != list(CAMERAS)):
        raise ValueError('Only the matching UR frozen-video A task export is supported')
    if manifest.get('normalization') != config['statistics']:
        raise ValueError('External and checkpoint normalization differ')


def _restore_action_expert(model, payload, config, torch):
    _validate_manifest(payload, config)
    params = {name: p for name, p in model.named_parameters() if name.startswith('action_expert.')}
    values = payload.get('trainable_state', {})
    if set(params) != set(values) or len(params) != 130 or sum(p.numel() for p in params.values()) != 138987271:
        raise ValueError('Expected all 130 UR action-expert tensors; LoRA/partial exports are unsupported')
    with torch.no_grad():
        for name, parameter in params.items():
            value = values[name]
            if value.shape != parameter.shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError('Action tensor shape/dtype/finite check failed: ' + name)
            parameter.copy_(value)
    model.requires_grad_(False).eval()


@contextmanager
def _allocate_export_only():
    # Allocate compact architecture, then strict-load the existing complete official export.
    # Never load/download the 5B teacher or a text encoder.
    from .vendor.efficient_wam_runtime.models.compact_wan import CompactWANModel
    from .vendor.efficient_wam_runtime.models.wan_model import WanVideoModel, _load_wan_arch_config
    original = CompactWANModel.__dict__['from_teacher_checkpoint']
    def allocate(cls, config, device='cuda'):
        arch = _load_wan_arch_config(config.config_path or config.checkpoint_path)
        arch.update(config.to_wan_model_config())
        return cls(config=config, video_model=WanVideoModel(model_config=arch, vae_path=config.vae_path,
                                                          device=device, precision=config.precision))
    CompactWANModel.from_teacher_checkpoint = classmethod(allocate)
    try:
        yield
    finally:
        CompactWANModel.from_teacher_checkpoint = original


class EfficientWAMPolicy:
    """No assets/model are opened until the first predict; never executes actions."""
    def __init__(self, config_path, device='cuda:0'):
        self.config_path = Path(config_path)
        self.device = str(device)
        self._runner = None

    def _load(self):
        import torch
        from torch import nn
        from .vendor.efficient_wam_runtime import model_loader
        from .vendor.efficient_wam_runtime.runner import EfficientWAMRunner
        from .vendor.efficient_wam_runtime.third_party.wan.modules import attention
        cfg = _json(self.config_path); root = self.config_path.resolve().parent
        required = {'schema', 'format', 'checkpoint', 'checkpoint_step', 'base_checkpoint', 'base_config',
                    'wan_directory', 'text_embeddings', 'normalization', 'camera_shapes', 'prompt', 'task'}
        if not required <= cfg.keys() or cfg['schema'] != 'safe_wam_ur_assets/v1' or cfg['format'] != FORMAT:
            raise ValueError('Expected safe_wam_ur_assets/v1 frozen-video A configuration')
        if not self.device.startswith('cuda:') or not torch.cuda.is_available():
            raise ValueError('Real WAM generation requires an explicitly selected available CUDA device')
        if set(cfg['camera_shapes']) != set(CAMERAS) or any(len(s) != 3 or s[2] != 3 or min(s[:2]) < 1 for s in cfg['camera_shapes'].values()):
            raise ValueError('Declare all three audited native RGB shapes')
        cfg['statistics'] = _stats(_json(_path(root, cfg['normalization'])))
        payload = torch.load(_path(root, cfg['checkpoint']), map_location='cpu', weights_only=True, mmap=True)
        _validate_manifest(payload, cfg)
        base_cfg = copy.deepcopy(_json(_path(root, cfg['base_config'])))
        # Existing historical digests are identity metadata only; no fresh file hashes are computed.
        if payload.get('manifest', {}).get('base_sha256') != base_cfg.get('checkpoint_sha256'):
            raise ValueError('Task export and official base identity metadata differ')
        base_cfg.update(checkpoint_path=str(_path(root, cfg['base_checkpoint'])), wan_path=str(_path(root, cfg['wan_directory'])))
        base_cfg = model_loader.normalize_deploy_config(base_cfg)
        attention.FLASH_ATTN_2_AVAILABLE = attention.FLASH_ATTN_3_AVAILABLE = False
        with _allocate_export_only():
            model = model_loader.build_model_from_config(base_cfg, device=self.device)
        expert = model.action_expert
        for encoder in (expert.input_encoder.state_encoder, expert.input_encoder.action_encoder):
            old = encoder[0]; encoder[0] = nn.Linear(7, old.out_features, device=old.weight.device, dtype=torch.float32)
        old = expert.decoder.action_head[0]
        expert.decoder.action_head[0] = nn.Linear(old.in_features, 7, device=old.weight.device, dtype=torch.float32)
        expert.config.state_dim = expert.config.action_dim = model.config.state_dim = model.config.action_dim = 7
        expert.float()
        _restore_action_expert(model, payload, cfg, torch)
        text = torch.load(_path(root, cfg['text_embeddings']), map_location='cpu', weights_only=True)
        if text.get('prompt') != cfg['prompt'] or text.get('task') != cfg['task']:
            raise ValueError('Cached task text does not match the action expert')
        embeddings = text['embeddings']
        if len(embeddings) != 1 or embeddings[0].ndim != 2 or embeddings[0].shape[1] != 4096 or not bool(torch.isfinite(embeddings[0]).all()):
            raise ValueError('Expected one finite cached UMT5 embedding')
        class IdentityNorm:
            def normalize(self, x): return x
            def denormalize(self, x): return x
        runtime = SimpleNamespace(model=model, device=self.device, num_inference_steps=10, num_video_inference_steps=2,
            video_refresh_steps=(0, 1), chunk_size=16, num_video_frames=8, scene_prefix='', save_predicted_video=False,
            video_stop_cosine_threshold=None, action_skip_cosine_threshold=None, action_normalizer=IdentityNorm())
        self._runner = EfficientWAMRunner(runtime)
        self._runner.cached_text_embeddings = [e.to(self.device, dtype=torch.bfloat16) for e in embeddings]
        self._runner.current_instruction = cfg['prompt']
        self._config = cfg

    def predict(self, observation):
        state = np.asarray(observation.state)
        if state.shape != (7,) or not np.isrealobj(state) or not np.isfinite(state).all() or not 0 <= state[-1] <= 1:
            raise ValueError('Expected six measured joint angles in rad and [0,1] gripper feedback')
        if not observation.frames:
            raise ValueError('A current three-camera frame is required')
        frame = observation.frames[-1]
        if not np.isfinite(frame.timestamp_s) or frame.timestamp_s > 0:
            raise ValueError('Current frame time must be finite and <= decision time')
        if self._runner is None:
            self._load()
        import torch
        mosaic = prepare_mosaic(frame.images, self._config['camera_shapes'], self._config.get('hand_4_3_to_16_9_canvas', False))
        state = normalize_state(state[None], self._config['statistics'])
        pixels = torch.from_numpy(mosaic).permute(2, 0, 1)[None].to(self.device).float() / 255
        state_tensor = torch.from_numpy(state).to(self.device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            prediction = self._runner._sample_action_chunk(pixels, state_tensor)[0].float().cpu().numpy()[0]
        physical = denormalize_actions(prediction, self._config['statistics'])
        if physical.shape != (16, 7) or not np.isfinite(physical).all():
            raise ValueError('WAM did not return finite physical [16,7] speedL candidates')
        # No clipping/binarization: the downstream input contract decides whether the candidate is admissible.
        return physical
