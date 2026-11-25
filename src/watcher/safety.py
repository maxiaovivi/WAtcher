"""Safety head: risk of a candidate chunk from the current camera views, robot state and actions.

`RiskModel` is a compact re-implementation of the original 06 safety core (`TemporalRiskCore` with a
4x4 spatial grid) for one current observation and unknown action durations. Parameter names are
unchanged, so the original `safety-offline.pt` loads with strict=True and gives the same scores
(checked in tests/test_safety.py against outputs recorded from the original implementation).
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .contracts import ACTION_DIM, CAMERAS, HORIZON
from .gate import RiskScores

PROFILE = "robotwin_legacy_target14"
IMAGE_SIZE = (128, 128)
GRID = 4  # image patches per side
CONFIG_DEFAULTS = dict(
    model_dim=128, vision_channels=(32, 64, 128), transformer_layers=3, attention_heads=4,
    feedforward_multiplier=4, dropout=0.1, max_views=8, time_frequencies=8, profile_specific_heads=False,
    coarse_visual_grid=0, chunk_visual_context=False, optional_action_times=False, sensor_timing=False)


def _mlp(width_in, width_out):
    return nn.Sequential(nn.LayerNorm(width_in), nn.Linear(width_in, width_out), nn.GELU(),
                         nn.Linear(width_out, width_out))


def _fourier(values, frequencies):
    angles = values.unsqueeze(-1) * frequencies.to(values.dtype)
    return torch.cat([values.unsqueeze(-1), angles.sin(), angles.cos()], dim=-1)


def spatial_positions(grid, dim, like):
    """Sinusoidal (y, x) encodings of the grid x grid patches: [grid * grid, dim]."""
    coords = torch.linspace(-1, 1, grid, device=like.device, dtype=like.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    scales = torch.exp(torch.arange(dim // 4, device=like.device, dtype=like.dtype)
                       * (-math.log(10000.) / max(1, dim // 4 - 1)))

    def encode(x):
        angle = x.flatten()[:, None] * scales[None, :] * math.pi
        return torch.cat((angle.sin(), angle.cos()), dim=-1)
    return torch.cat((encode(yy), encode(xx)), dim=-1)


class FrameEncoder(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        layers, previous = [], 3
        for width in channels:
            groups = next(g for g in (8, 4, 2, 1) if width % g == 0)
            layers += [nn.Conv2d(previous, width, 5, stride=2, padding=2, bias=False),
                       nn.GroupNorm(groups, width), nn.GELU()]
            previous = width
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(channels[-1], dim)


class TimeEncoding(nn.Module):
    """Fourier features of a time in seconds."""

    def __init__(self, frequencies, dim):
        super().__init__()
        self.register_buffer("frequencies", torch.logspace(-1, 2, frequencies))
        self.projection = nn.Sequential(nn.Linear(1 + 2 * frequencies, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, seconds):
        return self.projection(_fourier(seconds, self.frequencies))


class SequenceEncoding(nn.Module):
    """Action order (1..H) plus an 'action time unknown' embedding."""

    def __init__(self, frequencies, dim):
        super().__init__()
        self.register_buffer("index_frequencies", torch.logspace(-2, 0, frequencies))
        self.index_projection = nn.Sequential(nn.Linear(1 + 2 * frequencies, dim), nn.GELU(), nn.Linear(dim, dim))
        self.availability_embedding = nn.Embedding(2, dim)

    def forward(self, steps, like):
        indices = torch.arange(1, steps + 1, device=like.device, dtype=like.dtype)
        return self.index_projection(_fourier(indices, self.index_frequencies))[None] + self.availability_embedding.weight[0]


class ProfileAdapter(nn.Module):
    """Encoders specific to one robot profile (state/action sizes and motion features)."""

    def __init__(self, profile, dim, coarse_grid, max_views, chunk_context, dropout):
        super().__init__()
        self.motion_mode, self.action_dim = profile["motion_mode"], profile["action_dim"]
        self.coarse_grid, self.max_views = coarse_grid, max_views
        self.state_encoder = _mlp(profile["state_dim"], dim)
        self.action_encoder = _mlp(self.action_dim, dim)
        self.motion_encoder = _mlp(2 * self.action_dim, dim) if self.motion_mode != "none" else None
        self.profile_embedding = nn.Parameter(torch.zeros(dim))
        coarse = 3 * coarse_grid ** 2
        self.visual_encoder = _mlp(coarse, dim) if coarse_grid else None
        self.chunk_visual_encoder = nn.Sequential(
            nn.LayerNorm(4 * max_views * coarse), nn.Linear(4 * max_views * coarse, dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim // 2, dim)) if chunk_context else None

    def motion_tokens(self, state, action):
        # Displacement from the previous target (or the command itself); rates are zero because
        # action durations are unknown.
        if self.motion_mode == "position_target":
            first = action - torch.cat([state[:, -1:, :self.action_dim], action[:, :-1]], dim=1)
        else:
            first = action
        return self.motion_encoder(torch.cat([first, torch.zeros_like(first)], dim=-1))

    def chunk_visual_tokens(self, coarse, batch, views):
        sequence = F.pad(coarse.reshape(batch, 1, views, -1), (0, 0, 0, self.max_views - views))
        current = sequence[:, -1]
        summary = torch.cat([current, sequence.mean(dim=1), sequence[:, -1] - sequence[:, 0],
                             torch.zeros_like(current)], dim=-1)
        return self.chunk_visual_encoder(summary.flatten(1))


class RiskModel(nn.Module):
    """Transformer over 4x4 patch tokens of every view, one state token and 16 action tokens.

    The CLS token reads out the post-chunk risk, each action token its step risk. `config` is the
    checkpoint's `model_config`.
    """

    def __init__(self, config):
        super().__init__()
        unknown = set(config) - set(CONFIG_DEFAULTS) - {"profiles"}
        config = {**CONFIG_DEFAULTS, **config}
        if unknown or config["sensor_timing"] or not config["optional_action_times"]:
            raise ValueError("Unsupported safety model config: needs optional_action_times and no sensor_timing")
        self.config = config
        dim = config["model_dim"]
        self.frame_encoder = FrameEncoder(config["vision_channels"], dim)
        self.time_encoder = TimeEncoding(config["time_frequencies"], dim)
        self.action_sequence_encoder = SequenceEncoding(config["time_frequencies"], dim)
        self.view_embedding = nn.Embedding(config["max_views"], dim)
        self.modality_embedding = nn.Parameter(torch.zeros(3, dim))  # video, state, action
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=config["attention_heads"], dim_feedforward=dim * config["feedforward_multiplier"],
            dropout=config["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=config["transformer_layers"], norm=nn.LayerNorm(dim),
                                            enable_nested_tensor=False)

        def head():  # logits for (safe, risk)
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        self.chunk_head, self.step_head = head(), head()  # chunk_head: unused legacy readout, kept for loading
        keys = [profile["key"] for profile in config["profiles"]]
        self.profile_adapters = nn.ModuleDict({profile["key"]: ProfileAdapter(
            profile, dim, config["coarse_visual_grid"], config["max_views"], config["chunk_visual_context"],
            config["dropout"]) for profile in config["profiles"]})
        specific = config["profile_specific_heads"]
        self.profile_chunk_heads = nn.ModuleDict({key: head() for key in keys} if specific else {})
        self.profile_step_heads = nn.ModuleDict({key: head() for key in keys} if specific else {})

    def forward(self, profile, video, state, action):
        """video [B,V,3,H,W] in [0,1]; normalized state [B,S] and actions [B,16,A].

        Returns step [B,16], post [B] and overall = max(step, post) [B] risk probabilities.
        """
        adapter = self.profile_adapters[profile]
        batch, views = video.shape[:2]
        dim = self.cls_token.shape[-1]
        now = video.new_zeros(batch, 1)  # every input is observed at the decision time
        images = video.flatten(0, 1)
        patches = F.adaptive_avg_pool2d(self.frame_encoder.features(images), GRID).flatten(2).transpose(1, 2)
        patches = self.frame_encoder.output(patches)
        coarse = F.adaptive_avg_pool2d(images, adapter.coarse_grid).flatten(1) if adapter.coarse_grid else None
        if adapter.visual_encoder is not None:
            patches = patches + adapter.visual_encoder(coarse)[:, None, :]
        visual = (patches.reshape(batch, 1, views, GRID * GRID, dim)
                  + spatial_positions(GRID, dim, video)[None, None, None]
                  + self.time_encoder(now)[:, :, None, None]
                  + self.view_embedding(torch.arange(views, device=video.device))[None, None, :, None]
                  + self.modality_embedding[0] + adapter.profile_embedding)
        state = state[:, None]
        state_tokens = (adapter.state_encoder(state) + self.time_encoder(now)
                        + self.modality_embedding[1] + adapter.profile_embedding)
        action_tokens = adapter.action_encoder(action)
        if adapter.motion_encoder is not None:
            action_tokens = action_tokens + adapter.motion_tokens(state, action)
        action_tokens = (action_tokens + self.action_sequence_encoder(action.shape[1], action)
                         + self.modality_embedding[2] + adapter.profile_embedding)
        tokens = torch.cat([self.cls_token.expand(batch, -1, -1), visual.reshape(batch, -1, dim),
                            state_tokens, action_tokens], dim=1)
        padding = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        fused = self.fusion(tokens, src_key_padding_mask=padding)
        steps, post = fused[:, -action.shape[1]:], fused[:, 0]
        if adapter.chunk_visual_encoder is not None:
            post = post + adapter.chunk_visual_tokens(coarse, batch, views)
        head = self.profile_step_heads[profile] if self.config["profile_specific_heads"] else self.step_head
        step_logits, post_logits = head(steps), head(post)
        odds = torch.cat([step_logits[..., 1] - step_logits[..., 0], (post_logits[:, 1] - post_logits[:, 0])[:, None]], 1)
        maximum = odds.amax(dim=1)
        overall = torch.stack([torch.zeros_like(maximum), maximum], dim=-1).float().softmax(-1)[:, 1]
        return step_logits.float().softmax(-1)[..., 1], post_logits.float().softmax(-1)[:, 1], overall


def _normalize(value, mean, std):
    return value if mean is None else (value - mean) / std


class SafetyHead:
    """Load a safety checkpoint and score [16, 14] candidates; risk >= threshold means REJECT."""

    def __init__(self, checkpoint, device="cpu"):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        profile = payload["profiles"][PROFILE]
        if (PROFILE not in payload.get("trained_profiles", ()) or payload.get("action_horizon") != HORIZON
                or (profile["state_dim"], profile["action_dim"]) != (ACTION_DIM, ACTION_DIM)
                or list(profile["camera_names"]) != ["head", "left", "right"]
                or tuple(profile["image_size"]) != IMAGE_SIZE or profile.get("action_timing") != "unknown_duration"):
            raise ValueError(f"{checkpoint} is not a RoboTwin {PROFILE} safety checkpoint")
        self.threshold = float(payload["threshold"])
        if not 0 < self.threshold < 1:
            raise ValueError("Checkpoint threshold must lie in (0, 1)")
        self.status = str(payload.get("status", ""))
        self.device = torch.device(device)
        self.model = RiskModel(payload["model_config"]).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(payload["model"], strict=True)
        stats = {key: None if profile.get(key) is None else torch.tensor(profile[key], dtype=torch.float32)
                 for key in ("state_mean", "state_std", "action_mean", "action_std")}
        self._state_stats = stats["state_mean"], stats["state_std"]
        self._action_stats = stats["action_mean"], stats["action_std"]

    def score(self, images, state, actions):
        """images: {camera: HxWx3 uint8 RGB}; state [14] and actions [16, 14] in physical units."""
        views = [np.asarray(Image.fromarray(images[name]).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR))
                 for name in CAMERAS]
        video = torch.from_numpy(np.stack(views).transpose(0, 3, 1, 2).copy())[None].float() / 255.0
        state = _normalize(torch.tensor(np.asarray(state, np.float32))[None], *self._state_stats)
        action = _normalize(torch.tensor(np.asarray(actions, np.float32))[None], *self._action_stats)
        with torch.inference_mode():
            step, post, overall = self.model(PROFILE, video.to(self.device), state.to(self.device),
                                             action.to(self.device))
        return RiskScores(step[0].cpu().numpy(), float(post[0]), float(overall[0]), self.threshold)


def random_checkpoint(path, *, seed=0, **config):
    """Write a randomly initialized safety head (smoke tests only: its risk scores are meaningless)."""
    config = dict(profiles=[dict(key=PROFILE, state_dim=ACTION_DIM, action_dim=ACTION_DIM, motion_mode="position_target")],
                  optional_action_times=True, **config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RiskModel(config)
    profile = dict(name=PROFILE, state_dim=ACTION_DIM, action_dim=ACTION_DIM, control_dt=None,
                   action_timing="unknown_duration", camera_names=["head", "left", "right"], image_size=list(IMAGE_SIZE))
    torch.save(dict(status="RANDOM_WEIGHTS", trained_profiles=[PROFILE], action_horizon=HORIZON, threshold=0.5,
                    profiles={PROFILE: profile}, model_config=config, model=model.state_dict()), path)
    return Path(path)
