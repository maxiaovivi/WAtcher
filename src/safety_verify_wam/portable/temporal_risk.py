"""Experimental unified readout; legacy constructors/checkpoints stay unchanged."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .contracts import SafetyBatch
from .multidomain import MultiProfilePortableSafetyCore, MultiProfileSafetyConfig
from .sensor_timing import validate_sensor_configuration
from .sequence_timing import action_sequence_tokens


def max_risk_logits(step_logits, post_logits, action_mask):
    """A calibrated max score, not a probability of independent event union."""
    if step_logits.shape[:2] != action_mask.shape or step_logits.shape[-1] != 2:
        raise ValueError("Risk logits and action mask dimensions differ")
    if post_logits.shape != (step_logits.shape[0], 2):
        raise ValueError("One post-chunk risk position is required per sample")
    odds = step_logits[..., 1] - step_logits[..., 0]
    odds = odds.masked_fill(~action_mask, -torch.inf)
    post_odds = post_logits[:, 1] - post_logits[:, 0]
    maximum = torch.cat((odds, post_odds[:, None]), dim=1).amax(dim=1)
    return torch.stack((torch.zeros_like(maximum), maximum), dim=-1)


def spatial_positions(grid, dim, *, device, dtype):
    if dim % 4:
        raise ValueError("Spatial encoding dimension must be divisible by four")
    coords = torch.linspace(-1, 1, grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    scales = torch.exp(torch.arange(dim // 4, device=device, dtype=dtype)
                       * (-math.log(10000.) / max(1, dim // 4 - 1)))
    def encode(x):
        angle = x.flatten()[:, None] * scales[None, :] * math.pi
        return torch.cat((angle.sin(), angle.cos()), dim=-1)
    return torch.cat((encode(yy), encode(xx)), dim=-1)


class TemporalRiskCore(MultiProfilePortableSafetyCore):
    """Shared step/post classifier with optional CNN spatial memory.

    The existing CLS position becomes a post-chunk query. Its input embedding
    and all common weights retain matched initialization. Legacy chunk heads
    stay in the state dict for reproducibility, but receive no updates.
    """

    def __init__(self, config: MultiProfileSafetyConfig, *, spatial_grid=1):
        if spatial_grid not in (1, 4):
            raise ValueError("This pilot fixes the visual grid to one or four")
        super().__init__(config)
        self.spatial_grid = spatial_grid
        for module in (self.chunk_head, self.profile_chunk_heads):
            module.requires_grad_(False)

    def _spatial_features(self, key: str, batch: SafetyBatch):
        adapter = self.profile_adapters[key]
        adapter.validate(batch)
        validate_sensor_configuration(self.config.sensor_timing, batch)
        if batch.video.device != self.cls_token.device:
            raise RuntimeError("Model and batch must be on the same device")
        dtype = self.cls_token.dtype
        video, state, action = (x.to(dtype=dtype) for x in (batch.video, batch.state, batch.action))
        vt, st, at = (x.to(dtype=dtype) for x in
                      (batch.video_timestamps, batch.state_timestamps, batch.action_timestamps))
        b, f, v = video.shape[:3]
        if v > self.config.max_views:
            raise ValueError("Video view count exceeds configured maximum")
        flat = video.reshape(-1, *video.shape[3:])
        feature_map = self.frame_encoder.features(flat)
        patches = F.adaptive_avg_pool2d(feature_map, self.spatial_grid)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.frame_encoder.output(patches)
        coarse = adapter.coarse_visual_features(flat)
        # Keep the matched per-profile visual residual; only the CNN pooling
        # changes relative to the unified-readout configuration.
        patches = patches + adapter.visual_tokens(flat, coarse=coarse)[:, None, :]
        n = self.spatial_grid ** 2
        patches = patches.reshape(b, f, v, n, -1)
        positions = spatial_positions(self.spatial_grid, self.config.model_dim,
                                      device=video.device, dtype=dtype)
        visual = (patches + positions[None, None, None]
                  + self.time_encoder(vt)[:, :, None, None]
                  + self.view_embedding(torch.arange(v, device=video.device))[None, None, :, None]
                  + self.modality_embedding[0] + adapter.profile_embedding)
        state_tokens = (adapter.state_encoder(state) + self.time_encoder(st)
                        + self.modality_embedding[1] + adapter.profile_embedding)
        if self.config.sensor_timing:
            camera_clock, state_clock = self.sensor_time_encoders[key](batch, dtype=dtype)
            visual = visual + camera_clock[:, :, :, None]
            state_tokens = state_tokens + state_clock
        action_tokens = (adapter.action_encoder(action)
                         + adapter.motion_tokens(state, action, st, at, batch)
                         + action_sequence_tokens(self.time_encoder, self.action_sequence_encoder, batch, dtype)
                         + self.modality_embedding[2] + adapter.profile_embedding)
        visual = visual.reshape(b, f * v * n, -1)
        tokens = torch.cat((self.cls_token.expand(b, -1, -1), visual, state_tokens, action_tokens), dim=1)
        mask = torch.cat((torch.zeros(b, 1, device=video.device, dtype=torch.bool),
                          ~batch.video_mask[..., None].expand(b, f, v, n).reshape(b, -1),
                          ~batch.state_mask, ~batch.action_mask), dim=1)
        fused = self.fusion(tokens, src_key_padding_mask=mask)
        start = 1 + f * v * n + state.shape[1]
        post = fused[:, 0] + adapter.chunk_visual_tokens(coarse, batch_size=b, frames=f, views=v)
        return fused[:, start:start + action.shape[1]], post

    def forward(self, profile_key: str, batch: SafetyBatch):
        if profile_key not in self.profile_adapters:
            raise KeyError(profile_key)
        if self.spatial_grid == 1:
            legacy = super().forward(profile_key, batch)
            steps, post = legacy["step_risk_features"], legacy["risk_features"]
        else:
            steps, post = self._spatial_features(profile_key, batch)
        head = self.profile_step_heads[profile_key] if self.config.profile_specific_heads else self.step_head
        step_logits = head(steps).masked_fill(~batch.action_mask[..., None], 0.)
        post_logits = head(post)
        logits = max_risk_logits(step_logits, post_logits, batch.action_mask)
        probabilities = logits.float().softmax(-1)
        step_probabilities = step_logits.float().softmax(-1)
        return {
            "class_logits": logits, "class_probabilities": probabilities,
            "risk_probability": probabilities[:, 1],
            "step_class_logits": step_logits, "step_class_probabilities": step_probabilities,
            "step_risk_probability": step_probabilities[..., 1],
            "post_class_logits": post_logits,
            "post_risk_probability": post_logits.float().softmax(-1)[:, 1],
            "action_mask": batch.action_mask,
            "risk_features": post, "step_risk_features": steps,
        }
