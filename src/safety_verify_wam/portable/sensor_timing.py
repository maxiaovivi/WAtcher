"""Per-camera and per-state-component clocks, relative to the decision instant."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from .contracts import SafetyBatch


@dataclass(frozen=True)
class SensorTiming:
    """Sample/arrival seconds; unknown state arrivals have masked zero values.

    Camera arrays are [B,F,V], state arrays [B,T,S]. Sample clocks are required
    for all valid inputs. Arrival knowledge is separate from a sampling proxy.
    """

    video_capture: torch.Tensor
    video_arrival: torch.Tensor
    state_capture: torch.Tensor
    state_arrival: torch.Tensor
    state_arrival_known: torch.Tensor

    @classmethod
    def from_arrays(cls, **values: Any) -> "SensorTiming":
        expected = {field.name for field in fields(cls)}
        if set(values) != expected:
            raise ValueError(f"Sensor timing fields must be exactly {sorted(expected)}")
        if torch.as_tensor(values['state_arrival_known']).dtype != torch.bool:
            raise TypeError("state_arrival_known must be an explicit boolean mask")
        return cls(**{name: torch.as_tensor(value, dtype=torch.bool if name.endswith('_known')
                                           else torch.float64).detach().clone()
                      for name, value in values.items()})

    def detached_copy(self) -> "SensorTiming":
        return SensorTiming(**{field.name: getattr(self, field.name).detach().clone()
                               for field in fields(self)})

    def to(self, device: torch.device | str, *, dtype: torch.dtype | None = None) -> "SensorTiming":
        return SensorTiming(**{field.name: getattr(self, field.name).to(
            device=device, dtype=torch.bool if field.name.endswith('_known') else dtype or getattr(self, field.name).dtype,
            non_blocking=True) for field in fields(self)})

    def validate(self, batch: "SafetyBatch") -> None:
        video_valid = batch.video_mask
        state_valid = batch.state_mask.unsqueeze(-1).expand_as(batch.state)
        for reference, valid, name in ((batch.video_timestamps, video_valid.any(dim=2), 'video'),
                                       (batch.state_timestamps, batch.state_mask, 'state')):
            if reference.shape != valid.shape or not torch.isfinite(reference).all() or bool((reference[valid] > 0).any()):
                raise ValueError(f"{name} observation watermark must be finite, causal and correctly shaped")
            history = reference.masked_fill(~valid, -torch.inf).cummax(dim=1).values
            previous = torch.cat([torch.full_like(history[:, :1], -torch.inf), history[:, :-1]], dim=1)
            if bool(((reference < previous) & valid).any()):
                raise ValueError(f"{name} observation watermark must be nondecreasing")
        if self.state_arrival_known.shape != state_valid.shape or self.state_arrival_known.dtype != torch.bool:
            raise ValueError("state_arrival_known must match the boolean state sensor mask")
        for field in fields(self):
            value = getattr(self, field.name)
            valid = video_valid if field.name.startswith('video_') else state_valid
            if value.shape != valid.shape or value.device != batch.video.device:
                raise ValueError(f"{field.name} must match its sensor shape and batch device")
            if field.name.endswith('_known'):
                if value.dtype != torch.bool or bool((value & ~valid).any()):
                    raise ValueError("state_arrival_known must be a valid boolean sensor mask")
                continue
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{field.name} must contain finite seconds")
            if bool((value[~valid] != 0).any()) or bool((value[valid] > 0).any()):
                raise ValueError(f"{field.name} must be causal with zero padding")
            comparable = valid
            if field.name == 'state_arrival':
                comparable = comparable & self.state_arrival_known
            # Compare across missing observations as well as adjacent entries.
            history = value.masked_fill(~comparable, -torch.inf).cummax(dim=1).values
            previous = torch.cat([torch.full_like(history[:, :1], -torch.inf), history[:, :-1]], dim=1)
            if bool(((value < previous) & comparable).any()):
                raise ValueError(f"{field.name} must be nondecreasing for each sensor")
        if bool(((self.video_capture > self.video_arrival) & video_valid).any()):
            raise ValueError("Camera arrival cannot precede capture")
        known = self.state_arrival_known & state_valid
        if bool(((self.state_capture > self.state_arrival) & known).any()):
            raise ValueError("State arrival cannot precede sampling")
        if bool((self.state_arrival[~self.state_arrival_known] != 0).any()):
            raise ValueError("Unknown state arrival must use a masked zero placeholder")
        if bool(((self.video_arrival > batch.video_timestamps.unsqueeze(-1)) & video_valid).any()):
            raise ValueError("Camera arrived after its observation watermark")
        if bool(((self.state_capture > batch.state_timestamps.unsqueeze(-1)) & state_valid).any()):
            raise ValueError("Component sampled after its state watermark")


def validate_sensor_configuration(enabled: bool, batch: "SafetyBatch") -> None:
    if enabled != (batch.sensor_timing is not None):
        raise ValueError("sensor_timing model option and input clocks must agree; clocks cannot be discarded")
    if batch.sensor_timing is not None:
        batch.sensor_timing.validate(batch)


class SensorTimeEncoder(nn.Module):
    """Keep capture, arrival, latency and availability knowledge distinguishable."""

    def __init__(self, state_dim: int, model_dim: int) -> None:
        super().__init__()
        self.video_encoder = nn.Sequential(nn.Linear(4, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.state_encoder = nn.Sequential(nn.Linear(4 * state_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))

    def forward(self, batch: "SafetyBatch", *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        clocks = batch.sensor_timing
        if clocks is None:
            raise ValueError("SensorTimeEncoder requires explicit clocks")
        video = torch.stack([clocks.video_capture, clocks.video_arrival,
                             clocks.video_arrival - clocks.video_capture,
                             torch.ones_like(clocks.video_capture)], dim=-1)
        known = clocks.state_arrival_known
        latency = torch.where(known, clocks.state_arrival - clocks.state_capture, 0.)
        state = torch.stack([clocks.state_capture, clocks.state_arrival, latency, known.to(clocks.state_capture.dtype)], dim=-1)
        return self.video_encoder(video.to(dtype=dtype)), self.state_encoder(state.flatten(-2).to(dtype=dtype))


def position_target_dt(batch: "SafetyBatch", state_indices: torch.Tensor, action_times: torch.Tensor,
                       state_times: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Return a rate denominator, not an invented duration for coincident clocks.

    A command issued at the observation timestamp has no positive elapsed-time
    evidence for an initial velocity. Infinity masks that derived rate to zero
    while leaving its displacement and recorded timestamp unchanged.
    """
    if batch.sensor_timing is None:
        current = state_times.gather(1, state_indices.unsqueeze(1)).squeeze(1)
        previous = torch.cat([current.unsqueeze(1), action_times[:, :-1]], dim=1)
        delta = (action_times - previous).unsqueeze(-1)
    else:
        captured = batch.sensor_timing.state_capture.to(dtype=action_times.dtype)
        gather = state_indices.view(-1, 1, 1).expand(-1, 1, captured.shape[-1])
        current = captured.gather(1, gather)[..., :action_dim]
        previous = torch.cat([current, action_times[:, :-1, None].expand(-1, -1, action_dim)], dim=1)
        delta = action_times.unsqueeze(-1) - previous
    return torch.where(delta > 0, delta, torch.full_like(delta, torch.inf))
