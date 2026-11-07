"""Separate command order from physical seconds available before execution."""
from __future__ import annotations

import torch
from torch import nn

from .contracts import SafetyBatch


def known_action_times(batch: SafetyBatch) -> torch.Tensor:
    known = batch.action_time_known
    if known is None:
        return torch.ones_like(batch.action_mask)
    if known.dtype != torch.bool or known.shape != batch.action_mask.shape:
        raise ValueError("Invalid action-time availability mask")
    if not torch.isfinite(batch.action_timestamps).all():
        raise ValueError("Nonfinite command time")
    if bool((batch.action_timestamps[~known] != 0).any()):
        raise ValueError("Unknown times must not contain observed future durations")
    return known


def mask_unknown_rates(rates: torch.Tensor, batch: SafetyBatch) -> torch.Tensor:
    if batch.action_time_known is None:
        return rates
    known = known_action_times(batch)
    intervals = known & torch.cat([torch.ones_like(known[:, :1]), known[:, :-1]], dim=1)
    return torch.where(intervals.unsqueeze(-1), rates, torch.zeros_like(rates))


class ActionSequenceEncoding(nn.Module):
    """Ordinal Fourier features plus an explicit known/unknown-time embedding.

    Indices have no unit of seconds. Physical-time features are added only where
    a planned/fixed command time is actually available at the decision instant.
    """

    def __init__(self, frequencies: int, output_dim: int) -> None:
        super().__init__()
        self.register_buffer("index_frequencies", torch.logspace(-2, 0, frequencies))
        self.index_projection = nn.Sequential(
            nn.Linear(1 + 2 * frequencies, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim)
        )
        self.availability_embedding = nn.Embedding(2, output_dim)

    def forward(self, batch: SafetyBatch, time_encoder: nn.Module) -> torch.Tensor:
        known = known_action_times(batch)
        dtype = self.index_projection[0].weight.dtype
        indices = torch.arange(1, batch.action.shape[1] + 1, device=batch.action.device, dtype=dtype)
        angles = indices[:, None] * self.index_frequencies.to(dtype=dtype)
        features = torch.cat([indices[:, None], angles.sin(), angles.cos()], dim=-1)
        ordinal = self.index_projection(features).unsqueeze(0)
        seconds = time_encoder(batch.action_timestamps.to(dtype=dtype))
        seconds = torch.where(known.unsqueeze(-1), seconds, torch.zeros_like(seconds))
        return ordinal + seconds + self.availability_embedding(known.long())


def action_sequence_tokens(time_encoder: nn.Module, sequence_encoder: ActionSequenceEncoding | None,
                           batch: SafetyBatch, dtype: torch.dtype) -> torch.Tensor:
    known = known_action_times(batch)
    if sequence_encoder is None:
        if bool((batch.action_mask & ~known).any()):
            raise ValueError("Legacy time-only model cannot consume unknown-duration commands; train the sequence-enabled model")
        return time_encoder(batch.action_timestamps.to(dtype=dtype))
    return sequence_encoder(batch, time_encoder)
