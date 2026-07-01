"""Tensor helpers shared by the Torch circuit operators."""

from __future__ import annotations

from typing import Any

import torch


def device_key(device: torch.device | str) -> tuple[str, int | None]:
    """Return a hashable key for a Torch device."""

    device = torch.device(device)
    return device.type, device.index


def to_reference_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    """Move ``value`` to the same device and dtype as ``reference``."""

    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(value, dtype=reference.dtype, device=reference.device)


def prepare_broadcast_batch(
    t: Any,
    x: Any,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Broadcast two scalar/batched inputs and flatten them for circuit loops."""

    t = to_reference_tensor(t, reference)
    x = to_reference_tensor(x, reference)
    t, x = torch.broadcast_tensors(t, x)
    output_shape = t.shape
    return t.reshape(-1), x.reshape(-1), output_shape


def as_scalar(value: Any, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Convert ``value`` to a scalar tensor aligned with ``reference``."""

    tensor = to_reference_tensor(value, reference)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar.")
    return tensor.reshape(())
