"""General full-state Torch operations for small PennyLane-style circuits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .math import device_key


_index_cache: dict[tuple[Any, ...], Any] = {}
_sign_cache: dict[tuple[Any, ...], torch.Tensor] = {}


def clear_caches() -> None:
    """Clear cached full-state index and sign tensors."""

    _index_cache.clear()
    _sign_cache.clear()


def init_zero_state(
    n_wires: int,
    *,
    reference: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a real/imaginary statevector initialized to ``|0...0>``."""

    if reference is not None:
        device = reference.device
        dtype = reference.dtype
    if device is None or dtype is None:
        raise ValueError("Provide either reference or both device and dtype.")
    if batch_size is None:
        state_r = torch.zeros(2**n_wires, dtype=dtype, device=device)
        state_r[0] = 1.0
    else:
        state_r = torch.zeros(batch_size, 2**n_wires, dtype=dtype, device=device)
        state_r[:, 0] = 1.0
    state_i = torch.zeros_like(state_r)
    return state_r, state_i


def wire_mask(wire: int, n_wires: int) -> int:
    """Return the basis-index bit mask for a wire using PennyLane wire order."""

    return 1 << (n_wires - 1 - wire)


def controlled_pair_indices(
    target: int,
    controls: Sequence[int],
    n_wires: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return index pairs whose target bit differs and controls are active."""

    key = ("pairs", n_wires, target, tuple(controls), device_key(device))
    if key not in _index_cache:
        idx0, idx1 = [], []
        target_mask = wire_mask(target, n_wires)
        for idx in range(2**n_wires):
            if idx & target_mask:
                continue
            if all(idx & wire_mask(control, n_wires) for control in controls):
                idx0.append(idx)
                idx1.append(idx | target_mask)
        _index_cache[key] = (
            torch.tensor(idx0, dtype=torch.long, device=device),
            torch.tensor(idx1, dtype=torch.long, device=device),
        )
    return _index_cache[key]


def controlled_ising_indices(
    controls: Sequence[int],
    wires: Sequence[int],
    n_wires: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return controlled IsingZZ indices split by ZZ parity."""

    key = ("ising", n_wires, tuple(controls), tuple(wires), device_key(device))
    if key not in _index_cache:
        same, diff = [], []
        masks = [wire_mask(wire, n_wires) for wire in wires]
        for idx in range(2**n_wires):
            if not all(idx & wire_mask(control, n_wires) for control in controls):
                continue
            same_parity = bool(idx & masks[0]) == bool(idx & masks[1])
            (same if same_parity else diff).append(idx)
        _index_cache[key] = (
            torch.tensor(same, dtype=torch.long, device=device),
            torch.tensor(diff, dtype=torch.long, device=device),
        )
    return _index_cache[key]


def z_signs(wire: int, n_wires: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return +1/-1 signs for a PauliZ expectation on ``wire``."""

    key = ("zsign", n_wires, wire, device_key(device), dtype)
    if key not in _sign_cache:
        signs = [
            1.0 if not (idx & wire_mask(wire, n_wires)) else -1.0
            for idx in range(2**n_wires)
        ]
        _sign_cache[key] = torch.tensor(signs, dtype=dtype, device=device)
    return _sign_cache[key]


def z_signs_many(
    wires: Sequence[int],
    n_wires: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return stacked +1/-1 signs for PauliZ expectations on ``wires``."""

    key = ("zsigns", n_wires, tuple(wires), device_key(device), dtype)
    if key not in _sign_cache:
        _sign_cache[key] = torch.stack(
            [z_signs(wire, n_wires, device, dtype) for wire in wires],
            dim=0,
        )
    return _sign_cache[key]


def complex_mul(
    real: torch.Tensor,
    imag: torch.Tensor,
    factor_r: torch.Tensor | float,
    factor_i: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply a split real/imaginary tensor by a complex factor."""

    return real * factor_r - imag * factor_i, imag * factor_r + real * factor_i


def _is_batched(state_r: torch.Tensor) -> bool:
    return state_r.dim() == 2


def _take(state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return state[:, indices] if _is_batched(state) else state[indices]


def _put(state: torch.Tensor, indices: torch.Tensor, value: torch.Tensor) -> None:
    if _is_batched(state):
        state[:, indices] = value
    else:
        state[indices] = value


def _parameter(value: torch.Tensor, state_r: torch.Tensor) -> torch.Tensor:
    if _is_batched(state_r) and value.numel() > 1:
        return value.reshape(-1, 1)
    return value.reshape(()) if value.numel() == 1 else value


def apply_h(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    wire: int,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a Hadamard gate."""

    idx0, idx1 = controlled_pair_indices(wire, [], n_wires, state_r.device)
    r0, i0 = _take(state_r, idx0), _take(state_i, idx0)
    r1, i1 = _take(state_r, idx1), _take(state_i, idx1)
    inv_sqrt2 = 2**-0.5
    new_r, new_i = state_r.clone(), state_i.clone()
    _put(new_r, idx0, (r0 + r1) * inv_sqrt2)
    _put(new_i, idx0, (i0 + i1) * inv_sqrt2)
    _put(new_r, idx1, (r0 - r1) * inv_sqrt2)
    _put(new_i, idx1, (i0 - i1) * inv_sqrt2)
    return new_r, new_i


def apply_controlled_x(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    controls: Sequence[int],
    target: int,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an X gate controlled by zero or more wires."""

    idx0, idx1 = controlled_pair_indices(target, controls, n_wires, state_r.device)
    new_r, new_i = state_r.clone(), state_i.clone()
    _put(new_r, idx0, _take(state_r, idx1))
    _put(new_i, idx0, _take(state_i, idx1))
    _put(new_r, idx1, _take(state_r, idx0))
    _put(new_i, idx1, _take(state_i, idx0))
    return new_r, new_i


def apply_x(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    target: int,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an uncontrolled X gate."""

    return apply_controlled_x(state_r, state_i, [], target, n_wires)


def apply_controlled_rz(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    controls: Sequence[int],
    target: int,
    phase: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a controlled RZ rotation."""

    idx0, idx1 = controlled_pair_indices(target, controls, n_wires, state_r.device)
    phase = _parameter(phase, state_r)
    cos_half = torch.cos(phase / 2)
    sin_half = torch.sin(phase / 2)
    new_r, new_i = state_r.clone(), state_i.clone()
    r0, i0 = complex_mul(_take(state_r, idx0), _take(state_i, idx0), cos_half, -sin_half)
    r1, i1 = complex_mul(_take(state_r, idx1), _take(state_i, idx1), cos_half, sin_half)
    _put(new_r, idx0, r0)
    _put(new_i, idx0, i0)
    _put(new_r, idx1, r1)
    _put(new_i, idx1, i1)
    return new_r, new_i


def apply_controlled_rx_from_z(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    controls: Sequence[int],
    target: int,
    z: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``RX(-2 * acos(z))`` controlled by zero or more wires."""

    idx0, idx1 = controlled_pair_indices(target, controls, n_wires, state_r.device)
    z = _parameter(z, state_r)
    angle = -2 * torch.acos(z)
    cos_half = torch.cos(angle / 2)
    sin_half = torch.sin(angle / 2)
    zero = torch.zeros((), dtype=state_r.dtype, device=state_r.device)
    r0, i0 = _take(state_r, idx0), _take(state_i, idx0)
    r1, i1 = _take(state_r, idx1), _take(state_i, idx1)
    off_r1, off_i1 = complex_mul(r1, i1, zero, -sin_half)
    off_r0, off_i0 = complex_mul(r0, i0, zero, -sin_half)
    new_r, new_i = state_r.clone(), state_i.clone()
    _put(new_r, idx0, cos_half * r0 + off_r1)
    _put(new_i, idx0, cos_half * i0 + off_i1)
    _put(new_r, idx1, off_r0 + cos_half * r1)
    _put(new_i, idx1, off_i0 + cos_half * i1)
    return new_r, new_i


def apply_controlled_rx(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    controls: Sequence[int],
    target: int,
    angle: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a controlled RX rotation."""

    idx0, idx1 = controlled_pair_indices(target, controls, n_wires, state_r.device)
    angle = _parameter(angle, state_r)
    cos_half = torch.cos(angle / 2)
    sin_half = torch.sin(angle / 2)
    zero = torch.zeros((), dtype=state_r.dtype, device=state_r.device)
    r0, i0 = _take(state_r, idx0), _take(state_i, idx0)
    r1, i1 = _take(state_r, idx1), _take(state_i, idx1)
    off_r1, off_i1 = complex_mul(r1, i1, zero, -sin_half)
    off_r0, off_i0 = complex_mul(r0, i0, zero, -sin_half)
    new_r, new_i = state_r.clone(), state_i.clone()
    _put(new_r, idx0, cos_half * r0 + off_r1)
    _put(new_i, idx0, cos_half * i0 + off_i1)
    _put(new_r, idx1, off_r0 + cos_half * r1)
    _put(new_i, idx1, off_i0 + cos_half * i1)
    return new_r, new_i


def apply_rx(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    target: int,
    angle: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an uncontrolled RX rotation."""

    return apply_controlled_rx(state_r, state_i, [], target, angle, n_wires)


def apply_controlled_ry(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    controls: Sequence[int],
    target: int,
    angle: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a controlled RY rotation."""

    idx0, idx1 = controlled_pair_indices(target, controls, n_wires, state_r.device)
    angle = _parameter(angle, state_r)
    cos_half = torch.cos(angle / 2)
    sin_half = torch.sin(angle / 2)
    r0, i0 = _take(state_r, idx0), _take(state_i, idx0)
    r1, i1 = _take(state_r, idx1), _take(state_i, idx1)
    new_r, new_i = state_r.clone(), state_i.clone()
    _put(new_r, idx0, cos_half * r0 - sin_half * r1)
    _put(new_i, idx0, cos_half * i0 - sin_half * i1)
    _put(new_r, idx1, sin_half * r0 + cos_half * r1)
    _put(new_i, idx1, sin_half * i0 + cos_half * i1)
    return new_r, new_i


def apply_ry(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    target: int,
    angle: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an uncontrolled RY rotation."""

    return apply_controlled_ry(state_r, state_i, [], target, angle, n_wires)


def apply_controlled_rot(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    controls: Sequence[int],
    target: int,
    angles: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply PennyLane-style Rot(phi, theta, omega)."""

    phi, theta, omega = angles[..., 0], angles[..., 1], angles[..., 2]
    state_r, state_i = apply_controlled_rz(state_r, state_i, controls, target, phi, n_wires)
    state_r, state_i = apply_controlled_ry(state_r, state_i, controls, target, theta, n_wires)
    state_r, state_i = apply_controlled_rz(state_r, state_i, controls, target, omega, n_wires)
    return state_r, state_i


def apply_qsp_sequence(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    controls: Sequence[int],
    target: int,
    phases: torch.Tensor,
    z: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``RZ(phases[0])`` followed by repeated RX-from-z/RZ blocks."""

    state_r, state_i = apply_controlled_rz(state_r, state_i, controls, target, phases[0], n_wires)
    for phase in phases[1:]:
        state_r, state_i = apply_controlled_rx_from_z(
            state_r,
            state_i,
            controls,
            target,
            z,
            n_wires,
        )
        state_r, state_i = apply_controlled_rz(state_r, state_i, controls, target, phase, n_wires)
    return state_r, state_i


def apply_controlled_isingzz(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    controls: Sequence[int],
    wires: Sequence[int],
    theta: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a controlled IsingZZ rotation."""

    idx_same, idx_diff = controlled_ising_indices(controls, wires, n_wires, state_r.device)
    theta = _parameter(theta, state_r)
    cos_half = torch.cos(theta / 2)
    sin_half = torch.sin(theta / 2)
    new_r, new_i = state_r.clone(), state_i.clone()
    r_same, i_same = complex_mul(
        _take(state_r, idx_same),
        _take(state_i, idx_same),
        cos_half,
        -sin_half,
    )
    r_diff, i_diff = complex_mul(
        _take(state_r, idx_diff),
        _take(state_i, idx_diff),
        cos_half,
        sin_half,
    )
    _put(new_r, idx_same, r_same)
    _put(new_i, idx_same, i_same)
    _put(new_r, idx_diff, r_diff)
    _put(new_i, idx_diff, i_diff)
    return new_r, new_i


def apply_isingzz(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    wires: Sequence[int],
    theta: torch.Tensor,
    n_wires: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an uncontrolled IsingZZ rotation."""

    return apply_controlled_isingzz(state_r, state_i, [], wires, theta, n_wires)


def expval_z(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    wire: int,
    n_wires: int,
) -> torch.Tensor:
    """Return the PauliZ expectation value on ``wire``."""

    probs = state_r.square() + state_i.square()
    return (probs * z_signs(wire, n_wires, state_r.device, state_r.dtype)).sum(dim=-1)


def expval_z_many(
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Return PauliZ expectation values on multiple wires."""

    probs = state_r.square() + state_i.square()
    signs = z_signs_many(wires, n_wires, state_r.device, state_r.dtype)
    if _is_batched(probs):
        return probs @ signs.T
    return signs @ probs
