"""PennyLane-like public API for composing Torch circuits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch

from . import statevector
from .math import to_reference_tensor


@dataclass
class QState:
    """Split real/imaginary quantum state used by the qmps gate API."""

    real: torch.Tensor
    imag: torch.Tensor
    n_wires: int
    batch_shape: torch.Size | None = None

    @property
    def device(self) -> torch.device:
        return self.real.device

    @property
    def dtype(self) -> torch.dtype:
        return self.real.dtype

    @property
    def shape(self) -> torch.Size:
        return self.real.shape

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.real, self.imag


@dataclass(frozen=True)
class PauliZObservable:
    wires: int


def State(
    n_wires: int,
    *,
    batch: torch.Tensor | None = None,
    batch_size: int | None = None,
    reference: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> QState:
    """Create a statevector initialized to ``|0...0>``.

    Pass ``batch`` or ``batch_size`` to create a batched statevector for
    parallel Torch/MPS execution.  When ``batch`` is provided, measurements are
    reshaped back to the batch tensor's shape.
    """

    batch_shape = None
    if batch is not None:
        if reference is not None:
            batch_tensor = to_reference_tensor(batch, reference)
        elif torch.is_tensor(batch):
            batch_tensor = batch.to(
                device=device if device is not None else batch.device,
                dtype=dtype if dtype is not None else batch.dtype,
            )
        else:
            batch_tensor = torch.tensor(batch, dtype=dtype, device=device)
        batch_size = batch_tensor.numel()
        batch_shape = batch_tensor.shape
        reference = batch_tensor

    real, imag = statevector.init_zero_state(
        n_wires,
        reference=reference,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    if batch_shape is None and batch_size is not None:
        batch_shape = torch.Size([batch_size])
    return QState(real, imag, n_wires, batch_shape=batch_shape)


ZeroState = State


def _set_state(state: QState, tensors: tuple[torch.Tensor, torch.Tensor]) -> QState:
    state.real, state.imag = tensors
    return state


def _one_wire(wires: int | Sequence[int]) -> int:
    if isinstance(wires, int):
        return wires
    if len(wires) != 1:
        raise ValueError("Expected exactly one wire.")
    return int(wires[0])


def _two_wires(wires: Sequence[int]) -> list[int]:
    if isinstance(wires, int):
        raise ValueError("Expected two wires.")
    if len(wires) != 2:
        raise ValueError("Expected exactly two wires.")
    return [int(wires[0]), int(wires[1])]


def _wire_list(wires: Sequence[int]) -> list[int]:
    if isinstance(wires, int):
        return [wires]
    return [int(wire) for wire in wires]


def _controls(control: int | Sequence[int] | None) -> list[int]:
    if control is None:
        return []
    if isinstance(control, int):
        return [control]
    return [int(wire) for wire in control]


def _aligned(value: torch.Tensor | float, state: QState) -> torch.Tensor:
    tensor = to_reference_tensor(value, state.real)
    if state.batch_shape is not None and tensor.numel() == state.real.shape[0]:
        return tensor.reshape(-1)
    return tensor


def Hadamard(state: QState, *, wires: int | Sequence[int]) -> QState:
    """Apply a Hadamard gate."""

    wire = _one_wire(wires)
    return _set_state(state, statevector.apply_h(state.real, state.imag, wire, state.n_wires))


def PauliX(
    state: QState,
    *,
    wires: int | Sequence[int],
    controls: int | Sequence[int] | None = None,
) -> QState:
    """Apply a PauliX gate, optionally controlled by one or more wires."""

    wire = _one_wire(wires)
    control_wires = _controls(controls)
    return _set_state(
        state,
        statevector.apply_controlled_x(state.real, state.imag, control_wires, wire, state.n_wires),
    )


X = PauliX


def CNOT(state: QState, *, wires: Sequence[int]) -> QState:
    """Apply a CNOT gate."""

    control, target = _two_wires(wires)
    return PauliX(state, wires=target, controls=control)


def RZ(
    state: QState,
    phi: torch.Tensor | float,
    *,
    wires: int | Sequence[int],
    controls: int | Sequence[int] | None = None,
) -> QState:
    """Apply an RZ rotation, optionally controlled by one or more wires."""

    wire = _one_wire(wires)
    phase = _aligned(phi, state)
    control_wires = _controls(controls)
    return _set_state(
        state,
        statevector.apply_controlled_rz(state.real, state.imag, control_wires, wire, phase, state.n_wires),
    )


def RX(
    state: QState,
    phi: torch.Tensor | float,
    *,
    wires: int | Sequence[int],
    controls: int | Sequence[int] | None = None,
) -> QState:
    """Apply an RX rotation, optionally controlled by one or more wires."""

    wire = _one_wire(wires)
    angle = _aligned(phi, state)
    control_wires = _controls(controls)
    return _set_state(
        state,
        statevector.apply_controlled_rx(state.real, state.imag, control_wires, wire, angle, state.n_wires),
    )


def RY(
    state: QState,
    phi: torch.Tensor | float,
    *,
    wires: int | Sequence[int],
    controls: int | Sequence[int] | None = None,
) -> QState:
    """Apply an RY rotation, optionally controlled by one or more wires."""

    wire = _one_wire(wires)
    angle = _aligned(phi, state)
    control_wires = _controls(controls)
    return _set_state(
        state,
        statevector.apply_controlled_ry(state.real, state.imag, control_wires, wire, angle, state.n_wires),
    )


def Rot(
    state: QState,
    phi: torch.Tensor | float,
    theta: torch.Tensor | float,
    omega: torch.Tensor | float,
    *,
    wires: int | Sequence[int],
    controls: int | Sequence[int] | None = None,
) -> QState:
    """Apply PennyLane-style ``Rot(phi, theta, omega)``."""

    wire = _one_wire(wires)
    angles = torch.stack(
        [
            _aligned(phi, state),
            _aligned(theta, state),
            _aligned(omega, state),
        ],
        dim=-1,
    )
    control_wires = _controls(controls)
    return _set_state(
        state,
        statevector.apply_controlled_rot(state.real, state.imag, control_wires, wire, angles, state.n_wires),
    )


def RXFromZ(
    state: QState,
    z: torch.Tensor | float,
    *,
    wires: int | Sequence[int],
    controls: int | Sequence[int] | None = None,
) -> QState:
    """Apply ``RX(-2 * acos(z))``, optionally controlled by one or more wires."""

    wire = _one_wire(wires)
    z = _aligned(z, state)
    control_wires = _controls(controls)
    return _set_state(
        state,
        statevector.apply_controlled_rx_from_z(state.real, state.imag, control_wires, wire, z, state.n_wires),
    )


def QSP(
    state: QState,
    phases: torch.Tensor,
    z: torch.Tensor | float,
    *,
    wires: int | Sequence[int],
    controls: int | Sequence[int] | None = None,
) -> QState:
    """Apply the QSP block ``RZ, (RXFromZ, RZ)*``."""

    wire = _one_wire(wires)
    z = _aligned(z, state)
    control_wires = _controls(controls)
    return _set_state(
        state,
        statevector.apply_qsp_sequence(state.real, state.imag, control_wires, wire, phases, z, state.n_wires),
    )


def IsingZZ(
    state: QState,
    phi: torch.Tensor | float,
    *,
    wires: Sequence[int],
    controls: int | Sequence[int] | None = None,
) -> QState:
    """Apply an IsingZZ rotation, optionally controlled by one or more wires."""

    wire_pair = _two_wires(wires)
    angle = _aligned(phi, state)
    control_wires = _controls(controls)
    return _set_state(
        state,
        statevector.apply_controlled_isingzz(
            state.real,
            state.imag,
            control_wires,
            wire_pair,
            angle,
            state.n_wires,
        ),
    )


def StronglyEntanglingLayers(
    state: QState,
    weights: torch.Tensor,
    *,
    wires: Sequence[int],
    ranges: Sequence[int] | None = None,
) -> QState:
    """Apply PennyLane's StronglyEntanglingLayers template with CNOT entanglers."""

    wire_order = _wire_list(wires)
    n_layers = weights.shape[0]
    if weights.shape[1] != len(wire_order) or weights.shape[2] != 3:
        raise ValueError("weights must have shape (n_layers, n_wires, 3).")
    if ranges is None:
        if len(wire_order) == 1:
            ranges = [0] * n_layers
        else:
            ranges = [(layer % (len(wire_order) - 1)) + 1 for layer in range(n_layers)]
    if len(ranges) != n_layers:
        raise ValueError("ranges must have length n_layers.")

    for layer in range(n_layers):
        for wire_index, wire in enumerate(wire_order):
            angles = weights[layer, wire_index]
            Rot(state, angles[0], angles[1], angles[2], wires=wire)
        if len(wire_order) > 1:
            shift = int(ranges[layer])
            for wire_index, control in enumerate(wire_order):
                target = wire_order[(wire_index + shift) % len(wire_order)]
                CNOT(state, wires=[control, target])
    return state


def ctrl(operation: Callable[..., QState], control: int | Sequence[int]) -> Callable[..., QState]:
    """Return a controlled version of a qmps gate function."""

    control_wires = _controls(control)

    def controlled(state: QState, *args, **kwargs) -> QState:
        existing = _controls(kwargs.pop("controls", None))
        kwargs["controls"] = existing + control_wires
        return operation(state, *args, **kwargs)

    return controlled


def PauliZ(*, wires: int | Sequence[int]) -> PauliZObservable:
    """Create a PauliZ observable."""

    return PauliZObservable(_one_wire(wires))


def expval(observable: PauliZObservable, state: QState) -> torch.Tensor:
    """Return the expectation value of an observable on a state."""

    if isinstance(observable, PauliZObservable):
        return expval_z(state, wires=observable.wires)
    raise TypeError("Unsupported observable.")


def expval_z(state: QState, *, wires: int | Sequence[int]) -> torch.Tensor:
    """Return a PauliZ expectation value."""

    if isinstance(wires, int):
        wire = wires
        value = statevector.expval_z(state.real, state.imag, wire, state.n_wires)
        if state.batch_shape is not None:
            return value.reshape(state.batch_shape)
        return value

    wire_list = _wire_list(wires)
    value = statevector.expval_z_many(state.real, state.imag, wire_list, state.n_wires)
    if state.batch_shape is not None:
        return value.reshape(*state.batch_shape, len(wire_list))
    return value
