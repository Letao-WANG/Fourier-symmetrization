"""Reusable QPINN ansatz helpers built on the qmps gate API."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations, product

import torch

from .math import to_reference_tensor
from .operation import RX, State, StronglyEntanglingLayers, expval_z


@dataclass(frozen=True)
class QFMConfig:
    """Configuration for the re-uploading QFM ansatz."""

    dim: int
    n_qubits: int
    n_upload_layers: int
    strong_layers_per_block: int
    exp_base: float = 3.0
    encoding_scale: float = 1.0
    output_scale: float = 1.0
    diff_generator_per_layer: bool = False
    acos_eps: float = 1.0e-6
    randomized_b2_samples: int = 6

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        if self.n_qubits < self.dim:
            raise ValueError("n_qubits must be at least dim.")
        if self.n_upload_layers <= 0:
            raise ValueError("n_upload_layers must be positive.")
        if self.strong_layers_per_block <= 0:
            raise ValueError("strong_layers_per_block must be positive.")


def map_circuit_input(
    x: torch.Tensor,
    input_map: str,
    *,
    acos_eps: float = 1.0e-6,
) -> torch.Tensor:
    """Map physical PDE coordinates to circuit inputs."""

    if input_map == "raw":
        return x
    if input_map == "chebyshev":
        return torch.acos(torch.clamp(x, -1.0 + acos_eps, 1.0 - acos_eps))
    raise ValueError(f"Unknown input_map: {input_map}")


def boundary_envelope(x: torch.Tensor) -> torch.Tensor:
    """Cosine envelope that enforces zero Dirichlet boundary values on [-1, 1]^d."""

    return torch.prod(torch.cos(0.5 * torch.pi * x), dim=1)


def _hyperoctahedral_group(
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    signs = torch.tensor(
        list(product([-1.0, 1.0], repeat=dim)),
        dtype=dtype,
        device=device,
    )
    perms = torch.tensor(
        list(permutations(range(dim))),
        dtype=torch.long,
        device=device,
    )
    return signs, perms


def hyperoctahedral_transformed_batch(
    z: torch.Tensor,
    element_indices: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Apply selected signed permutations to a batch of circuit inputs."""

    dim = z.shape[1]
    signs, perms = _hyperoctahedral_group(dim, z.dtype, z.device)
    n_signs = signs.shape[0]
    perm_indices = torch.arange(perms.shape[0], device=z.device).repeat_interleave(n_signs)
    sign_indices = torch.arange(n_signs, device=z.device).repeat(perms.shape[0])

    selected_perms = perms[perm_indices[element_indices]]
    selected_signs = signs[sign_indices[element_indices]]
    permuted = z[:, selected_perms].permute(1, 0, 2)
    transformed = permuted * selected_signs[:, None, :]
    return transformed.reshape(-1, dim), element_indices.shape[0]


def sample_hyperoctahedral_indices(
    dim: int,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample indices from the signed-permutation group."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    group_size = (2**dim)
    for factor in range(2, dim + 1):
        group_size *= factor
    if sample_count <= group_size:
        return torch.randperm(group_size, device=device)[:sample_count]
    return torch.randint(group_size, (sample_count,), device=device)


def group_transformed_batch(
    z: torch.Tensor,
    group: str,
    *,
    sample_count: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Return a stacked batch for exact or randomized group averaging."""

    if group == "none":
        return z, 1

    if group == "hyperoctahedral":
        group_size = (2**z.shape[1])
        for factor in range(2, z.shape[1] + 1):
            group_size *= factor
        indices = torch.arange(group_size, device=z.device)
        return hyperoctahedral_transformed_batch(z, indices)

    if group == "randomized_hyperoctahedral":
        if sample_count is None:
            raise ValueError("sample_count is required for randomized_hyperoctahedral.")
        indices = sample_hyperoctahedral_indices(z.shape[1], sample_count, z.device)
        return hyperoctahedral_transformed_batch(z, indices)

    raise ValueError(f"Unknown group: {group}")


class ReuploadingStrongEntangleQFM(torch.nn.Module):
    """Data re-uploading QFM with optional signed-permutation averaging."""

    def __init__(
        self,
        config: QFMConfig,
        *,
        group: str,
        input_map: str,
        init_scale: float = 0.2,
        hard_boundary: bool = True,
        group_sample_count: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.group = group
        self.input_map = input_map
        self.hard_boundary = hard_boundary
        self.group_sample_count = group_sample_count
        weight_shape = (
            config.n_upload_layers + 1,
            config.strong_layers_per_block,
            config.n_qubits,
            3,
        )
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(weight_shape, dtype=dtype, device=device)
        )

    def encoding_beta(self, upload: int, wire: int) -> float:
        exponent = (
            upload * self.config.n_qubits + wire
            if self.config.diff_generator_per_layer
            else wire
        )
        return self.config.exp_base**exponent

    def map_input(self, x: torch.Tensor) -> torch.Tensor:
        return map_circuit_input(x, self.input_map, acos_eps=self.config.acos_eps)

    def base_circuit(self, z: torch.Tensor) -> torch.Tensor:
        state = State(self.config.n_qubits, batch=z[:, 0], reference=self.weights)
        wires = range(self.config.n_qubits)

        for upload in range(self.config.n_upload_layers):
            StronglyEntanglingLayers(state, self.weights[upload], wires=wires)
            for wire in wires:
                coordinate = wire % self.config.dim
                angle = (
                    self.config.encoding_scale
                    * self.encoding_beta(upload, wire)
                    * z[:, coordinate]
                )
                RX(state, angle, wires=wire)

        StronglyEntanglingLayers(state, self.weights[-1], wires=wires)
        z_expectations = expval_z(state, wires=wires)
        zero_projector_probs = 0.5 * (1.0 + z_expectations)
        return self.config.output_scale * zero_projector_probs.mean(dim=-1)

    def forward_from_circuit_input(self, z: torch.Tensor) -> torch.Tensor:
        sample_count = self.group_sample_count
        if sample_count is None and self.group == "randomized_hyperoctahedral":
            sample_count = self.config.randomized_b2_samples
        transformed_z, group_size = group_transformed_batch(
            z,
            self.group,
            sample_count=sample_count,
        )
        transformed_outputs = self.base_circuit(transformed_z)
        return transformed_outputs.reshape(group_size, z.shape[0]).mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = to_reference_tensor(x, self.weights)
        free_output = self.forward_from_circuit_input(self.map_input(x))
        if self.hard_boundary:
            return boundary_envelope(x) * free_output
        return free_output


def make_qfm_models(
    config: QFMConfig,
    model_specs: Mapping[str, Mapping[str, object]],
    *,
    seed: int | None = None,
    hard_boundary: bool = True,
    init_scale: float = 0.2,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> dict[str, ReuploadingStrongEntangleQFM]:
    """Instantiate models from specs with a shared random initialization."""

    if seed is not None:
        torch.manual_seed(seed)
    reference = ReuploadingStrongEntangleQFM(
        config,
        group="none",
        input_map="raw",
        init_scale=init_scale,
        hard_boundary=hard_boundary,
        dtype=dtype,
        device=device,
    )
    reference_state = deepcopy(reference.state_dict())

    models = {}
    for name, spec in model_specs.items():
        model = ReuploadingStrongEntangleQFM(
            config,
            init_scale=init_scale,
            hard_boundary=hard_boundary,
            dtype=dtype,
            device=device,
            **spec,
        )
        model.load_state_dict(reference_state)
        models[name] = model
    return models


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a Torch module."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
