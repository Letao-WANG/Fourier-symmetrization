"""PDE definitions and autograd losses for the QPINN notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import spsolve


ModeCoefficients = dict[tuple[int, int], float]


def cosine_mode(x: torch.Tensor, mode: int) -> torch.Tensor:
    """Return the Dirichlet cosine mode used by the benchmark sources."""

    return torch.cos(0.5 * mode * torch.pi * x)


def cosine_mode_np(x: np.ndarray, mode: int) -> np.ndarray:
    """Numpy version of ``cosine_mode``."""

    return np.cos(0.5 * mode * np.pi * x)


def laplacian(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``Delta u`` and ``u`` for a scalar model output."""

    x_req = x.detach().clone().requires_grad_(True)
    u = model(x_req)
    grad_u = torch.autograd.grad(u.sum(), x_req, create_graph=True)[0]
    lap = torch.zeros_like(u)
    for dim in range(x.shape[1]):
        second = torch.autograd.grad(grad_u[:, dim].sum(), x_req, create_graph=True)[0][:, dim]
        lap = lap + second
    return lap, u


def laplacian_and_gradient(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``Delta u``, ``u``, and ``nabla u`` for a scalar model output."""

    x_req = x.detach().clone().requires_grad_(True)
    u = model(x_req)
    grad_u = torch.autograd.grad(u.sum(), x_req, create_graph=True)[0]
    lap = torch.zeros_like(u)
    for dim in range(x.shape[1]):
        second = torch.autograd.grad(grad_u[:, dim].sum(), x_req, create_graph=True)[0][:, dim]
        lap = lap + second
    return lap, u, grad_u


def _metrics(
    loss: torch.Tensor,
    residual_loss: torch.Tensor,
    boundary_loss: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": float(loss.detach().cpu()),
        "residual": float(residual_loss.detach().cpu()),
        "boundary": float(boundary_loss.detach().cpu()),
    }


@dataclass(frozen=True)
class ScreenedPoissonProblem:
    """Screened Poisson benchmark on ``[-1, 1]^2``."""

    lambda_: float = 1.0
    source_mode_coefficients: ModeCoefficients = field(
        default_factory=lambda: {
            (1, 1): 1.0,
            (1, 3): 0.3,
            (3, 1): 0.3,
        }
    )

    def source(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for (m1, m2), coefficient in self.source_mode_coefficients.items():
            value = value + coefficient * cosine_mode(x[:, 0], m1) * cosine_mode(x[:, 1], m2)
        return value

    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for (m1, m2), coefficient in self.source_mode_coefficients.items():
            eigenvalue = (0.5 * torch.pi * m1) ** 2 + (0.5 * torch.pi * m2) ** 2
            value = value + (
                coefficient
                * cosine_mode(x[:, 0], m1)
                * cosine_mode(x[:, 1], m2)
                / (self.lambda_ + eigenvalue)
            )
        return value

    def reference_solution(self, points: torch.Tensor, grid_n: int) -> torch.Tensor:
        del grid_n
        return self.analytical_solution(points)

    def loss(
        self,
        model: torch.nn.Module,
        interior_x: torch.Tensor,
        boundary_x: torch.Tensor,
        boundary_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        lap, u_interior = laplacian(model, interior_x)
        residual = -lap + self.lambda_ * u_interior - self.source(interior_x)
        residual_loss = torch.mean(residual**2)
        boundary_loss = torch.mean(model(boundary_x) ** 2)
        loss = residual_loss + boundary_weight * boundary_loss
        return loss, _metrics(loss, residual_loss, boundary_loss)


@dataclass(frozen=True)
class StationaryHamiltonJacobiProblem:
    """Stationary viscous Hamilton-Jacobi benchmark on ``[-1, 1]^2``."""

    alpha0: float = 0.0
    rho: float = 0.15
    grad_power: float = 1.5
    grad_eps: float = 1.0e-12
    reference_newton_tol: float = 1.0e-10
    source_mode_coefficients: ModeCoefficients = field(
        default_factory=lambda: {
            (1, 1): 0.50,
            (1, 3): 0.15,
            (3, 1): 0.15,
            (1, 5): 0.20,
            (5, 1): 0.20,
        }
    )

    def source(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for (m1, m2), coefficient in self.source_mode_coefficients.items():
            value = value + coefficient * cosine_mode(x[:, 0], m1) * cosine_mode(x[:, 1], m2)
        return value

    def hamiltonian(self, x: torch.Tensor, grad_u: torch.Tensor) -> torch.Tensor:
        grad_norm_sq = torch.sum(grad_u**2, dim=1)
        exponent = 0.5 * self.grad_power
        momentum = (grad_norm_sq + self.grad_eps).pow(exponent) - self.grad_eps**exponent
        return self.rho * momentum - self.source(x)

    def residual(
        self,
        x: torch.Tensor,
        grad_u: torch.Tensor,
        lap_u: torch.Tensor,
    ) -> torch.Tensor:
        return -lap_u + self.hamiltonian(x, grad_u) - self.alpha0

    def loss(
        self,
        model: torch.nn.Module,
        interior_x: torch.Tensor,
        boundary_x: torch.Tensor,
        boundary_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        lap, _, grad_u = laplacian_and_gradient(model, interior_x)
        residual = self.residual(interior_x, grad_u, lap)
        residual_loss = torch.mean(residual**2)
        boundary_loss = torch.mean(model(boundary_x) ** 2)
        loss = residual_loss + boundary_weight * boundary_loss
        return loss, _metrics(loss, residual_loss, boundary_loss)

    def reference_solution(self, points: torch.Tensor, grid_n: int) -> torch.Tensor:
        axis, solution_grid, residual_inf = self.solve_reference_grid(grid_n)
        if residual_inf > 1.0e-6:
            print(f"HJ reference solve residual_inf={residual_inf:.3e}")
        target = torch.tensor(
            solution_grid.reshape(-1),
            dtype=points.dtype,
            device=points.device,
        )
        if points.shape[0] != target.numel():
            raise ValueError(
                "HJ reference_solution expects points from the matching evaluation grid."
            )
        del axis
        return target

    def source_grid(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        value = np.zeros_like(x1, dtype=float)
        for (m1, m2), coefficient in self.source_mode_coefficients.items():
            value = value + coefficient * cosine_mode_np(x1, m1) * cosine_mode_np(x2, m2)
        return value

    def solve_reference_grid(self, n: int) -> tuple[np.ndarray, np.ndarray, float]:
        axis = np.linspace(-1.0, 1.0, n, dtype=float)
        h = axis[1] - axis[0]
        x1_grid, x2_grid = np.meshgrid(axis, axis, indexing="ij")
        interior_n = n - 2

        main = 2.0 * np.ones(interior_n) / h**2
        off = -1.0 * np.ones(interior_n - 1) / h**2
        one_dim_neg_laplacian = sparse.diags([off, main, off], [-1, 0, 1], format="csr")
        identity = sparse.eye(interior_n, format="csr")
        neg_laplacian = (
            sparse.kron(identity, one_dim_neg_laplacian, format="csr")
            + sparse.kron(one_dim_neg_laplacian, identity, format="csr")
        )
        source_grid = self.source_grid(x1_grid, x2_grid)
        potential_grid = -source_grid
        rhs = (source_grid + self.alpha0)[1:-1, 1:-1].reshape(-1)
        initial = spsolve(neg_laplacian, rhs)

        def residual(flat_values: np.ndarray) -> np.ndarray:
            values = np.zeros((n, n), dtype=float)
            values[1:-1, 1:-1] = flat_values.reshape(interior_n, interior_n)
            lap = (
                values[:-2, 1:-1]
                + values[2:, 1:-1]
                + values[1:-1, :-2]
                + values[1:-1, 2:]
                - 4.0 * values[1:-1, 1:-1]
            ) / h**2
            grad_x1 = (values[2:, 1:-1] - values[:-2, 1:-1]) / (2.0 * h)
            grad_x2 = (values[1:-1, 2:] - values[1:-1, :-2]) / (2.0 * h)
            grad_norm_sq = grad_x1**2 + grad_x2**2
            exponent = 0.5 * self.grad_power
            momentum = (grad_norm_sq + self.grad_eps) ** exponent - self.grad_eps**exponent
            return (
                -lap
                + self.rho * momentum
                + potential_grid[1:-1, 1:-1]
                - self.alpha0
            ).reshape(-1)

        solution_flat = newton_krylov(
            residual,
            initial,
            f_tol=self.reference_newton_tol,
            maxiter=40,
        )
        residual_inf = float(np.max(np.abs(residual(solution_flat))))
        solution_grid = np.zeros((n, n), dtype=float)
        solution_grid[1:-1, 1:-1] = solution_flat.reshape(interior_n, interior_n)
        return axis, solution_grid, residual_inf
