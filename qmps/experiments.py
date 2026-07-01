"""Small experiment helpers used by the QPINN benchmark notebooks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import sys
import tempfile

import numpy as np
import torch


LossFn = Callable[
    [torch.nn.Module, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, dict[str, float]],
]
ModelFactory = Callable[[int, bool], dict[str, torch.nn.Module]]


MODEL_DISPLAY_LABELS = {
    "$B_2$-QCM": "$B_2$-QCM",
    "Randomized $B_2$-QCM": "Randomized $B_2$-QCM",
    "Ordinary QCM": "Ordinary QCM",
    "Ordinary QFM": "Ordinary QFM",
}

MODEL_COLORS = {
    "Ordinary QFM": "#1f77b4",
    "Ordinary QCM": "#7b2cbf",
    "$B_2$-QCM": "#d62728",
    "Randomized $B_2$-QCM": "#2ca02c",
}

CONSTRAINT_LABELS = ("Hard constraint", "Soft constraint")


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    dtype: torch.dtype
    figure_dir: Path
    data_dir: Path
    is_notebook: bool


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 300
    n_runs: int = 10
    interior_batch: int = 50
    boundary_batch: int = 50
    lr: float = 1.0e-1
    lr_decay: float = 0.99
    lr_min: float = 1.0e-4
    interior_margin: float = 0.96
    grad_clip: float = 5.0


def setup_runtime(
    use_double_precision: bool,
    seed: int,
    figure_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Runtime:
    """Prepare matplotlib cache, select device/dtype, and seed random generators."""

    mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    xdg_cache = Path(tempfile.gettempdir()) / "xdg-cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    is_notebook = "ipykernel" in sys.modules
    import matplotlib

    if not is_notebook:
        matplotlib.use("Agg")

    device = select_accelerated_device(use_double_precision)
    dtype = candidate_dtype(device, use_double_precision)
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if figure_dir is None:
        figure_dir = locate_figure_dir()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        data_dir = locate_training_data_dir()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device={device}, real dtype={dtype}, state representation=real/imag pair")
    return Runtime(
        device=device,
        dtype=dtype,
        figure_dir=figure_dir,
        data_dir=data_dir,
        is_notebook=is_notebook,
    )


def locate_figure_dir() -> Path:
    return Path.cwd().resolve() / "figures"


def locate_training_data_dir() -> Path:
    return Path.cwd().resolve() / "training_data"


def candidate_dtype(candidate: torch.device, use_double_precision: bool) -> torch.dtype:
    if use_double_precision and candidate.type != "mps":
        return torch.float64
    return torch.float32


def autograd_works(candidate: torch.device, use_double_precision: bool) -> bool:
    dtype = candidate_dtype(candidate, use_double_precision)
    try:
        x = torch.linspace(-0.7, 0.7, 8, dtype=dtype, device=candidate, requires_grad=True)
        y = torch.sum(torch.sin(x) ** 2 + 0.3 * torch.cos(3.0 * x))
        grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
        curvature = torch.autograd.grad(grad_y.sum(), x)[0]
        if candidate.type == "cuda":
            torch.cuda.synchronize()
        return bool(torch.isfinite(curvature).all())
    except Exception as exc:
        print(f"Skipping {candidate.type}: autograd check failed ({exc}).")
        return False


def select_accelerated_device(use_double_precision: bool) -> torch.device:
    candidates = []
    if torch.cuda.is_available():
        candidates.append(torch.device("cuda"))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        candidates.append(torch.device("mps"))
    candidates.append(torch.device("cpu"))
    for candidate in candidates:
        if autograd_works(candidate, use_double_precision):
            return candidate
    return torch.device("cpu")


def sample_interior(
    config: TrainingConfig,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    x = 2.0 * torch.rand(config.interior_batch, dim, dtype=dtype, device=device) - 1.0
    return x * config.interior_margin


def sample_boundary(
    config: TrainingConfig,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    x = 2.0 * torch.rand(config.boundary_batch, dim, dtype=dtype, device=device) - 1.0
    sides = torch.randint(0, 2 * dim, (config.boundary_batch,), device=device)
    for dim_index in range(dim):
        x[sides == 2 * dim_index, dim_index] = -1.0
        x[sides == 2 * dim_index + 1, dim_index] = 1.0
    return x


def make_training_batches(
    seed: int,
    config: TrainingConfig,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(seed)
    return [
        (
            sample_interior(config, dim, dtype, device),
            sample_boundary(config, dim, dtype, device),
        )
        for _ in range(config.steps)
    ]


def print_parameter_counts(models: Mapping[str, torch.nn.Module]) -> None:
    print("Trainable parameter counts:")
    for model_name, model in models.items():
        count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(f"  {model_name:24s}: {count:d}")


def train_model(
    model_name: str,
    model: torch.nn.Module,
    training_batches: list[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: LossFn,
    config: TrainingConfig,
    run_index: int,
    label_prefix: str,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    min_lr_factor = config.lr_min / config.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(min_lr_factor, config.lr_decay**step),
    )
    history = defaultdict(list)

    for step, (interior_x, boundary_x) in enumerate(training_batches, start=1):
        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_fn(model, interior_x, boundary_x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        metrics["lr"] = current_lr
        for key, value in metrics.items():
            history[key].append(value)

        if run_index == 0 and (step == 1 or step % max(1, config.steps // 5) == 0):
            print(
                f"{label_prefix:16s} | {model_name:24s} "
                f"run {run_index + 1:2d}/{config.n_runs}, step {step:4d}/{config.steps}: "
                f"lr={metrics['lr']:.3e}, loss={metrics['loss']:.3e}, "
                f"residual={metrics['residual']:.3e}, boundary={metrics['boundary']:.3e}"
            )

    return dict(history)


def make_loss_functions(
    hard_loss: LossFn,
    soft_loss: LossFn,
) -> dict[str, LossFn]:
    return {
        "Hard constraint": hard_loss,
        "Soft constraint": soft_loss,
    }


def train_boundary_comparison(
    make_models: ModelFactory,
    loss_functions: Mapping[str, LossFn],
    model_names: list[str],
    config: TrainingConfig,
    runtime: Runtime,
    dim: int,
    seed: int,
) -> dict:
    """Train hard-boundary and soft-penalty versions of each model."""

    histories = {
        (constraint_label, model_name): []
        for constraint_label in CONSTRAINT_LABELS
        for model_name in model_names
    }
    final_residuals = {key: [] for key in histories}
    final_objectives = {key: [] for key in histories}
    final_boundaries = {key: [] for key in histories}
    models_by_run = []
    model_state_dicts_by_run = []
    run_metadata = []

    for run_index in range(config.n_runs):
        init_seed = seed + 1000 * run_index
        batch_seed = seed + 1000 * run_index + 137
        run_metadata.append(
            {
                "run_index": run_index,
                "run_number": run_index + 1,
                "run_label": _run_label(run_index),
                "init_seed": init_seed,
                "batch_seed": batch_seed,
            }
        )
        print(
            f"\n=== Run {run_index + 1}/{config.n_runs} | "
            f"init_seed={init_seed}, batch_seed={batch_seed} ==="
        )
        training_batches = make_training_batches(
            batch_seed,
            config,
            dim,
            runtime.dtype,
            runtime.device,
        )
        models_by_constraint = {
            "Hard constraint": make_models(init_seed, True),
            "Soft constraint": make_models(init_seed, False),
        }
        run_models = {}
        for constraint_label in CONSTRAINT_LABELS:
            loss_fn = loss_functions[constraint_label]
            for model_name, model in models_by_constraint[constraint_label].items():
                history = train_model(
                    model_name,
                    model,
                    training_batches,
                    loss_fn,
                    config,
                    run_index,
                    constraint_label,
                )
                key = (constraint_label, model_name)
                histories[key].append(history)
                final_residuals[key].append(history["residual"][-1])
                final_objectives[key].append(history["loss"][-1])
                final_boundaries[key].append(history["boundary"][-1])
                run_models[key] = model
        models_by_run.append(run_models)
        model_state_dicts_by_run.append(_state_dicts_by_constraint(run_models))

    representative_run_index = {}
    for key in histories:
        values = final_objectives[key] if key[0] == "Soft constraint" else final_residuals[key]
        representative_run_index[key] = int(np.argmin(values))

    representative_models = {
        key: models_by_run[run_index][key]
        for key, run_index in representative_run_index.items()
    }
    return {
        "histories": histories,
        "final_residuals": final_residuals,
        "final_objectives": final_objectives,
        "final_boundaries": final_boundaries,
        "model_state_dicts_by_run": model_state_dicts_by_run,
        "models_by_run": models_by_run,
        "run_metadata": run_metadata,
        "representative_run_index": representative_run_index,
        "representative_models": representative_models,
        "model_names": model_names,
        "config": config,
    }


def state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _state_dicts_by_constraint(
    run_models: Mapping[tuple[str, str], torch.nn.Module],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    nested = {constraint_label: {} for constraint_label in CONSTRAINT_LABELS}
    for (constraint_label, model_name), model in run_models.items():
        nested[constraint_label][model_name] = state_dict_to_cpu(model)
    return nested


def summarize_history_metric(histories: Mapping, metric: str, floor: float = 1.0e-300) -> dict:
    runs = {
        key: np.asarray([history[metric] for history in value], dtype=float)
        for key, value in histories.items()
    }
    log_runs = {key: np.log(np.clip(value, floor, None)) for key, value in runs.items()}
    log_mean = {key: value.mean(axis=0) for key, value in log_runs.items()}
    log_variance = {
        key: value.var(axis=0, ddof=1) if value.shape[0] > 1 else np.zeros(value.shape[1])
        for key, value in log_runs.items()
    }
    log_std = {key: np.sqrt(value) for key, value in log_variance.items()}
    return {
        "runs": runs,
        "geo_mean": {key: np.exp(value) for key, value in log_mean.items()},
        "geo_lower": {key: np.exp(log_mean[key] - log_std[key]) for key in runs},
        "geo_upper": {key: np.exp(log_mean[key] + log_std[key]) for key in runs},
    }


def evaluation_grid(
    n: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    if dim != 2:
        raise ValueError("evaluation_grid is currently written for two-dimensional plots.")
    axis = torch.linspace(-1.0, 1.0, n, dtype=dtype, device=device)
    x1, x2 = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)
    return points, x1.detach().cpu().numpy(), x2.detach().cpu().numpy()


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.norm(pred - target).cpu() / torch.linalg.norm(target).cpu())


def attach_final_errors(
    result: dict,
    problem,
    runtime: Runtime,
    dim: int,
    grid_n: int,
) -> dict[tuple[str, str], list[float]]:
    """Evaluate and attach final relative L2 errors for every run and model."""

    points, x1_grid, x2_grid = evaluation_grid(grid_n, dim, runtime.dtype, runtime.device)
    target = problem.reference_solution(points, grid_n).detach()
    result["evaluation_cache"] = {
        "dim": dim,
        "grid_n": grid_n,
        "x1_grid": x1_grid,
        "x2_grid": x2_grid,
        "target_grid": target.reshape(grid_n, grid_n).detach().cpu().numpy(),
    }
    final_errors = {key: [] for key in result["histories"]}
    with torch.no_grad():
        for run_models in result["models_by_run"]:
            for key, model in run_models.items():
                pred = model(points).detach()
                final_errors[key].append(relative_l2_error(pred, target))
    result["final_errors"] = final_errors
    return final_errors


def plot_training_losses(
    result: dict,
    figure_dir: str | Path,
    figure_prefix: str,
    dpi: int = 300,
) -> Path:
    import matplotlib.pyplot as plt

    config = result["config"]
    steps = np.arange(1, config.steps + 1)
    objective_summary = summarize_history_metric(result["histories"], "loss")
    residual_summary = summarize_history_metric(result["histories"], "residual")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    panels = [
        ("Hard constraint", residual_summary, "residual loss"),
        ("Soft constraint", objective_summary, "total loss"),
    ]
    for ax, (constraint_label, summary, ylabel) in zip(axes, panels):
        for model_name in result["model_names"]:
            key = (constraint_label, model_name)
            color = MODEL_COLORS.get(model_name)
            label = MODEL_DISPLAY_LABELS.get(model_name, model_name)
            ax.plot(steps, summary["geo_mean"][key], label=label, color=color, linewidth=2.0)
            ax.fill_between(
                steps,
                summary["geo_lower"][key],
                summary["geo_upper"][key],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.set_title(constraint_label)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"{figure_prefix}_training_losses.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_solution_panels(
    result: dict,
    problem,
    runtime: Runtime,
    dim: int,
    grid_n: int,
    figure_prefix: str,
    dpi: int = 300,
) -> Path:
    import matplotlib.pyplot as plt

    points, x1_grid, x2_grid = evaluation_grid(grid_n, dim, runtime.dtype, runtime.device)
    cache = result.get("evaluation_cache", {})
    target = None
    if cache.get("dim") == dim and cache.get("grid_n") == grid_n:
        x1_grid = cache["x1_grid"]
        x2_grid = cache["x2_grid"]
        target_grid = cache["target_grid"]
    else:
        target = problem.reference_solution(points, grid_n).detach()
        target_grid = target.reshape(grid_n, grid_n).detach().cpu().numpy()

    predictions = {}
    errors = {}
    final_errors = result.get("final_errors", {})
    with torch.no_grad():
        for key, model in result["representative_models"].items():
            pred = model(points).detach()
            predictions[key] = pred.reshape(grid_n, grid_n).cpu().numpy()
            if key in final_errors:
                run_index = result["representative_run_index"][key]
                errors[key] = float(final_errors[key][run_index])
            else:
                if target is None:
                    target = torch.tensor(
                        target_grid.reshape(-1),
                        dtype=pred.dtype,
                        device=pred.device,
                    )
                errors[key] = relative_l2_error(pred, target)

    all_values = [target_grid] + list(predictions.values())
    vmin = min(float(np.min(values)) for values in all_values)
    vmax = max(float(np.max(values)) for values in all_values)

    model_names = result["model_names"]
    fig, axes = plt.subplots(2, len(model_names) + 1, figsize=(3.2 * (len(model_names) + 1), 6.0))
    last_image = None
    for row, constraint_label in enumerate(CONSTRAINT_LABELS):
        image = axes[row, 0].contourf(
            x1_grid,
            x2_grid,
            target_grid,
            levels=40,
            vmin=vmin,
            vmax=vmax,
        )
        last_image = image
        axes[row, 0].set_title("reference")
        axes[row, 0].set_ylabel(f"{constraint_label}\n$x_2$")
        axes[row, 0].set_xlabel("$x_1$")
        for col, model_name in enumerate(model_names, start=1):
            key = (constraint_label, model_name)
            image = axes[row, col].contourf(
                x1_grid,
                x2_grid,
                predictions[key],
                levels=40,
                vmin=vmin,
                vmax=vmax,
            )
            last_image = image
            title = MODEL_DISPLAY_LABELS.get(model_name, model_name)
            axes[row, col].set_title(f"{title}\nrel. $L^2$={errors[key]:.2e}")
            axes[row, col].set_xlabel("$x_1$")
        for ax in axes[row]:
            ax.set_aspect("equal")
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
    fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
    runtime.figure_dir.mkdir(parents=True, exist_ok=True)
    path = runtime.figure_dir / f"{figure_prefix}_solutions.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_training_data(result: dict, data_dir: str | Path, figure_prefix: str) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{figure_prefix}_training_data.pt"
    payload = {
        "format_version": 2,
        "figure_prefix": figure_prefix,
        "model_names": list(result["model_names"]),
        "constraint_labels": list(CONSTRAINT_LABELS),
        "training_config": _config_payload(result["config"]),
        "run_metadata": _run_metadata_payload(result),
        "histories": _histories_payload(result),
        "final_metrics": _final_metrics_payload(result),
        "model_state_dicts_by_run": _model_state_dicts_by_run_payload(result),
        "representative": _representative_payload(result),
    }
    torch.save(payload, path)
    return path


def _run_label(run_index: int) -> str:
    return f"run_{run_index + 1:03d}"


def _config_payload(config) -> dict:
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    return dict(config)


def _run_metadata_payload(result: dict) -> list[dict]:
    config = result["config"]
    raw_metadata = result.get("run_metadata", [])
    payload = []
    for run_index in range(config.n_runs):
        if run_index < len(raw_metadata):
            metadata = dict(raw_metadata[run_index])
        else:
            metadata = {
                "run_index": run_index,
                "run_number": run_index + 1,
            }
        metadata["run_label"] = _run_label(run_index)
        payload.append(metadata)
    return payload


def _histories_payload(result: dict) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    payload = {constraint_label: {} for constraint_label in CONSTRAINT_LABELS}
    for (constraint_label, model_name), histories in result["histories"].items():
        model_payload = {}
        for run_index, history in enumerate(histories):
            model_payload[_run_label(run_index)] = {
                metric_name: [float(value) for value in values]
                for metric_name, values in history.items()
            }
        payload[constraint_label][model_name] = model_payload
    return payload


def _final_metrics_payload(result: dict) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    payload = {constraint_label: {} for constraint_label in CONSTRAINT_LABELS}
    final_errors = result.get("final_errors", {})
    for (constraint_label, model_name), histories in result["histories"].items():
        model_payload = {}
        for run_index, history in enumerate(histories):
            run_metrics = {
                "run_index": run_index,
                "run_number": run_index + 1,
                "final_loss": _last_metric(history, "loss"),
                "final_residual": _last_metric(history, "residual"),
                "final_boundary": _last_metric(history, "boundary"),
                "final_lr": _last_metric(history, "lr"),
            }
            key = (constraint_label, model_name)
            if key in final_errors and run_index < len(final_errors[key]):
                run_metrics["final_relative_l2_error"] = float(final_errors[key][run_index])
            model_payload[_run_label(run_index)] = run_metrics
        payload[constraint_label][model_name] = model_payload
    return payload


def _model_state_dicts_by_run_payload(
    result: dict,
) -> dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]]:
    if "model_state_dicts_by_run" in result:
        state_dicts_by_run = result["model_state_dicts_by_run"]
    else:
        state_dicts_by_run = [
            _state_dicts_by_constraint(run_models)
            for run_models in result.get("models_by_run", [])
        ]
    return {
        _run_label(run_index): state_dicts
        for run_index, state_dicts in enumerate(state_dicts_by_run)
    }


def _representative_payload(result: dict) -> dict[str, dict[str, dict]]:
    payload = {constraint_label: {} for constraint_label in CONSTRAINT_LABELS}
    final_metrics = _final_metrics_payload(result)
    state_dicts_by_run = _model_state_dicts_by_run_payload(result)
    for key, run_index in result["representative_run_index"].items():
        constraint_label, model_name = key
        run_label = _run_label(run_index)
        entry = dict(final_metrics[constraint_label][model_name][run_label])
        entry["run_label"] = run_label
        model = result.get("representative_models", {}).get(key)
        if model is not None:
            entry["model_state_dict"] = state_dict_to_cpu(model)
        else:
            entry["model_state_dict"] = state_dicts_by_run[run_label][constraint_label][model_name]
        payload[constraint_label][model_name] = entry
    return payload


def _last_metric(history: Mapping[str, list[float]], metric_name: str) -> float:
    values = history.get(metric_name, [])
    if not values:
        return float("nan")
    return float(values[-1])
