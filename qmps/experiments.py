"""Small experiment helpers used by the QPINN benchmark notebooks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
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
    "$B_2$-QFM": "$B_2$-QFM",
    "Randomized $B_2$-QCM": "Randomized $B_2$-QCM",
    "Randomized $B_2$-QFM": "Randomized $B_2$-QFM",
    "Cheb-QFM": "QCM",
    "Ordinary QFM": "QFM",
    "Fully connected PINN": "Fully connected PINN",
}

MODEL_COLORS = {
    "Ordinary QFM": "#1f77b4",
    "Cheb-QFM": "#7b2cbf",
    "$B_2$-QCM": "#d62728",
    "$B_2$-QFM": "#d62728",
    "Randomized $B_2$-QCM": "#2ca02c",
    "Randomized $B_2$-QFM": "#2ca02c",
    "Fully connected PINN": "#00A6A6",
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


@dataclass(frozen=True)
class FullyConnectedPINNConfig:
    dim: int = 2
    hidden_layers: tuple[int, ...] = (4, 4)


class FullyConnectedPINN(torch.nn.Module):
    """Small fully connected PINN baseline with optional hard boundary envelope."""

    def __init__(
        self,
        config: FullyConnectedPINNConfig,
        *,
        hard_boundary: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hard_boundary = hard_boundary
        layers = []
        in_features = config.dim
        for out_features in config.hidden_layers:
            layers.append(torch.nn.Linear(in_features, out_features, dtype=dtype, device=device))
            layers.append(torch.nn.Tanh())
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, 1, dtype=dtype, device=device))
        self.net = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        if self.hard_boundary:
            return _zero_dirichlet_envelope(x) * raw
        return raw


def _zero_dirichlet_envelope(x: torch.Tensor) -> torch.Tensor:
    return torch.prod(torch.cos(0.5 * torch.pi * x), dim=1)


def make_fully_connected_pinn_models(
    config: FullyConnectedPINNConfig,
    *,
    seed: int | None = None,
    hard_boundary: bool = True,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> dict[str, FullyConnectedPINN]:
    """Instantiate the matched fully connected PINN baseline."""

    if seed is not None:
        torch.manual_seed(seed)
    return {
        "Fully connected PINN": FullyConnectedPINN(
            config,
            hard_boundary=hard_boundary,
            dtype=dtype,
            device=device,
        )
    }


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
    display_run_index: int | None = None,
    display_n_runs: int | None = None,
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
            shown_run_index = run_index if display_run_index is None else display_run_index
            shown_n_runs = config.n_runs if display_n_runs is None else display_n_runs
            print(
                f"{label_prefix:16s} | {model_name:24s} "
                f"run {shown_run_index + 1:2d}/{shown_n_runs}, step {step:4d}/{config.steps}: "
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
    display_run_index_offset: int = 0,
    display_n_runs: int | None = None,
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
        shown_run_index = display_run_index_offset + run_index
        shown_n_runs = config.n_runs if display_n_runs is None else display_n_runs
        print(
            f"\n=== Run {shown_run_index + 1}/{shown_n_runs} | "
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
                    display_run_index=shown_run_index,
                    display_n_runs=shown_n_runs,
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


def combine_boundary_results(*results: dict, model_names: list[str] | None = None) -> dict:
    """Combine separately trained model families into one plotting/save payload."""

    if not results:
        raise ValueError("At least one result is required.")

    combined = {
        "histories": {},
        "final_residuals": {},
        "final_objectives": {},
        "final_boundaries": {},
        "representative_run_index": {},
        "representative_models": {},
        "config": results[0]["config"],
        "run_metadata": results[0].get("run_metadata", []),
    }
    inferred_model_names = []
    n_runs = len(results[0].get("models_by_run", []))
    combined_models_by_run = [dict() for _ in range(n_runs)]
    combined_state_dicts_by_run = [
        {constraint_label: {} for constraint_label in CONSTRAINT_LABELS}
        for _ in range(n_runs)
    ]

    for result in results:
        inferred_model_names.extend(result["model_names"])
        for key in ("histories", "final_residuals", "final_objectives", "final_boundaries"):
            combined[key].update(result.get(key, {}))
        combined["representative_run_index"].update(result.get("representative_run_index", {}))
        combined["representative_models"].update(result.get("representative_models", {}))

        for run_index, run_models in enumerate(result.get("models_by_run", [])):
            combined_models_by_run[run_index].update(run_models)
        for run_index, state_dicts in enumerate(result.get("model_state_dicts_by_run", [])):
            for constraint_label in CONSTRAINT_LABELS:
                combined_state_dicts_by_run[run_index][constraint_label].update(
                    state_dicts.get(constraint_label, {})
                )

    combined["models_by_run"] = combined_models_by_run
    combined["model_state_dicts_by_run"] = combined_state_dicts_by_run
    combined["model_names"] = model_names or inferred_model_names
    return combined


def combine_run_results(*results: dict) -> dict:
    """Combine separately saved single-run results into one plotting payload."""

    if not results:
        raise ValueError("At least one result is required.")

    model_names = list(results[0]["model_names"])
    histories = {}
    final_residuals = {}
    final_objectives = {}
    final_boundaries = {}
    final_errors = {}
    models_by_run = []
    state_dicts_by_run = []
    run_metadata = []

    for run_index, result in enumerate(results):
        if list(result["model_names"]) != model_names:
            raise ValueError("All run results must use the same model order.")
        for key, values in result["histories"].items():
            histories.setdefault(key, []).extend(values)
        for key, values in result["final_residuals"].items():
            final_residuals.setdefault(key, []).extend(values)
        for key, values in result["final_objectives"].items():
            final_objectives.setdefault(key, []).extend(values)
        for key, values in result["final_boundaries"].items():
            final_boundaries.setdefault(key, []).extend(values)
        for key, values in result.get("final_errors", {}).items():
            final_errors.setdefault(key, []).extend(values)

        if result.get("models_by_run"):
            models_by_run.append(result["models_by_run"][0])
        if result.get("model_state_dicts_by_run"):
            state_dicts_by_run.append(result["model_state_dicts_by_run"][0])

        metadata = dict(result.get("run_metadata", [{}])[0])
        metadata.setdefault("run_index", run_index)
        metadata.setdefault("run_number", metadata["run_index"] + 1)
        metadata.setdefault("run_label", _run_label(metadata["run_index"]))
        run_metadata.append(metadata)

    config = replace(results[0]["config"], n_runs=len(results))
    combined = {
        "histories": histories,
        "final_residuals": final_residuals,
        "final_objectives": final_objectives,
        "final_boundaries": final_boundaries,
        "final_errors": final_errors,
        "model_state_dicts_by_run": state_dicts_by_run,
        "models_by_run": models_by_run,
        "run_metadata": run_metadata,
        "model_names": model_names,
        "config": config,
    }
    _attach_representative_models(combined)
    return combined


def train_and_save_boundary_comparison_runs(
    make_models: ModelFactory,
    make_fc_models: ModelFactory,
    loss_functions: Mapping[str, LossFn],
    model_names: list[str],
    config: TrainingConfig,
    runtime: Runtime,
    dim: int,
    seed: int,
    problem,
    grid_n: int,
    figure_prefix: str,
) -> dict:
    """Train and save one self-contained training-data file after each run."""

    single_run_config = replace(config, n_runs=1)
    start_run_index = next_training_run_index(runtime.data_dir, figure_prefix)
    end_run_index = start_run_index + config.n_runs
    for local_run_index in range(config.n_runs):
        run_index = start_run_index + local_run_index
        run_seed = seed + 1000 * run_index
        qfm_result = train_boundary_comparison(
            make_models=make_models,
            loss_functions=loss_functions,
            model_names=model_names,
            config=single_run_config,
            runtime=runtime,
            dim=dim,
            seed=run_seed,
            display_run_index_offset=run_index,
            display_n_runs=end_run_index,
        )
        fc_result = train_boundary_comparison(
            make_models=make_fc_models,
            loss_functions=loss_functions,
            model_names=["Fully connected PINN"],
            config=single_run_config,
            runtime=runtime,
            dim=dim,
            seed=run_seed,
            display_run_index_offset=run_index,
            display_n_runs=end_run_index,
        )
        run_result = combine_boundary_results(qfm_result, fc_result)
        attach_final_errors(run_result, problem, runtime, dim=dim, grid_n=grid_n)
        path = save_training_data(run_result, runtime.data_dir, figure_prefix, run_index=run_index)
        print(f"saved run data: {path}")

    return load_training_data(
        runtime.data_dir,
        figure_prefix,
        make_models=make_models,
        make_fc_models=make_fc_models,
    )


def _attach_representative_models(result: dict) -> None:
    representative_run_index = {}
    for key in result["histories"]:
        values = result["final_objectives"][key] if key[0] == "Soft constraint" else result["final_residuals"][key]
        representative_run_index[key] = int(np.argmin(values))

    result["representative_run_index"] = representative_run_index
    if result.get("models_by_run"):
        result["representative_models"] = {
            key: result["models_by_run"][run_index][key]
            for key, run_index in representative_run_index.items()
        }
    else:
        result.setdefault("representative_models", {})


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


def final_metric_summary(result: dict) -> dict[tuple[str, str], dict[str, tuple[float, float, int]]]:
    """Return final loss/error mean and sample standard deviation for each model."""

    summary = {}
    final_errors = result.get("final_errors", {})
    for constraint_label in CONSTRAINT_LABELS:
        for model_name in result["model_names"]:
            key = (constraint_label, model_name)
            loss_values = result.get("final_objectives", {}).get(key, [])
            error_values = final_errors.get(key, [])
            summary[key] = {
                "final_loss": _mean_std_count(loss_values),
                "relative_l2_error": _mean_std_count(error_values),
            }
    return summary


def print_final_metric_summary(result: dict) -> None:
    """Print final loss and relative L2 error as mean +/- std over runs."""

    summary = final_metric_summary(result)
    print("\nFinal metrics (mean +/- std over runs)")
    for constraint_label in CONSTRAINT_LABELS:
        print(f"\n{constraint_label}")
        print(f"{'model':28s} {'final loss':>25s} {'relative L2 error':>25s}")
        for model_name in result["model_names"]:
            key = (constraint_label, model_name)
            display_label = MODEL_DISPLAY_LABELS.get(model_name, model_name)
            loss_text = _format_mean_std(summary[key]["final_loss"])
            error_text = _format_mean_std(summary[key]["relative_l2_error"])
            print(f"{display_label:28s} {loss_text:>25s} {error_text:>25s}")


def _mean_std_count(values) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), 0
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), std, int(array.size)


def _format_mean_std(stat: tuple[float, float, int]) -> str:
    mean, std, count = stat
    if count == 0:
        return "n/a"
    return f"{mean:.3e} +/- {std:.3e}"


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


def plot_final_loss_comparison(
    result: dict,
    figure_dir: str | Path,
    figure_prefix: str,
    dpi: int = 300,
    loss_ylim: tuple[float, float] = (1.0e-3, 1.6e2),
    loss_yticks: list[float] | None = None,
) -> Path:
    """Draw the final hard/soft loss comparison in the reference notebook style."""

    import matplotlib.pyplot as plt

    if loss_yticks is None:
        loss_yticks = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2]
    loss_yticklabels = [rf"$10^{{{int(np.round(np.log10(tick)))}}}$" for tick in loss_yticks]
    model_names = list(result["model_names"])
    draw_order = ["Fully connected PINN"] if "Fully connected PINN" in model_names else []
    draw_order.extend(model_name for model_name in model_names if model_name != "Fully connected PINN")
    legend_order = [model_name for model_name in model_names if model_name != "Fully connected PINN"]
    if "Fully connected PINN" in model_names:
        legend_order.append("Fully connected PINN")
    zorder = {
        "Fully connected PINN": 2,
        "Ordinary QFM": 3,
        "Cheb-QFM": 4,
        "Randomized $B_2$-QCM": 5,
        "Randomized $B_2$-QFM": 5,
        "$B_2$-QCM": 6,
        "$B_2$-QFM": 6,
    }

    summaries = {
        "Hard constraint": summarize_history_metric(result["histories"], "residual"),
        "Soft constraint": summarize_history_metric(result["histories"], "loss"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8), sharex=True, sharey=True)
    configs = {
        "Hard constraint": {"ylabel": "Total loss"},
        "Soft constraint": {"ylabel": ""},
    }
    legend_handles = {}
    for ax, constraint_label in zip(axes, CONSTRAINT_LABELS):
        summary = summaries[constraint_label]
        for model_name in draw_order:
            key = (constraint_label, model_name)
            display_label = MODEL_DISPLAY_LABELS.get(model_name, model_name)
            model_zorder = zorder.get(model_name, 3)
            linewidth = 2.8 if model_name == "Fully connected PINN" else 2.6
            if "B_2" in model_name:
                linewidth = 3.0
            mean = summary["geo_mean"][key]
            steps = np.arange(1, len(mean) + 1)
            ax.fill_between(
                steps,
                summary["geo_lower"][key],
                summary["geo_upper"][key],
                color=MODEL_COLORS.get(model_name),
                alpha=0.14,
                linewidth=0,
                zorder=model_zorder - 0.5,
            )
            (line,) = ax.plot(
                steps,
                mean,
                label=display_label,
                color=MODEL_COLORS.get(model_name),
                linewidth=linewidth,
                zorder=model_zorder,
            )
            legend_handles[display_label] = line

        ax.set_yscale("log")
        ax.set_ylim(*loss_ylim)
        ax.set_yticks(loss_yticks)
        ax.set_yticklabels(loss_yticklabels)
        ax.set_title(constraint_label, fontsize=26, pad=12)
        ax.set_xlabel("Training step", fontsize=24, labelpad=10)
        ax.set_ylabel(configs[constraint_label]["ylabel"], fontsize=24, labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=18, width=1.1, length=5)
        ax.tick_params(axis="both", which="minor", width=0.9, length=3)
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.08)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    legend_labels = [MODEL_DISPLAY_LABELS.get(model_name, model_name) for model_name in legend_order]
    fig.legend(
        [legend_handles[label] for label in legend_labels],
        legend_labels,
        frameon=False,
        loc="lower center",
        ncol=len(legend_labels),
        fontsize=21,
        handlelength=2.4,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"{figure_prefix}_loss_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_solution_comparison(
    result: dict,
    problem,
    runtime: Runtime,
    dim: int,
    grid_n: int,
    figure_prefix: str,
    dpi: int = 300,
) -> Path:
    """Draw the final hard/soft solution comparison in the reference notebook style."""

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
    column_names = ["Reference", *result["model_names"]]
    fig, axes = plt.subplots(
        len(CONSTRAINT_LABELS),
        len(column_names),
        figsize=(4.6 * len(column_names), 8.0),
        constrained_layout=True,
        squeeze=False,
    )

    last_image = None
    for row, constraint_label in enumerate(CONSTRAINT_LABELS):
        for col, column_name in enumerate(column_names):
            ax = axes[row, col]
            if column_name == "Reference":
                values = target_grid
                title = "Reference"
            else:
                key = (constraint_label, column_name)
                values = predictions[key]
                display_label = MODEL_DISPLAY_LABELS.get(column_name, column_name)
                title = f"{display_label}\nerror={errors[key]:.2e}"

            last_image = ax.contourf(
                x1_grid,
                x2_grid,
                values,
                levels=56,
                cmap="Blues",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=18, pad=9)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(axis="both", labelsize=13, width=1.0, length=4)
            if row == len(CONSTRAINT_LABELS) - 1:
                ax.set_xlabel("$x_1$", fontsize=16, labelpad=2)
            if col == 0:
                ax.set_ylabel(f"{constraint_label}\n$x_2$", fontsize=18, labelpad=7)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)

    colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
    colorbar.set_label("$u(x_1,x_2)$", fontsize=18, labelpad=8)
    colorbar.ax.tick_params(labelsize=14, width=1.0, length=4)
    runtime.figure_dir.mkdir(parents=True, exist_ok=True)
    path = runtime.figure_dir / f"{figure_prefix}_solution_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def training_data_file(
    data_dir: str | Path,
    figure_prefix: str,
    run_index: int | None = None,
) -> Path:
    run_suffix = f"_{_run_label(run_index)}" if run_index is not None else ""
    return Path(data_dir) / f"{figure_prefix}{run_suffix}_training_data.pt"


def training_data_files(data_dir: str | Path, figure_prefix: str) -> list[Path]:
    return sorted(
        Path(data_dir).glob(f"{figure_prefix}_run_*_training_data.pt"),
        key=lambda path: _training_data_path_run_index(path, figure_prefix),
    )


def available_training_data_files(data_dir: str | Path, figure_prefix: str) -> list[Path]:
    return training_data_files(data_dir, figure_prefix)


def next_training_run_index(data_dir: str | Path, figure_prefix: str) -> int:
    run_indices = [
        _training_data_path_run_index(path, figure_prefix)
        for path in training_data_files(data_dir, figure_prefix)
    ]
    return max(run_indices) + 1 if run_indices else 0


def load_training_data(
    data_dir: str | Path,
    figure_prefix: str,
    make_models: ModelFactory,
    make_fc_models: ModelFactory | None = None,
) -> dict:
    data_dir = Path(data_dir)
    paths = training_data_files(data_dir, figure_prefix)
    if not paths:
        expected_pattern = Path(data_dir) / f"{figure_prefix}_run_###_training_data.pt"
        raise FileNotFoundError(
            f"No single-run training data files found. Expected files like {expected_pattern}."
        )
    return combine_run_results(
        *[
            _load_training_data_file(path, make_models, make_fc_models)
            for path in paths
        ]
    )


def _load_training_data_file(
    path: Path,
    make_models: ModelFactory,
    make_fc_models: ModelFactory | None = None,
) -> dict:
    payload = _torch_load_training_payload(path)
    result = _result_from_training_payload(payload)
    result["representative_models"] = _representative_models_from_payload(
        payload,
        make_models=make_models,
        make_fc_models=make_fc_models,
    )
    if _payload_run_count(payload) == 1:
        result["models_by_run"] = [dict(result["representative_models"])]
    return result


def _torch_load_training_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _result_from_training_payload(payload: dict) -> dict:
    model_names = list(payload["model_names"])
    histories = {}
    final_residuals = {}
    final_objectives = {}
    final_boundaries = {}
    final_errors = {}
    representative_run_index = {}
    run_label_positions = {}
    histories_payload = payload.get("histories", {})
    final_metrics_payload = payload.get("final_metrics", {})

    for constraint_label in CONSTRAINT_LABELS:
        for model_name in model_names:
            key = (constraint_label, model_name)
            run_items = sorted(
                histories_payload.get(constraint_label, {}).get(model_name, {}).items(),
                key=lambda item: _run_label_to_index(item[0]),
            )
            run_histories = [
                {
                    metric_name: [float(value) for value in values]
                    for metric_name, values in history.items()
                }
                for _, history in run_items
            ]
            histories[key] = run_histories
            run_label_positions[key] = {
                run_label: index
                for index, (run_label, _) in enumerate(run_items)
            }
            final_residuals[key] = [_last_metric(history, "residual") for history in run_histories]
            final_objectives[key] = [_last_metric(history, "loss") for history in run_histories]
            final_boundaries[key] = [_last_metric(history, "boundary") for history in run_histories]

            error_values = []
            for run_label, _ in run_items:
                run_metrics = (
                    final_metrics_payload.get(constraint_label, {})
                    .get(model_name, {})
                    .get(run_label, {})
                )
                if "final_relative_l2_error" in run_metrics:
                    error_values.append(float(run_metrics["final_relative_l2_error"]))
            if len(error_values) == len(run_histories):
                final_errors[key] = error_values

    representative_payload = payload.get("representative", {})
    for constraint_label in CONSTRAINT_LABELS:
        for model_name, entry in representative_payload.get(constraint_label, {}).items():
            key = (constraint_label, model_name)
            run_label = _representative_run_label(entry)
            representative_run_index[key] = run_label_positions[key].get(run_label, 0)

    return {
        "histories": histories,
        "final_residuals": final_residuals,
        "final_objectives": final_objectives,
        "final_boundaries": final_boundaries,
        "final_errors": final_errors,
        "model_state_dicts_by_run": _state_dicts_by_run_from_payload(payload),
        "models_by_run": [],
        "run_metadata": list(payload.get("run_metadata", [])),
        "representative_run_index": representative_run_index,
        "representative_models": {},
        "model_names": model_names,
        "config": TrainingConfig(**payload.get("training_config", {})),
    }


def _representative_models_from_payload(
    payload: dict,
    make_models: ModelFactory,
    make_fc_models: ModelFactory | None,
) -> dict:
    models = {}
    representative_payload = payload.get("representative", {})
    run_metadata = list(payload.get("run_metadata", []))
    for constraint_label in CONSTRAINT_LABELS:
        hard_boundary = constraint_label == "Hard constraint"
        for model_name, entry in representative_payload.get(constraint_label, {}).items():
            run_label = _representative_run_label(entry)
            seed = _run_init_seed(run_metadata, run_label)
            factory = make_fc_models if model_name == "Fully connected PINN" else make_models
            if factory is None:
                raise ValueError(f"No model factory was provided for {model_name}.")
            model_group = factory(seed=seed, hard_boundary=hard_boundary)
            if model_name not in model_group:
                raise KeyError(f"{model_name} is not available from the provided model factory.")
            model = model_group[model_name]
            model.load_state_dict(_representative_state_dict(payload, constraint_label, model_name, entry))
            model.eval()
            models[(constraint_label, model_name)] = model
    return models


def _representative_state_dict(
    payload: dict,
    constraint_label: str,
    model_name: str,
    entry: Mapping,
) -> dict[str, torch.Tensor]:
    if "model_state_dict" in entry:
        return entry["model_state_dict"]
    run_label = _representative_run_label(entry)
    return payload["model_state_dicts_by_run"][run_label][constraint_label][model_name]


def _state_dicts_by_run_from_payload(payload: dict) -> list[dict]:
    state_dicts_by_run = payload.get("model_state_dicts_by_run", {})
    return [
        state_dicts
        for _, state_dicts in sorted(
            state_dicts_by_run.items(),
            key=lambda item: _run_label_to_index(item[0]),
        )
    ]


def _representative_run_index(entry: Mapping) -> int:
    if "run_index" in entry:
        return int(entry["run_index"])
    return _run_label_to_index(entry["run_label"])


def _representative_run_label(entry: Mapping) -> str:
    if "run_label" in entry:
        return str(entry["run_label"])
    return _run_label(_representative_run_index(entry))


def _run_init_seed(run_metadata: list[dict], run_label: str) -> int | None:
    for metadata in run_metadata:
        if metadata.get("run_label") == run_label and "init_seed" in metadata:
            return int(metadata["init_seed"])
    return None


def _run_label_to_index(run_label: str) -> int:
    return int(run_label.rsplit("_", 1)[1]) - 1


def _training_data_path_run_index(path: Path, figure_prefix: str) -> int:
    filename = path.name
    prefix = f"{figure_prefix}_"
    suffix = "_training_data.pt"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        return -1
    run_label = filename[len(prefix):-len(suffix)]
    return _run_label_to_index(run_label)


def _payload_run_count(payload: dict) -> int:
    histories_payload = payload.get("histories", {})
    for constraint_payload in histories_payload.values():
        for model_payload in constraint_payload.values():
            return len(model_payload)
    return 0


def save_training_data(
    result: dict,
    data_dir: str | Path,
    figure_prefix: str,
    run_index: int | None = None,
) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    run_index_offset = run_index or 0
    path = training_data_file(data_dir, figure_prefix, run_index=run_index)
    payload = {
        "format_version": 2,
        "figure_prefix": figure_prefix,
        "model_names": list(result["model_names"]),
        "constraint_labels": list(CONSTRAINT_LABELS),
        "training_config": _config_payload(result["config"]),
        "run_metadata": _run_metadata_payload(result, run_index_offset=run_index_offset),
        "histories": _histories_payload(result, run_index_offset=run_index_offset),
        "final_metrics": _final_metrics_payload(result, run_index_offset=run_index_offset),
        "model_state_dicts_by_run": _model_state_dicts_by_run_payload(
            result,
            run_index_offset=run_index_offset,
        ),
        "representative": _representative_payload(result, run_index_offset=run_index_offset),
    }
    torch.save(payload, path)
    return path


def _run_label(run_index: int) -> str:
    return f"run_{run_index + 1:03d}"


def _config_payload(config) -> dict:
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    return dict(config)


def _run_metadata_payload(result: dict, run_index_offset: int = 0) -> list[dict]:
    config = result["config"]
    raw_metadata = result.get("run_metadata", [])
    payload = []
    for run_index in range(config.n_runs):
        absolute_run_index = run_index_offset + run_index
        if run_index < len(raw_metadata):
            metadata = dict(raw_metadata[run_index])
        else:
            metadata = {
                "run_index": absolute_run_index,
                "run_number": absolute_run_index + 1,
            }
        metadata["run_index"] = absolute_run_index
        metadata["run_number"] = absolute_run_index + 1
        metadata["run_label"] = _run_label(absolute_run_index)
        payload.append(metadata)
    return payload


def _histories_payload(
    result: dict,
    run_index_offset: int = 0,
) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    payload = {constraint_label: {} for constraint_label in CONSTRAINT_LABELS}
    for (constraint_label, model_name), histories in result["histories"].items():
        model_payload = {}
        for run_index, history in enumerate(histories):
            model_payload[_run_label(run_index_offset + run_index)] = {
                metric_name: [float(value) for value in values]
                for metric_name, values in history.items()
            }
        payload[constraint_label][model_name] = model_payload
    return payload


def _final_metrics_payload(
    result: dict,
    run_index_offset: int = 0,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    payload = {constraint_label: {} for constraint_label in CONSTRAINT_LABELS}
    final_errors = result.get("final_errors", {})
    for (constraint_label, model_name), histories in result["histories"].items():
        model_payload = {}
        for run_index, history in enumerate(histories):
            absolute_run_index = run_index_offset + run_index
            run_metrics = {
                "run_index": absolute_run_index,
                "run_number": absolute_run_index + 1,
                "final_loss": _last_metric(history, "loss"),
                "final_residual": _last_metric(history, "residual"),
                "final_boundary": _last_metric(history, "boundary"),
                "final_lr": _last_metric(history, "lr"),
            }
            key = (constraint_label, model_name)
            if key in final_errors and run_index < len(final_errors[key]):
                run_metrics["final_relative_l2_error"] = float(final_errors[key][run_index])
            model_payload[_run_label(absolute_run_index)] = run_metrics
        payload[constraint_label][model_name] = model_payload
    return payload


def _model_state_dicts_by_run_payload(
    result: dict,
    run_index_offset: int = 0,
) -> dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]]:
    if "model_state_dicts_by_run" in result:
        state_dicts_by_run = result["model_state_dicts_by_run"]
    else:
        state_dicts_by_run = [
            _state_dicts_by_constraint(run_models)
            for run_models in result.get("models_by_run", [])
        ]
    return {
        _run_label(run_index_offset + run_index): state_dicts
        for run_index, state_dicts in enumerate(state_dicts_by_run)
    }


def _representative_payload(result: dict, run_index_offset: int = 0) -> dict[str, dict[str, dict]]:
    payload = {constraint_label: {} for constraint_label in CONSTRAINT_LABELS}
    final_metrics = _final_metrics_payload(result, run_index_offset=run_index_offset)
    state_dicts_by_run = _model_state_dicts_by_run_payload(
        result,
        run_index_offset=run_index_offset,
    )
    for key, run_index in result["representative_run_index"].items():
        constraint_label, model_name = key
        run_label = _run_label(run_index_offset + run_index)
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
