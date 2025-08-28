"""Benchmark runner for TorchSOM (and optional MiniSom) with Typer CLI.

Usage examples:
  python benchmark/benchmark.py --config-path configs/benchmark.yaml
  python benchmark/benchmark.py --config-path configs/benchmark.yaml --mode azure

On Azure ML, outputs will be written under the AZUREML_OUTPUTS_PATH directory,
typically ./outputs inside the job working directory.
"""

import datetime
import random
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import typer
import yaml
from minisom import MiniSom

from torchsom.core import SOM
from torchsom.visualization import SOMVisualizer, VisualizationConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)


def get_device(
    device_str: str,
) -> torch.device:
    """Return a torch.device, gracefully falling back if CUDA is unavailable."""
    if device_str == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def fix_reproducibility(
    seed: int,
) -> None:
    """Set Python, NumPy, and Torch seeds for reproducibility."""
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    # Torch CPU
    torch.manual_seed(seed)
    # Torch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(
    path: Path,
) -> None:
    """Create a directory path if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def compute_errors(
    som: Union[SOM, MiniSom],
    train_features: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Compute QE and TE on train and test sets."""
    full_train_qe = float(som.quantization_error(data=train_features))
    full_train_te = float(som.topographic_error(data=train_features))
    full_test_qe = float(som.quantization_error(data=test_features))
    full_test_te = float(som.topographic_error(data=test_features))
    return full_train_qe, full_train_te, full_test_qe, full_test_te


def dump_yaml(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Append a document to a YAML file in a deterministic key order."""
    with path.open("a") as f:
        yaml.safe_dump(payload, f, sort_keys=False, explicit_start=True)


@app.command("run")
def run_benchmark(
    config_path: Path = typer.Option(
        Path("configs/benchmark.yaml"),
        exists=True,
        readable=True,
        help="Path to YAML configuration file.",
    ),
    data_path: Optional[Path] = typer.Option(
        None, help="Override dataset CSV path (takes precedence over config)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Override output directory root (ignored on Azure if outputs env is set).",
    ),
) -> None:
    """Run the TorchSOM (and optional MiniSom) benchmark using the provided config."""
    formatted_current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Current time: {formatted_current_time}\n")

    # Load configuration
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    general_params = cfg.get("general", {})
    som_params = cfg.get("som", {})

    # General params
    random_seed = general_params.get("random_seed", 42)
    data_name = general_params.get("data_name", "blobs_300_4.csv")
    n_iter = general_params.get("n_iter", 5)
    mode = general_params.get("mode", "local")
    output_folder = output_dir or general_params.get("output_dir", "results")
    device_name = general_params.get("device", "cpu")
    save_plots = general_params.get("save_plots", True)
    save_format = general_params.get("save_format", "pdf")
    train_ratio = general_params.get("train_ratio", 0.8)
    use_minisom = general_params.get("use_minisom", False)
    use_torchsom = general_params.get("use_torchsom", True)

    # SOM params
    x_size = som_params.get("x_size", 25)
    y_size = som_params.get("y_size", 15)
    sigma = som_params.get("sigma", 1.45)
    learning_rate = som_params.get("learning_rate", 0.95)
    epochs = som_params.get("epochs", 100)
    topology = som_params.get("topology", "rectangular")
    initialization_mode = som_params.get("initialization_mode", "pca")
    neighborhood_order = som_params.get("neighborhood_order", 3)
    verbose = som_params.get("verbose", True)

    # Handle data and results paths
    if mode == "local":
        data_path = Path(f"../data/benchmark/{data_name}.csv")
        results_path = Path(f"{output_folder}/{data_name}/{formatted_current_time}")
    else:
        data_path = Path(f"{output_folder}/data/benchmark/{data_name}.csv")
        results_path = Path(
            f"{output_folder}/results/{data_name}/{formatted_current_time}"
        )

    # Setup experiment
    device = get_device(device_name)
    fix_reproducibility(random_seed)

    typer.echo(f"Loading dataset from: {data_path}\n")
    blobs_df = pd.read_csv(data_path)
    feature_columns = blobs_df.columns[:-1]
    feature_names = feature_columns.to_list()

    # Prepare torch tensors for torchsom
    blobs_torch = torch.tensor(blobs_df.to_numpy(dtype=np.float32), device=device)
    all_features, all_targets = blobs_torch[:, :-1], blobs_torch[:, -1].long()
    shuffled_indices = torch.randperm(len(all_features), device=device)
    all_features, all_targets = (
        all_features[shuffled_indices],
        all_targets[shuffled_indices],
    )
    train_count = int(train_ratio * len(all_features))
    train_features, train_targets = (
        all_features[:train_count],
        all_targets[:train_count],
    )
    test_features, test_targets = (
        all_features[train_count:],
        all_targets[train_count:],
    )
    batch_size = som_params.get("batch_size", train_features.shape[0])

    # Prepare numpy arrays for minisom
    train_features_np, _train_targets_np = (
        train_features.detach().cpu().numpy().astype(np.float32),
        train_targets.detach().cpu().numpy().astype(np.float32),
    )
    test_features_np, _test_targets_np = (
        test_features.detach().cpu().numpy().astype(np.float32),
        test_targets.detach().cpu().numpy().astype(np.float32),
    )
    input_len = train_features_np.shape[1]

    # Prepare run subdirectory
    dataset_name = str(data_path.stem)
    device_dir = device.type
    run_dir = results_path / topology / device_dir
    ensure_dir(run_dir)
    results_yaml = run_dir / "results.yml"

    init_results = {
        "dataset": dataset_name,
        "device": device_dir,
        "n_iter": n_iter,
        "x_size": x_size,
        "y_size": y_size,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "topology": topology,
        "initialization_mode": initialization_mode,
        "results_path": str(results_path),
        "run_dir": str(run_dir),
    }
    dump_yaml(results_yaml, init_results)

    if use_torchsom:
        typer.echo(f"Running TorchSOM benchmark with device: {device}")
        torchsom = SOM(
            x=x_size,
            y=y_size,
            num_features=all_features.shape[1],
            sigma=sigma,
            learning_rate=learning_rate,
            lr_decay_function="asymptotic_decay",
            sigma_decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            topology=topology,
            distance_function="euclidean",
            random_seed=random_seed,
            epochs=epochs,
            initialization_mode=initialization_mode,
            batch_size=batch_size,
            neighborhood_order=neighborhood_order,
            device=device,
        )

        times_init, times_fit = [], []
        train_qe_full, train_te_full = [], []
        test_qe_full, test_te_full = [], []
        for _ in range(n_iter):
            start = time.perf_counter()
            torchsom.initialize_weights(data=train_features, mode=initialization_mode)
            end = time.perf_counter()
            times_init.append(end - start)

            start = time.perf_counter()
            qe_curve, te_curve = torchsom.fit(data=train_features)
            end = time.perf_counter()
            times_fit.append(end - start)

            train_qe, train_te, test_qe, test_te = compute_errors(
                som=torchsom, train_features=train_features, test_features=test_features
            )
            train_qe_full.append(train_qe)
            train_te_full.append(train_te)
            test_qe_full.append(test_qe)
            test_te_full.append(test_te)

        total_fit = [init_t + fit_t for init_t, fit_t in zip(times_init, times_fit)]
        torchsom_results: dict[str, Any] = {
            "torchsom": {
                "avg_init_time": f"{np.mean(times_init):.2f}s",
                "std_init_time": f"{np.std(times_init):.2f}s",
                "avg_train_time": f"{np.mean(times_fit):.2f}s",
                "std_train_time": f"{np.std(times_fit):.2f}s",
                "avg_total_time": f"{np.mean(total_fit):.2f}s",
                "std_total_time": f"{np.std(total_fit):.2f}s",
                "avg_final_full_train_QE": f"{np.mean(train_qe_full):.2f}",
                "std_final_full_train_QE": f"{np.std(train_qe_full):.2f}",
                "avg_final_full_train_TE": f"{np.mean(train_te_full):.2f}",
                "std_final_full_train_TE": f"{np.std(train_te_full):.2f}",
                "avg_final_full_test_QE": f"{np.mean(test_qe_full):.2f}",
                "std_final_full_test_QE": f"{np.std(test_qe_full):.2f}",
                "avg_final_full_test_TE": f"{np.mean(test_te_full):.2f}",
                "std_final_full_test_TE": f"{np.std(test_te_full):.2f}",
                "times_init": times_init,
                "times_fit": times_fit,
                "total_fit": total_fit,
                "train_qe_full": train_qe_full,
                "train_te_full": train_te_full,
                "test_qe_full": test_qe_full,
                "test_te_full": test_te_full,
            }
        }
        dump_yaml(results_yaml, torchsom_results)

        if save_plots:
            typer.echo(f"Generating TorchSOM plots in {save_format} format\n")
            config = VisualizationConfig(save_format=str(save_format))
            visualizer = SOMVisualizer(som=torchsom, config=config)
            bmus_map = torchsom.build_map(
                "bmus_data",
                data=train_features,
                return_indices=True,
                batch_size=train_features.shape[0],
            )
            visualizer.plot_training_errors(
                quantization_errors=qe_curve,
                topographic_errors=te_curve,
                save_path=str(run_dir),
            )
            visualizer.plot_distance_map(
                save_path=str(run_dir),
                distance_metric=torchsom.distance_fn_name,
                neighborhood_order=torchsom.neighborhood_order,
                scaling="sum",
            )
            visualizer.plot_hit_map(
                data=train_features,
                save_path=str(run_dir),
                batch_size=train_features.shape[0],
            )
            visualizer.plot_classification_map(
                data=train_features,
                target=train_targets,
                save_path=str(run_dir),
                neighborhood_order=torchsom.neighborhood_order,
                bmus_data_map=bmus_map,
            )
            visualizer.plot_component_planes(
                component_names=feature_names, save_path=str(run_dir)
            )

    if use_minisom:
        typer.echo("Running MiniSom benchmark (necessarily on CPU)")
        som = MiniSom(
            x=x_size,
            y=y_size,
            input_len=input_len,
            sigma=sigma,
            learning_rate=learning_rate,
            decay_function="asymptotic_decay",
            sigma_decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            topology="rectangular",
            activation_distance="euclidean",
            random_seed=random_seed,
        )

        times_init_m, times_fit_m = [], []
        train_qe_full_m, train_te_full_m = [], []
        test_qe_full_m, test_te_full_m = [], []
        for _ in range(n_iter):
            start = time.perf_counter()
            som.pca_weights_init(data=train_features_np)
            end = time.perf_counter()
            times_init_m.append(end - start)

            start = time.perf_counter()
            som.train(
                data=train_features_np,
                num_iteration=epochs,
                random_order=True,
                verbose=verbose,
                use_epochs=True,
            )
            end = time.perf_counter()
            times_fit_m.append(end - start)

            train_qe_m, train_te_m, test_qe_m, test_te_m = compute_errors(
                som=som,
                train_features=train_features_np,
                test_features=test_features_np,
            )
            train_qe_full_m.append(train_qe_m)
            train_te_full_m.append(train_te_m)
            test_qe_full_m.append(test_qe_m)
            test_te_full_m.append(test_te_m)

        total_fit_m = [i + f for i, f in zip(times_init_m, times_fit_m)]
        minisom_results: dict[str, Any] = {
            "minisom": {
                "avg_init_time": f"{np.mean(times_init_m):.2f}s",
                "std_init_time": f"{np.std(times_init_m):.2f}s",
                "avg_train_time": f"{np.mean(times_fit_m):.2f}s",
                "std_train_time": f"{np.std(times_fit_m):.2f}s",
                "avg_total_time": f"{np.mean(total_fit_m):.2f}s",
                "std_total_time": f"{np.std(total_fit_m):.2f}s",
                "avg_final_full_train_QE": f"{np.mean(train_qe_full_m):.2f}",
                "std_final_full_train_QE": f"{np.std(train_qe_full_m):.2f}",
                "avg_final_full_train_TE": f"{np.mean(train_te_full_m):.2f}",
                "std_final_full_train_TE": f"{np.std(train_te_full_m):.2f}",
                "avg_final_full_test_QE": f"{np.mean(test_qe_full_m):.2f}",
                "std_final_full_test_QE": f"{np.std(test_qe_full_m):.2f}",
                "avg_final_full_test_TE": f"{np.mean(test_te_full_m):.2f}",
                "std_final_full_test_TE": f"{np.std(test_te_full_m):.2f}",
                "times_init": times_init_m,
                "times_fit": times_fit_m,
                "total_fit": total_fit_m,
                "train_qe_full": train_qe_full_m,
                "train_te_full": train_te_full_m,
                "test_qe_full": test_qe_full_m,
                "test_te_full": test_te_full_m,
            }
        }
        dump_yaml(results_yaml, minisom_results)


if __name__ == "__main__":
    app()
