# Benchmark — JMLR-MLOSS reproducibility

This folder contains the **complete, runnable artefacts** used to generate every benchmark number reported in the *torchsom* JMLR-MLOSS paper:

- [`benchmark.py`](benchmark.py) — non-interactive comparison script (CLI)
- [`benchmark.ipynb`](benchmark.ipynb) — interactive notebook with plots
- [`configs/`](configs/) — YAML configurations describing each dataset / map-size combination

It compares `torchsom` against [`MiniSom`](https://github.com/JustGlowing/minisom) on `scikit-learn`'s synthetic `make_blobs` data, varying sample size (240 / 4 000 / 16 000) and feature count (4 / 50 / 300), with identical SOM hyperparameters: 25×15 grid, PCA initialization, rectangular topology, 100 epochs, Gaussian neighborhood, Euclidean distance. Each configuration is repeated 10× and reported as mean ± std.

## Reproducibility tags

| Tag                  | Use                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `jmlr-submission-v1` | Exact code as submitted to JMLR-MLOSS in October 2025 — reproduces the original Table 2.       |
| `jmlr-revision-v1`   | Code accompanying the accepted (revised) version — same benchmark scripts, same MiniSom pin.   |

```bash
# Reproduce the original submission's Table 2
git checkout jmlr-submission-v1
# Reproduce the revised version's Table 2 (same numbers; benchmark code unchanged)
git checkout jmlr-revision-v1
```

The MiniSom pin (`65b6ba6` = v2.3.5, 7 April 2025) is identical on both tags, so the comparison baseline is stable.

## Local setup

Assumes you have already followed the install instructions in the top-level [README](../README.md), i.e. `uv sync --all-extras`, which creates the project environment in `.venv`.

```bash
# 1) Install the exact MiniSom version benchmarked (= v2.3.5).
#    This pin is authoritative and overrides any other MiniSom in your environment.
uv pip install "git+https://github.com/JustGlowing/minisom.git@65b6ba6776f63db4536a85afa34bd7b2c6555960"

# 2) Run the full benchmark (CLI) or open the notebook
uv run python benchmark.py --config configs/benchmark.yaml
```

## Expected runtime

The full sweep (9 configurations × 3 backends × 10 repeats = 270 runs) is dominated by the MiniSom CPU runs on large data. Indicative wall-clock times on the hardware reported in the paper (Intel Xeon Platinum 8370C, 8 cores, 16 GB RAM; NVIDIA Tesla T4 GPU):

| Backend                | Smallest config (240 × 4) | Largest config (16 000 × 300) |
| ---------------------- | ------------------------- | ----------------------------- |
| MiniSom (CPU)          | ~2 s / repeat             | ~32 min / repeat              |
| torchsom (CPU)         | <1 s / repeat             | ~30 s / repeat                |
| torchsom (GPU, T4)     | <1 s / repeat             | ~12 s / repeat                |

End-to-end full-sweep wall-clock: roughly **6–8 hours on CPU-only**, or **~1 hour with a T4 GPU**.

## What success looks like

`benchmark.py` writes per-configuration CSV files (means ± stds for QE, TE, wall-clock time) plus a summary table matching Table 2 of the paper. The notebook additionally renders publication-style figures. Acceptable variation between local runs and Table 2: ≤ 5 % on wall-clock time (hardware-dependent), exact agreement on QE / TE (deterministic given seed control).

## Paper reference

If you use these benchmarks, please cite the paper:

> Berthier, L. *et al.* (2025). *torchsom: The Reference PyTorch Library for Self-Organizing Maps.* JMLR (MLOSS track). [arXiv:2510.11147](https://arxiv.org/abs/2510.11147)

<!-- ## Azure ML setup

```bash
# Install Azure client
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Connect to AZML
az login

# Install azure module
uv pip install azure-ai-ml azure-identity

# Provide keys to the environment
export AZUREML_SUBSCRIPTION="<key>"
export AZUREML_RESOURCE_GROUP="<key>"
export AZUREML_WORKSPACE_NAME="<key>"

# Create env on AZML
python environments/create_environment.py

# Run the raw command from the run_benchmark.yaml to run the job
``` -->
