# Benchmark — JMLR-MLOSS reproducibility

This folder contains the **complete, runnable artefacts** used to generate every benchmark number reported in the *torchsom* JMLR-MLOSS paper:

- [`benchmark.py`](benchmark.py) — non-interactive comparison script (CLI)
- [`benchmark.ipynb`](benchmark.ipynb) — interactive notebook with plots
- [`configs/`](configs/) — YAML configurations describing each dataset / map-size combination

It compares `torchsom` against three baselines on `scikit-learn`'s synthetic `make_blobs` data, varying sample size (240 / 4 000 / 16 000) and feature count (4 / 50 / 300), with identical SOM hyperparameters: 25×15 grid, PCA initialization, rectangular topology, 100 epochs, Gaussian neighborhood, Euclidean distance. Each configuration is repeated 10× and reported as mean ± std.

| Backend           | Device   | How it is run                                                                             |
| ----------------- | -------- | ----------------------------------------------------------------------------------------- |
| `torchsom`        | CPU, GPU | native PyTorch; `device: cpu` or `device: cuda`                                            |
| `MiniSom`         | CPU      | standard online `train()` — the canonical baseline, **no JIT**                            |
| `MiniSom`-JIT     | CPU      | numba batch-offline `train_batch_offline_fast()` (`use_minisom_jit`)                       |
| `somoclu`         | CPU, GPU | C++/OpenMP (`kerneltype=0`) or C++/CUDA (`kerneltype=1`), selected from `device`           |

Why these baselines: the GPU column needs a genuine GPU peer, so `somoclu` (C++/CUDA) is benchmarked on GPU against `torchsom` (GPU). On CPU, `MiniSom` is reported both in its standard form and with its optional numba JIT, alongside `somoclu` (CPU) and `torchsom` (CPU). MiniSom's JIT path is its **batch-offline** routine, so it differs from standard MiniSom by both numba compilation and the online→batch algorithm; the batch form aligns it with `torchsom`'s batch training.

The reported MiniSom-JIT time **includes** the one-time numba compilation, which is intrinsic to the JIT approach (standard MiniSom incurs no such cost). For finer analysis, `results.yml` also records the per-operation breakdown: `jit_compile_time` (one-time) and `avg_steadystate_train_time` (warm execution), in addition to the headline `avg_train_time` / `avg_total_time` that fold the two together.

## Reproducibility tags

| Tag                  | Use                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `jmlr-submission-v1` | Exact code as submitted to JMLR-MLOSS in October 2025 — reproduces the original Table 2.       |
| `jmlr-revision-v1`   | Code accompanying the first revision — same benchmark scripts, same MiniSom pin.                |
| `jmlr-revision-v2`   | Second revision — adds the `somoclu` (CPU/GPU) and MiniSom-JIT baselines; reproduces Tables 2a (CPU) and 2b (GPU). |

```bash
# Reproduce the original submission's Table 2
git checkout jmlr-submission-v1
# Reproduce the first revision's Table 2 (same numbers; benchmark code unchanged)
git checkout jmlr-revision-v1
# Reproduce the second revision's Tables 2a/2b (somoclu + MiniSom-JIT added)
git checkout jmlr-revision-v2
```

The original MiniSom pin (`65b6ba6` = v2.3.5, 7 April 2025) is identical on `jmlr-submission-v1` and `jmlr-revision-v1`, so that baseline is stable. The second revision additionally benchmarks MiniSom's JIT path, which is only available on MiniSom's development branch (see below).

## Local setup

Assumes you have already followed the install instructions in the top-level [README](../README.md), i.e. `uv sync --all-extras`, which creates the project environment in `.venv`.

```bash
# 1) MiniSom — standard baseline.
#    Original pin (authoritative for jmlr-submission-v1 / jmlr-revision-v1):
uv pip install "git+https://github.com/JustGlowing/minisom.git@65b6ba6776f63db4536a85afa34bd7b2c6555960"

# 2) MiniSom-JIT — requires the development branch (the JIT method
#    `train_batch_offline_fast` is not yet in a released version) plus numba.
#    This pin is used for jmlr-revision-v2 (standard + JIT from one version):
uv pip install "git+https://github.com/JustGlowing/minisom.git@2eb8bd75596c57dd6260d4f0feef839bc0d22d07" numba

# 3) somoclu — the C++/CUDA baseline.
#    Linux (CPU + GPU): pip provides a working build; for GPU it must be built with CUDA.
uv pip install somoclu
#    macOS note: the default pip install ships WITHOUT a compiled binary
#    ("the binary library cannot be imported"), so somoclu cannot train locally;
#    run the somoclu sweep on the Linux/GPU machine instead.

# 4) Run a sweep (CLI) or open the notebook
uv run python benchmark.py --config-path configs/benchmark.yaml
```

### CPU vs GPU runs

A single `device` switch in the config drives both `torchsom`'s device and somoclu's kernel:

- `device: cpu` → `torchsom` (CPU), `MiniSom`, `MiniSom`-JIT, `somoclu` (`kerneltype=0`, OpenMP CPU).
- `device: cuda` → `torchsom` (GPU) and `somoclu` (`kerneltype=1`, CUDA GPU). MiniSom is CPU-only by design and is skipped under `cuda`-only runs.

Enable backends per run via the `use_torchsom` / `use_minisom` / `use_minisom_jit` / `use_somoclu` flags. If somoclu's compiled binary is unavailable, its block is skipped with a clear message and the other backends still run.

## Expected runtime

The full sweep (9 configurations × backends × 10 repeats) is dominated by the MiniSom CPU runs on large data. Indicative wall-clock times on the hardware reported in the paper (Intel Xeon Platinum 8370C, 8 cores, 16 GB RAM; NVIDIA Tesla T4 GPU):

| Backend                | Smallest config (240 × 4) | Largest config (16 000 × 300) |
| ---------------------- | ------------------------- | ----------------------------- |
| MiniSom (CPU)          | ~2 s / repeat             | ~32 min / repeat              |
| torchsom (CPU)         | <1 s / repeat             | ~30 s / repeat                |
| torchsom (GPU, T4)     | <1 s / repeat             | ~12 s / repeat                |
