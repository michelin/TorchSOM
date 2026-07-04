# Benchmark — JMLR-MLOSS reproducibility

This folder contains the **complete, runnable artefacts** used to generate every benchmark number reported in the *torchsom* JMLR-MLOSS paper:

- [`benchmark.py`](benchmark.py) — non-interactive comparison script (CLI)
- [`benchmark.ipynb`](benchmark.ipynb) — interactive notebook with plots
- [`configs/`](configs/) — YAML configurations describing each dataset / map-size combination

It compares `torchsom` against the baselines on `scikit-learn`'s synthetic `make_blobs` data, varying sample size (240 / 4 000 / 16 000), feature count (4 / 50 / 300), map size (25×15 *small* and 90×70 *large*) and topology (rectangular and hexagonal). Shared SOM hyperparameters: PCA initialization, 100 epochs, Gaussian neighborhood, Euclidean distance. Each configuration is repeated 10× and reported as mean ± std.

Topographic error (TE) counts a sample as an error when its 1st and 2nd BMUs are **not grid-adjacent**, using 8-neighborhood adjacency on rectangular grids (grid-distance threshold √2 ≈ 1.42, so diagonal neighbors count as adjacent) and immediate-neighbor adjacency on hexagonal grids, matching the common MiniSom convention so TE is comparable across libraries.

| Backend           | Device   | How it is run                                                                             |
| ----------------- | -------- | ----------------------------------------------------------------------------------------- |
| `torchsom`        | CPU, GPU | native PyTorch; `device: cpu` or `device: cuda`                                            |
| `MiniSom`         | CPU      | standard online `train()` — the canonical baseline, **no JIT**                            |
| `MiniSom`-JIT     | CPU      | numba batch-offline `train_batch_offline_fast()` (`use_minisom_jit`)                       |
| `somoclu`         | GPU      | C++/CUDA (`kerneltype=1`) — **GPU-only baseline**, runs only under `device: cuda`         |

Why these baselines: the GPU column needs a genuine GPU peer, so `somoclu` (C++/CUDA), a massively parallel HPC library, is benchmarked on GPU against `torchsom` (GPU), as requested by the reviewer. On CPU, `torchsom` is compared against `MiniSom` in both its standard form and its optional numba JIT.

`results.yml` reports the steady-state (warm) per-training means, plus each backend's cold-start cost measured separately:

- `avg_total_time`: steady-state train time, averaged over the `n_iter` warm runs (one-time cost excluded).
- `jit_compile_time`: MiniSom-JIT's one-time numba compilation only, not a training run (CPU).
- `device_warmup_time`: GPU only. A full cold first training run, so it includes the one-time CUDA + cuSOLVER setup on top of one train and is larger than `avg_total_time`.

## Reproducibility tags

| Tag                  | Use                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `jmlr-submission-v1` | Exact code as submitted to JMLR-MLOSS in October 2025 — reproduces the original Table 2.       |
| `jmlr-revision-v1`   | Code accompanying the first revision — same benchmark scripts, same MiniSom pin.                |
| `jmlr-revision-v2`   | Second revision — adds the GPU-only `somoclu` and MiniSom-JIT baselines plus the TE fix; reproduces Tables 2 (CPU) and 3 (GPU). |

```bash
# Reproduce the original submission's Table 2
git checkout jmlr-submission-v1
# Reproduce the first revision's Table 2 (same numbers; benchmark code unchanged)
git checkout jmlr-revision-v1
# Reproduce the second revision's Tables 2 and 3 (somoclu + MiniSom-JIT added)
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

# 3) somoclu — the GPU-only C++/CUDA baseline (NOT installed by `uv sync`; build it here).
#    somoclu is benchmarked only on GPU, so build it from source with `nvcc` on PATH so the
#    CUDA kernel is compiled. somoclu does not declare numpy as a build dependency, so it
#    must be built with --no-build-isolation (numpy is already present from `uv sync`); a
#    plain `uv pip install somoclu` fails with "No module named 'numpy'".
#      which nvcc   # must resolve (e.g. run `module load cuda` first)
uv pip install --reinstall --no-binary somoclu --no-build-isolation somoclu
#    Verify the GPU kernel works:
#      python -c "import somoclu,numpy as np; s=somoclu.Somoclu(8,6,kerneltype=1); \
#                 s.train(np.random.rand(64,4).astype('float32'),epochs=2); print('GPU OK')"
#    macOS note: the default pip install ships WITHOUT a compiled binary
#    ("the binary library cannot be imported"), so somoclu cannot train locally;
#    run the somoclu sweep on the Linux/GPU machine instead.

# (optional) FAISS BMU-search backend for torchsom (enable with `use_faiss: true` in the config)
#   CPU index:  uv pip install faiss-cpu
#   GPU index:  uv pip install faiss-gpu   # CUDA; provides index_cpu_to_gpu
#   NOTE: faiss-cpu wheels are unstable on macOS/arm64 (segfault on small indices);
#   use FAISS on the Linux/GPU machine and keep the native backend on macOS.

# 4) Run a sweep (CLI) or open the notebook
uv run python benchmark.py --config-path configs/benchmark.yaml
```

### CPU vs GPU runs

A single `device` switch in the config drives `torchsom`'s device and whether the GPU-only `somoclu` baseline runs:

- `device: cpu` → `torchsom` (CPU), `MiniSom`, `MiniSom`-JIT.
- `device: cuda` → `torchsom` (GPU) and `somoclu` (`kerneltype=1`, CUDA GPU).

Enable backends per run via the `use_torchsom` / `use_minisom` / `use_minisom_jit` / `use_somoclu` flags.

## Reproducing the full paper sweeps

Two driver scripts cover every cell reported in the paper. Each repeats every config 10×
(`n_iter`) and writes one timestamped folder per (topology, map-size) under `results/` and
`logs/`. Datasets depend on map size:

| Topology    | 25×15 (small)  | 90×70 (large)            |
| ----------- | -------------- | ------------------------ |
| rectangular | all 9 datasets | 3 widest (300-feature)   |
| hexagonal   | all 9 datasets | 3 widest (300-feature)   |

- [`run_rectangular.sh`](run_rectangular.sh) — rectangular: 25×15 (9 datasets) + 90×70 (3)
- [`run_hexagonal.sh`](run_hexagonal.sh) — hexagonal: 25×15 (9 datasets) + 90×70 (3)

Each script runs a `cpu` pass (`torchsom`, `MiniSom`, `MiniSom`-JIT) **and** a `cuda` pass
(`torchsom`, `somoclu`) per config.

The full sweep is multi-day (the 90×70 CPU `MiniSom` runs dominate; see *Expected runtime*), so run it inside **tmux** so it survives SSH disconnects and can be detached/reattached.
First confirm the GPU + somoclu build are live (otherwise the `cuda` pass silently skips):

```bash
cd benchmark
uv run python -c "import torch; assert torch.cuda.is_available(); import somoclu; print('GPU + somoclu OK')"
```

```bash
# 1) start a named, detachable session (it inherits the current directory)
tmux new -s torchsom-bench

# 2) inside the session — rectangular THEN hexagonal, sequential so they don't
#    contend for the single GPU. '&&' runs hexagonal only if rectangular exits 0.
bash run_rectangular.sh && bash run_hexagonal.sh

# 3) detach with   Ctrl-b  then  d   (the sweep keeps running in the background)
#    reattach any time with:
tmux attach -t torchsom-bench
```

Or launch it fully detached in one line (run from the `benchmark/` directory; the scripts
self-`cd`, so the session inherits this directory):

```bash
tmux new -d -s torchsom-bench 'bash run_rectangular.sh && bash run_hexagonal.sh'
```

Monitor progress without attaching — each sweep keeps a live PASS/FAIL summary (from `benchmark/`):

```bash
tail -f logs/sweep_*/summary.log
```

Results land in `results/sweep_<timestamp>_<topology>_<size>/<dataset>/.../results.yml`.
Existing result folders are never overwritten; every launch creates new timestamped folders.

## Expected runtime

The sweep is dominated by the `MiniSom` CPU runs on large data. The table below is for the **25×15** map; the **90×70** map has ~17× the neurons, and the largest cell (`blobs_20000_300` @ 90×70 CPU) can take ~2–3 days per launch, making the full rectangular + hexagonal sweep a week-plus of wall-clock. Representative 25×15 wall-clock per run on the revision hardware (Intel Xeon Gold 6134, 16 cores, 187 GB RAM; NVIDIA Tesla V100-32GB), as reported in the paper's Tables 2 and 3:

| Backend             | Smallest config (240 × 4) | Largest config (16 000 × 300) |
| ------------------- | ------------------------- | ----------------------------- |
| MiniSom (CPU)       | ~1.5 s                    | ~15 min                       |
| MiniSom-JIT (CPU)   | ~4.3 s                    | ~4.6 min                      |
| torchsom (CPU)      | ~0.3 s                    | ~23 s                         |
| somoclu (GPU, V100) | ~0.15 s                   | ~161 s                        |
| torchsom (GPU, V100)| ~1.3 s                    | ~13 s                         |
