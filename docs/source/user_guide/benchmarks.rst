Benchmarks
==========

TorchSOM is benchmarked against `MiniSom <https://github.com/JustGlowing/minisom>`_,
the most widely adopted SOM library, and `somoclu <https://github.com/peterwittek/somoclu>`_,
a massively parallel C++/CUDA implementation. The results below are the ones published
in `the paper <https://www.jmlr.org/papers/v27/25-2467.html>`_ (Section 4, Appendix D).


Methodology
-----------

Backends are only compared on the same device:

- **CPU**: MiniSom (standard online training), MiniSom-JIT (its optional Numba-compiled
  batch-offline routine), and TorchSOM.
- **GPU**: somoclu (CUDA C++) and TorchSOM (CUDA PyTorch).

Data sets come from scikit-learn's ``make_blobs()``, with 240, 4000, or 16000 samples and
4, 50, or 300 features. Every backend uses the same hyperparameters: a 25 × 15
rectangular grid, PCA initialization, 100 training iterations, a Gaussian neighborhood,
and Euclidean distance.

Both tables report the test quantization error (QE), the test topographic error (TE),
and the wall-clock time, averaged over 10 runs. The time covers initialization, training,
and any one-time cost (Numba compilation, GPU setup), so it is the full cost of a first
use. Lower is better for all three. **Bold** marks the best value per metric and row, and
standard deviations are shown only when non-zero.

Hardware: Intel Xeon Gold 6134 CPU (16 cores at 3.20 GHz, 187 GB RAM) and NVIDIA Tesla
V100-PCIE GPU (5120 CUDA cores, 32 GB RAM).


CPU results
-----------

+--------------------+-----------------------------------+----------------------------------+--------------------------------------+
| Data set           | MiniSom (CPU)                     | MiniSom-JIT (CPU)                | TorchSOM (CPU)                       |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| Samples | Features | QE       | TE (%)  | Time (s)     | QE       | TE (%) | Time (s)     | QE       | TE (%) | Time (s)         |
+=========+==========+==========+=========+==============+==========+========+==============+==========+========+==================+
| 240     | 4        | **0.17** | 26 ± 4  | 1.45 ± 0.03  | 0.18     | 23     | 4.31 ± 0.01  | 0.24     | **2**  | **0.27 ± 0.04**  |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| 240     | 50       | 1.80     | 48 ± 7  | 2.74 ± 0.05  | **1.78** | 40     | 2.38 ± 0.01  | 1.79     | **5**  | **0.34 ± 0.07**  |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| 240     | 300      | 5.44     | 71 ± 10 | 13.84 ± 0.19 | 5.34     | 52     | 5.87 ± 0.01  | **5.21** | **27** | **0.52 ± 0.04**  |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| 4000    | 4        | **0.16** | 32 ± 1  | 25.08 ± 0.35 | **0.16** | 20     | 5.91 ± 0.03  | 0.23     | **0**  | **3.26 ± 0.05**  |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| 4000    | 50       | 1.66     | 55 ± 2  | 49.03 ± 1.10 | **1.64** | 28     | 16.33 ± 0.58 | 1.77     | **4**  | **3.50 ± 0.09**  |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| 4000    | 300      | 5.14     | 74 ± 2  | 226 ± 2      | **5.02** | 37     | 69.47 ± 0.35 | 5.13     | **6**  | **5.25 ± 0.10**  |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| 16000   | 4        | **0.15** | 32 ± 1  | 99.35 ± 1.53 | **0.15** | 19     | 17.58 ± 0.05 | 0.23     | **1**  | **14.55 ± 0.20** |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| 16000   | 50       | 1.64     | 56 ± 1  | 186 ± 2      | **1.60** | 25     | 51.35 ± 0.20 | 1.75     | **3**  | **15.63 ± 0.20** |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+
| 16000   | 300      | 5.15     | 76 ± 1  | 900 ± 10     | **4.98** | 31     | 276 ± 1      | 5.14     | **6**  | **22.68 ± 0.08** |
+---------+----------+----------+---------+--------------+----------+--------+--------------+----------+--------+------------------+

GPU results
-----------

+--------------------+--------------------------------------+------------------------------------------+
| Data set           | somoclu (GPU)                        | TorchSOM (GPU)                           |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| Samples | Features | QE       | TE (%) | Time (s)         | QE       | TE (%)     | Time (s)         |
+=========+==========+==========+========+==================+==========+============+==================+
| 240     | 4        | **0.19** | 23     | **0.15 ± 0.01**  | 0.24     | **2**      | 1.29 ± 0.04      |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| 240     | 50       | **1.80** | 57     | **0.51 ± 0.01**  | 1.83     | **3**      | 1.37 ± 0.04      |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| 240     | 300      | 5.34     | 53     | 2.37 ± 0.01      | **5.21** | **14 ± 2** | **1.34 ± 0.03**  |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| 4000    | 4        | **0.16** | 24     | **1.77 ± 0.02**  | 0.23     | **1**      | 3.32 ± 0.01      |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| 4000    | 50       | **1.61** | 37     | 8.05 ± 0.05      | 1.74     | **4**      | **3.44 ± 0.04**  |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| 4000    | 300      | **5.02** | 47     | 40.92 ± 1.30     | 5.13     | **7**      | **3.33 ± 0.03**  |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| 16000   | 4        | **0.15** | 22     | **10.48 ± 0.27** | 0.23     | **0**      | 12.71 ± 0.03     |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| 16000   | 50       | **1.60** | 30     | 31.82 ± 0.16     | 1.75     | **4**      | **13.29 ± 0.10** |
+---------+----------+----------+--------+------------------+----------+------------+------------------+
| 16000   | 300      | **4.98** | 37     | 161              | 5.14     | **5**      | **13.08 ± 0.04** |
+---------+----------+----------+--------+------------------+----------+------------+------------------+

Key findings
------------

**Topology preservation.** TorchSOM has the lowest TE in every rectangular
configuration, on both CPU and GPU: 62% to 100% below standard MiniSom, and below
MiniSom-JIT and somoclu throughout (at 16000 × 300, 6% versus 76%, 31%, and 37%). On
hexagonal maps it leads in nearly all configurations; the exceptions are the 4-feature
sets (Appendix D).

**Quantization error.** QE is comparable across backends, with TorchSOM marginally
higher.

**CPU speed.** TorchSOM is 81% to 98% faster than standard MiniSom. MiniSom updates one
sample at a time, while TorchSOM processes the whole batch with vectorized tensor
operations, so the gap grows with sample size and feature count. TorchSOM also matches or
beats MiniSom-JIT in every rectangular configuration (up to ~16×). The gap is smallest on
the 4-feature sets, where there is little batched work to amortize.

**GPU speed.** TorchSOM's GPU time includes a one-time initialization. On the smallest
and 4-feature sets, that overhead dominates and somoclu's compiled kernels are faster. As
the data grows, the overhead is amortized and TorchSOM is up to ~12× faster than somoclu
on high-dimensional data.

**Device choice.** The GPU helps most on large, high-dimensional data: at 16000 × 300 it
cuts the time by about 40% relative to the CPU. For small or low-dimensional data, the
multi-core CPU is competitive or faster.

**Larger maps.** On a 90 × 70 grid (about 17× the neurons), TorchSOM's CPU speedup grows
to ~200× over standard MiniSom and ~50× over MiniSom-JIT, and its GPU speedup over
somoclu reaches ~66×. At that size, TorchSOM's GPU run is ~8× faster than its CPU run. TE
stays well below every baseline and QE remains on par. These tables, and the hexagonal
results, are in Appendix D of the paper.

All TorchSOM times are conservative: TorchSOM also evaluates QE and TE over the full
batch at every epoch, an :math:`\mathcal{O}(2 \times \text{batch} \times \text{epochs})`
cost that MiniSom and somoclu avoid by measuring once at the end.


Reproducibility
---------------

All benchmark scripts and configuration files live in the
`benchmark/ <https://github.com/michelin/TorchSOM/tree/main/benchmark>`_ directory, and
its README walks through the setup step by step. Three Git tags pin the paper states:

- ``jmlr-submission-v1``: the original submission (October 2025).
- ``jmlr-revision-v1``: the first revision, with the same benchmark code.
- ``jmlr-revision-v2``: the published version, which adds the somoclu and MiniSom-JIT
  baselines and reproduces the tables above.

The first two tags use MiniSom v2.3.5 (commit ``65b6ba6``). ``jmlr-revision-v2`` uses
commit ``2eb8bd7`` from MiniSom's development branch for both MiniSom and MiniSom-JIT,
because the JIT routine was only available there at the time.


Further reading
---------------

- :doc:`comparison` — Feature-by-feature comparison with other SOM libraries
- Full benchmark tables, including 90 × 70 and hexagonal maps:
  `paper, Appendix D <https://www.jmlr.org/papers/v27/25-2467.html>`_
- :doc:`architecture` — Package design and module overview
- :doc:`../getting_started/basic_concepts` — SOM mathematical foundations
