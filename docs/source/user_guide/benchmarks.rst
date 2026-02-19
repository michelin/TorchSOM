Benchmarks
==========

TorchSOM's computational performance and fidelity are evaluated against
`MiniSom <https://github.com/JustGlowing/minisom>`_, the most widely adopted
and actively maintained SOM library. These benchmarks are from
`the paper <https://arxiv.org/abs/2510.11147>`_ (Section 4, Appendix D).

.. contents:: On this page
   :local:
   :depth: 2


Methodology
-----------

Synthetic datasets are generated using scikit-learn's ``make_blobs()``, varying
both sample size and feature dimensionality. Both implementations use identical
hyperparameters:

- **Grid size**: 25 x 15 (rectangular)
- **Initialization**: PCA
- **Training iterations**: 100 epochs
- **Neighborhood function**: Gaussian
- **Distance function**: Euclidean
- **Seed**: Fixed for reproducibility

Hardware:

- **CPU**: Intel Xeon Platinum 8370C (Ice Lake) @ 3.4 GHz, 8 cores, 16 GB RAM
- **GPU**: NVIDIA Tesla T4, 2560 CUDA cores, 16 GB RAM

Each configuration is averaged over **10 independent runs**.


Small Rectangular Map (25 x 15)
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 8 10 10 10 10 10 10 10 10 10

   * - Samples
     - Features
     - QE (MiniSom)
     - TE% (MiniSom)
     - Time (MiniSom)
     - QE (CPU)
     - TE% (CPU)
     - Time (CPU)
     - QE (GPU)
     - TE% (GPU)
     - Time (GPU)
   * - 240
     - 4
     - 0.17
     - 26
     - 1.82 s
     - 0.24
     - 12
     - 0.41 s
     - 0.24
     - 10
     - 0.24 s
   * - 240
     - 50
     - 1.79
     - 49
     - 3.84 s
     - 1.79
     - 12
     - 0.50 s
     - 1.83
     - 15
     - 0.25 s
   * - 240
     - 300
     - 5.43
     - 71
     - 15.4 s
     - 5.21
     - 47
     - 0.72 s
     - 5.21
     - 30
     - 0.28 s
   * - 4,000
     - 4
     - 0.16
     - 31
     - 54.6 s
     - 0.23
     - 6
     - 4.87 s
     - 0.23
     - 7
     - 2.61 s
   * - 4,000
     - 50
     - 1.66
     - 56
     - 89.2 s
     - 1.77
     - 14
     - 5.33 s
     - 1.74
     - 16
     - 2.62 s
   * - 4,000
     - 300
     - 5.14
     - 74
     - 261 s
     - 5.13
     - 19
     - 8.32 s
     - 5.13
     - 20
     - 2.77 s
   * - 16,000
     - 4
     - 0.15
     - 32
     - 1,054 s
     - 0.23
     - 6
     - 20.2 s
     - 0.23
     - 6
     - 10.9 s
   * - 16,000
     - 50
     - 1.64
     - 57
     - 1,220 s
     - 1.75
     - 13
     - 21.9 s
     - 1.75
     - 14
     - 10.8 s
   * - 16,000
     - 300
     - 5.15
     - 75
     - 1,939 s
     - 5.14
     - 18
     - 30.4 s
     - 5.14
     - 17
     - 11.6 s

*QE = Quantization Error (lower is better). TE% = Topographic Error as percentage (lower is better).*


Key Findings
------------

Training Speed
~~~~~~~~~~~~~~

TorchSOM achieves dramatic speedups over MiniSom:

- **CPU**: 77--98% faster across all configurations
- **GPU**: Up to 99% faster for large, high-dimensional datasets
- The 16,000-sample / 300-feature case: MiniSom takes **32 minutes**,
  torchsom (GPU) takes **12 seconds** — a **167x speedup**

These improvements hold even though torchsom computes both QE and TE at every
epoch (an :math:`\mathcal{O}(2 \times \text{batch} \times \text{epochs})`
overhead that MiniSom does not incur).

Topographic Error
~~~~~~~~~~~~~~~~~

TorchSOM consistently produces maps with **34--81% lower Topographic Error**,
indicating significantly better topology preservation. This means the spatial
relationships in the input data are more faithfully represented on the SOM grid.

Quantization Error
~~~~~~~~~~~~~~~~~~

Both libraries achieve comparable Quantization Error across all configurations,
confirming that torchsom's batch learning approach maintains representation
fidelity equivalent to MiniSom's online learning.


Large Map Results (90 x 70)
----------------------------

For large maps, torchsom is the only viable option. MiniSom fails to complete
within reasonable time for the largest configurations:

.. list-table::
   :header-rows: 1
   :widths: 10 10 15 15 15 15 15 15

   * - Samples
     - Features
     - QE (MiniSom)
     - Time (MiniSom)
     - QE (CPU)
     - Time (CPU)
     - QE (GPU)
     - Time (GPU)
   * - 240
     - 300
     - 5.45
     - 364 s
     - 5.17
     - 7.07 s
     - 5.18
     - 0.40 s
   * - 4,000
     - 300
     - 5.0
     - 6,149 s
     - 5.11
     - 77.8 s
     - 5.10
     - 4.57 s
   * - 16,000
     - 300
     - N/A
     - N/A
     - 5.11
     - 321 s
     - 5.12
     - 19.5 s

*MiniSom is unable to process the 16,000 x 300 configuration on a 90 x 70 grid
within a reasonable time.*


Interpretation
--------------

TorchSOM's advantages stem from two design decisions:

1. **Batch learning**: Instead of updating weights sample-by-sample (online),
   torchsom processes entire batches, enabling vectorized operations and
   efficient GPU utilization.

2. **PyTorch backend**: Leverages optimized BLAS routines and CUDA kernels for
   distance computation, BMU search, and weight updates, all operating on
   contiguous tensor memory.

Together, these enable torchsom to scale to datasets and map sizes that are
impractical with existing libraries, while maintaining or improving map quality.


Further Reading
---------------

- Full benchmark tables (including hexagonal maps): `arXiv paper, Appendix D <https://arxiv.org/abs/2510.11147>`_
- :doc:`architecture` — Package design and module overview
- :doc:`../getting_started/basic_concepts` — SOM mathematical foundations
