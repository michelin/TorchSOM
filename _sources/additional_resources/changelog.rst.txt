Changelog
=========

All notable changes to TorchSOM are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

For the full commit history, see the
`GitHub releases <https://github.com/michelin/TorchSOM/releases>`_ page.

The auto-generated ``CHANGELOG.md`` in the repository root is maintained by
`Commitizen <https://commitizen-tools.github.io/commitizen/>`_ and provides
a detailed, commit-level changelog.


v1.3.1
------

- The paper is published in the *Journal of Machine Learning Research*
  (`JMLR 27(190) <https://www.jmlr.org/papers/v27/25-2467.html>`_). Citations,
  ``CITATION.cff``, and the benchmarks and comparison pages now match the published
  version.
- Fix: map plots work again on matplotlib 3.11, which removed ``matplotlib.cm.get_cmap``
  (`#10 <https://github.com/michelin/TorchSOM/issues/10>`_).
- Fix: passing a BMU map built without ``return_indices=True`` to the supervised maps or
  ``collect_samples`` now raises a clear error, and the documentation examples build the
  map correctly (`#11 <https://github.com/michelin/TorchSOM/issues/11>`_).

v1.3.0
------

- Rectangular topographic error now counts diagonal neighbors as adjacent
  (8-neighborhood, threshold 1.0 → 1.42), as MiniSom and somoclu do. TE values on
  rectangular maps are lower than in v1.2.0 and earlier.
- Benchmark scripts add the somoclu (GPU) and MiniSom-JIT (CPU) baselines used in the
  published paper.
- Benchmark GPU timings synchronize CUDA and account for device warm-up.

v1.2.0
------

- Periodic boundary conditions (PBC) for toroidal SOM topologies
- FAISS backend for BMU search (``pip install "torchsom[faiss]"``) and a configurable
  search backend (``auto``, ``torch``, ``faiss``)
- Retrieval modes for ``collect_samples`` (``bmu_only``, ``bmu_neighborhood``,
  ``bmu_neighborhood_knn``), which can also return the collected sample indices

v1.1.1
------

- Release workflow fix; no library changes

v1.1.0
------

- Faster map building, weight initialization, weight updates, and distance computations
- Benchmark scripts comparing TorchSOM with MiniSom
- Expanded documentation

v1.0.0
------

Initial public release of TorchSOM, accompanying the
`arXiv paper <https://arxiv.org/abs/2510.11147>`_.

**Features**

- Classical SOM implementation with PyTorch backend
- GPU-accelerated training with batch learning
- scikit-learn-style API (``fit``, ``build_map``, ``cluster``)
- Rectangular and hexagonal grid topologies
- Four distance functions: Euclidean, Cosine, Manhattan, Chebyshev
- Four neighborhood functions: Gaussian, Mexican Hat, Bubble, Triangle
- Multiple decay schedulers for learning rate and neighborhood width
- PCA and random weight initialization
- Comprehensive visualization suite (seven visualization types)
- Clustering integration (K-Means, GMM, HDBSCAN)
- Just-In-Time Learning (JITL) via ``collect_samples()``
- Pydantic-based configuration with validation
- 90% test coverage
- Full documentation with tutorials and API reference


How to Contribute
-----------------

We welcome contributions! See our `contributing guide <https://github.com/michelin/TorchSOM/blob/main/CONTRIBUTING.md>`_ for details.

Report Issues
~~~~~~~~~~~~~

Found a bug or have a feature request? Please:

1. Check existing `GitHub Issues <https://github.com/michelin/TorchSOM/issues>`_
2. Create a new issue with detailed information
3. Include minimal reproduction examples
4. Specify your environment details
