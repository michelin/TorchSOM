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


v1.1.1
------

- Periodic Boundary Conditions (PBC) support for toroidal SOM topologies
- FAISS backend for accelerated BMU search (``pip install torchsom[faiss]``)
- Configurable search backend (``auto``, ``torch``, ``faiss``)
- Additional quality-of-life improvements and bug fixes

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
- Comprehensive visualization suite (9 map types)
- Clustering integration (K-Means, GMM, HDBSCAN)
- Just-In-Time Learning (JITL) via ``collect_samples()``
- Pydantic-based configuration with validation
- 90%+ test coverage
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
