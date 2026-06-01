Package Architecture
====================

TorchSOM follows a modular design built around three core components that provide
a complete SOM implementation with native PyTorch integration.


Module Overview
---------------

.. code-block:: text

   torchsom/
   ├── core/              # SOM implementations
   │   ├── base_som.py        # Abstract base class (BaseSOM)
   │   ├── som.py             # Classical SOM (batch learning)
   │   ├── growing/           # Growing SOM variant (WIP)
   │   └── hierarchical/      # Hierarchical SOM variant (WIP)
   ├── utils/             # Training utilities
   │   ├── distances.py       # Distance functions (Euclidean, Cosine, Manhattan, Chebyshev)
   │   ├── neighborhood.py    # Neighborhood kernels (Gaussian, Mexican Hat, Bubble, Triangle)
   │   ├── decay.py           # Schedulers for learning rate and neighborhood width
   │   ├── initialization.py  # Weight initialization (random, PCA)
   │   ├── grid.py            # Grid coordinate generation
   │   ├── topology.py        # Topology utilities
   │   ├── maps.py            # Map computation (hit, distance, metric, score, rank, classification)
   │   ├── metrics.py         # Quality metrics (QE, TE)
   │   ├── clustering.py      # Clustering algorithms (K-Means, GMM, HDBSCAN)
   │   └── search.py          # BMU search backends (PyTorch, FAISS)
   ├── visualization/     # Visualization suite
   │   ├── base.py            # SOMVisualizer factory
   │   ├── base_visualizer.py # Abstract base with shared methods
   │   ├── rectangular.py     # Rectangular topology visualizer
   │   ├── hexagonal.py       # Hexagonal topology visualizer
   │   ├── clustering.py      # Clustering visualization
   │   └── config.py          # VisualizationConfig
   └── configs/           # Configuration management
       └── som_config.py      # SOMConfig (Pydantic model)


Core Module (``torchsom.core``)
-------------------------------

The core module implements classical SOM algorithms within the PyTorch ecosystem.
The main ``SOM`` class inherits from ``BaseSOM`` and provides:

- **``fit(data)``** — Train the SOM with automatic GPU acceleration and batch processing.
  Returns per-epoch quantization and topographic errors for convergence monitoring.

- **``build_map(map_type, ...)``** — Generate various map types for visualization:
  ``"hit"``, ``"distance"``, ``"metric"``, ``"score"``, ``"rank"``, ``"classification"``, ``"bmus_data"``.

- **``cluster(method, ...)``** — Cluster SOM neurons using K-Means, GMM, or HDBSCAN
  on the weight space, latent space, or both.

- **``collect_samples(...)``** — Identify relevant historical samples for a given query
  using topology and latent-space distances, enabling Just-In-Time Learning (JITL)
  applications. Three retrieval modes are available through ``retrieval_mode``
  (``"bmu_only"``, ``"bmu_neighborhood"``, ``"bmu_neighborhood_knn"``), with the
  neighborhood extent set by ``neighborhood_order``.

- **``identify_bmus(data)``** — Find Best Matching Units for input data using the
  configured search backend (PyTorch or FAISS).

Class Hierarchy
~~~~~~~~~~~~~~~

.. code-block:: text

   BaseSOM (abstract)
   └── SOM
       ├── fit()
       ├── build_map()
       ├── build_multiple_maps()
       ├── cluster()
       ├── collect_samples()
       ├── identify_bmus()
       ├── quantization_error()
       └── topographic_error()

``BaseSOM`` defines the interface and common attributes (grid dimensions, topology,
device placement). ``SOM`` implements the full training loop with batch learning,
where each epoch shuffles the data, processes it in batches, and updates weights
using the neighborhood-weighted update rule.

.. note::

   The current release ships the classical ``SOM`` with rectangular and hexagonal
   topologies, each optionally wrapped into a torus via :ref:`periodic boundary
   conditions <topologies-pbc>`. Growing and Hierarchical SOM variants are on the
   roadmap (see the paper's Conclusion) and are not part of the public API yet.


Utilities Module (``torchsom.utils``)
-------------------------------------

This module provides essential components for SOM parameterization and training.

Distance Functions
~~~~~~~~~~~~~~~~~~

Four distance metrics are available for BMU selection in feature space:

- **Euclidean** (default): :math:`\delta(\mathbf{x}, \mathbf{w}) \coloneqq \sqrt{\sum_{a=1}^{d} (x_a - w_a)^2}`
- **Manhattan**: :math:`\delta(\mathbf{x}, \mathbf{w}) \coloneqq \sum_{a=1}^{d} |x_a - w_a|`
- **Cosine**: :math:`\delta(\mathbf{x}, \mathbf{w}) \coloneqq 1 - \frac{\mathbf{x} \cdot \mathbf{w}}{\lVert \mathbf{x} \rVert \lVert \mathbf{w} \rVert}`
- **Chebyshev**: :math:`\delta(\mathbf{x}, \mathbf{w}) \coloneqq \max_{a \le d} |x_a - w_a|`

Neighborhood Functions
~~~~~~~~~~~~~~~~~~~~~~

Four neighborhood kernels control the spatial extent of weight updates around the BMU:

- **Gaussian** (default): Smooth, continuous influence decay
- **Mexican Hat** (Ricker wavelet): Excitatory center with inhibitory surround
- **Bubble** (Step): Binary on/off within a fixed radius
- **Triangle** (Linear): Linear decay from BMU to radius boundary

Decay Schedulers
~~~~~~~~~~~~~~~~

Learning rate (:math:`\alpha`) and neighborhood width (:math:`\sigma`) decay over training:

- **Asymptotic decay** (default): :math:`\theta(t+1) \coloneqq \frac{\theta(t)}{1 + t / (T/2)}`
- **Inverse decay**: Gradual asymptotic reduction to 0 (for :math:`\alpha`) or 1 (for :math:`\sigma`)
- **Linear decay**: Uniform reduction to 0 (for :math:`\alpha`) or 1 (for :math:`\sigma`)


Grid Topologies
~~~~~~~~~~~~~~~

Maps use either a ``"rectangular"`` or ``"hexagonal"`` grid. Setting ``pbc=True`` wraps the
grid into a torus (periodic boundary conditions): grid-distance calculations then use the
minimum-image convention, so neighborhoods wrap across opposite edges and corner neurons are
no longer disadvantaged by the map boundary.

BMU Search Backends
~~~~~~~~~~~~~~~~~~~

The backend is chosen with the ``search_backend`` argument (``"auto"``, ``"torch"``, or
``"faiss"``); ``"auto"`` uses FAISS when it is installed and falls back to PyTorch otherwise.

- **PyTorch** (default): Full pairwise distance computation on GPU/CPU
- **FAISS** (optional): Approximate nearest-neighbor search for large maps,
  enabled with ``uv add torchsom[faiss]``


Visualization Module (``torchsom.visualization``)
--------------------------------------------------

The visualization module provides seven visualization types for both rectangular
and hexagonal topologies:

.. list-table::
   :header-rows: 1
   :widths: 32 15 53

   * - Visualization
     - Setting
     - Purpose
   * - U-matrix (distance map)
     - Unsupervised
     - Inter-neuron distances and cluster boundaries
   * - Hit map
     - Unsupervised
     - BMU activation frequency and data density
   * - Component planes
     - Unsupervised
     - Per-feature weight distribution across the grid
   * - Classification & metric maps
     - Supervised
     - Dominant class, or mean/std of a target, per neuron
   * - Score & rank maps
     - Supervised
     - Per-neuron reliability for regression
   * - Training curve
     - Unsupervised
     - QE and TE convergence during training
   * - Clustering maps
     - Unsupervised
     - Cluster assignment plus elbow, silhouette, and comparison diagnostics

The ``SOMVisualizer`` class acts as a factory that delegates to topology-specific
implementations (``RectangularVisualizer`` or ``HexagonalVisualizer``), and routes
clustering plots to a ``ClusteringVisualizer``. See the :doc:`visualization_help`
gallery for every plot with example figures.

.. code-block:: text

   SOMVisualizer (factory)
   ├── delegates to ──► RectangularVisualizer
   │                    └── inherits BaseVisualizer
   └── delegates to ──► HexagonalVisualizer
                        └── inherits BaseVisualizer


Training Data Flow
------------------

The end-to-end workflow for training and analyzing a SOM:

.. code-block:: text

   Input Data (N x k)
        │
        ▼
   ┌─────────────────┐
   │  Initialization  │  PCA or random sampling from data
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Training Loop   │  For each epoch:
   │                   │    1. Shuffle data
   │  ┌─────────────┐ │    2. For each batch:
   │  │ BMU Search   │ │       - Compute distances (feature space)
   │  │ (PyTorch /   │ │       - Find BMU per sample
   │  │  FAISS)      │ │
   │  └──────┬──────┘ │
   │         ▼         │    3. Compute neighborhood influence (grid space)
   │  ┌─────────────┐ │    4. Update weights:
   │  │ Weight       │ │       w(t+1) = w(t) + α(t) · h(t) · (x - w(t))
   │  │ Update       │ │    5. Decay α(t) and σ(t)
   │  └─────────────┘ │    6. Compute QE and TE
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Analysis        │  build_map(), cluster(), collect_samples()
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Visualization   │  SOMVisualizer.plot_*()
   └─────────────────┘


Configuration
-------------

SOM parameters are managed through ``SOMConfig``, a Pydantic model that validates
all inputs at construction time:

.. code-block:: python

   from torchsom.configs import SOMConfig

   config = SOMConfig(
       x=25, y=15,
       topology="hexagonal",
       epochs=100,
       learning_rate=0.95,
       sigma=1.75,
       neighborhood_function="gaussian",
       distance_function="euclidean",
       initialization_mode="pca",
   )

The config can be serialized to/from YAML or JSON for reproducible experiments.
See the :doc:`../api/configs` for full parameter documentation.


Next steps
----------

- :doc:`topologies` — Choosing a grid topology and using periodic boundary conditions
- :doc:`training` — Initialization, decay schedules, and BMU search backends
- :doc:`clustering` — Clustering SOM neurons and reading the diagnostics
- :doc:`jitl` — Retrieving relevant samples for just-in-time learning
- :doc:`visualization_help` — Visualization gallery
- :doc:`../getting_started/basic_concepts` — Mathematical foundations
- :doc:`../api/core` — Full API reference
