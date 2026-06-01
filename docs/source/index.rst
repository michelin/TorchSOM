TorchSOM Documentation
======================

.. image:: _static/assets/logo.png
   :alt: TorchSOM Logo
   :width: 280px
   :align: center

.. raw:: html

   <p align="center" style="margin-top: 0.5em; margin-bottom: 1.5em;">
     <em>GPU-accelerated Self-Organizing Maps in PyTorch with a scikit-learn API,
     advanced visualization, and clustering.</em>
   </p>

TorchSOM is the PyTorch-native reference implementation of the Self-Organizing Map
(SOM). It pairs a familiar scikit-learn-style API with GPU-accelerated batch
training, a built-in clustering interface, just-in-time-learning support, and a
visualization suite for both rectangular and hexagonal topologies. It accompanies the
paper *torchsom: The Reference PyTorch Library for Self-Organizing Maps*
(`Berthier et al., 2025 <https://arxiv.org/abs/2510.11147>`_).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started/quickstart
      :link-type: doc

      Install TorchSOM and train your first Self-Organizing Map in a few minutes.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      End-to-end worked examples on real datasets: classification, regression, and clustering.

   .. grid-item-card:: User Guide
      :link: user_guide/architecture
      :link-type: doc

      Architecture, topologies, training, clustering, JITL, visualization, and benchmarks.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Full API documentation for ``torchsom.core``, ``torchsom.utils``, and ``torchsom.visualization``.


Key Features
------------

.. grid:: 2
   :gutter: 2

   .. grid-item::

      **Performance**

      - GPU-accelerated batch training with PyTorch
      - 77–99% faster training than MiniSom (see :doc:`user_guide/benchmarks`)
      - Quantization-error parity with lower topographic error
      - Optional FAISS backend for BMU search on large maps

   .. grid-item::

      **Flexibility**

      - Rectangular and hexagonal topologies, optionally toroidal via periodic boundary conditions
      - Four distance metrics and four neighborhood kernels
      - Configurable learning-rate and neighborhood-width decay schedules
      - Configurable BMU search backend (PyTorch or FAISS)

   .. grid-item::

      **Visualization**

      - Seven visualization types for both rectangular and hexagonal topologies
      - Distance, hit, component-plane, classification, and metric maps
      - Score and rank maps for per-neuron reliability in regression
      - Clustering diagnostics: elbow, silhouette, and algorithm comparison

   .. grid-item::

      **Developer-Friendly**

      - scikit-learn-style API (``fit``, ``build_map``, ``cluster``)
      - Pydantic-based configuration with validation
      - Full type hints throughout
      - 90% test coverage, Apache 2.0 licensed


Quick Start
-----------

Install TorchSOM:

.. code-block:: bash

   uv add torchsom

Train and visualize a SOM:

.. code-block:: python

   import torch
   from torchsom import SOM, SOMVisualizer

   som = SOM(x=10, y=10, num_features=3, epochs=50)

   X = torch.randn(1000, 3)
   som.initialize_weights(data=X, mode="pca")
   q_errors, t_errors = som.fit(data=X)

   viz = SOMVisualizer(som=som)
   viz.plot_training_errors(
       quantization_errors=q_errors, topographic_errors=t_errors
   )
   viz.plot_distance_map()
   viz.plot_hit_map(data=X)


Citation
--------

If you use TorchSOM in your work, please cite:

.. code-block:: bibtex

   @misc{berthier2025torchsom,
       title={torchsom: The Reference PyTorch Library for Self-Organizing Maps},
       author={Berthier, Louis and Shokry, Ahmed and Moreaud, Maxime
               and Ramelet, Guillaume and Moulines, Eric},
       year={2025},
       eprint={2510.11147},
       archivePrefix={arXiv},
       primaryClass={stat.ML},
       url={https://arxiv.org/abs/2510.11147}
   }


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/quickstart
   getting_started/basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/architecture
   user_guide/topologies
   user_guide/training
   user_guide/clustering
   user_guide/jitl
   user_guide/visualization_help
   user_guide/benchmarks
   user_guide/comparison

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/index
   tutorials/iris
   tutorials/wine
   tutorials/boston_housing
   tutorials/energy_efficiency
   tutorials/clustering_walkthrough

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/core
   api/utils
   api/visualization
   api/configs

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources
   :hidden:

   additional_resources/changelog
   additional_resources/faq
   additional_resources/troubleshooting


Support
-------

- **Documentation**: You're reading it!
- **Bug Reports**: `GitHub Issues <https://github.com/michelin/TorchSOM/issues>`_
- **Source Code**: `GitHub <https://github.com/michelin/TorchSOM>`_


License
-------

TorchSOM is released under the `Apache License 2.0 <https://github.com/michelin/TorchSOM/blob/main/LICENSE>`_.


Indices and Tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
