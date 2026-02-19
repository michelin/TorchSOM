TorchSOM Documentation
======================

.. image:: _static/assets/logo.png
   :alt: TorchSOM Logo
   :width: 280px
   :align: center

.. raw:: html

   <p align="center" style="margin-top: 0.5em; margin-bottom: 1.5em;">
     <em>GPU-accelerated Self-Organizing Maps in PyTorch with a scikit-learn API,
     rich visualization, and clustering.</em>
   </p>

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started/quickstart
      :link-type: doc

      Install TorchSOM and train your first Self-Organizing Map in under 5 minutes.

   .. grid-item-card:: User Guide
      :link: user_guide/visualization_help
      :link-type: doc

      Comprehensive guides on visualization, architecture, and benchmarks.

   .. grid-item-card:: API Reference
      :link: api/core
      :link-type: doc

      Full API documentation for ``torchsom.core``, ``torchsom.utils``, and ``torchsom.visualization``.

   .. grid-item-card:: Paper
      :link: https://arxiv.org/abs/2510.11147
      :link-type: url

      Read the accompanying paper: *torchsom: The Reference PyTorch Library for Self-Organizing Maps* (Berthier et al., 2025).


Key Features
------------

.. grid:: 2
   :gutter: 2

   .. grid-item::

      **Performance**

      - GPU-accelerated training with PyTorch
      - Up to 99% faster than MiniSom
      - Efficient batch processing and memory management
      - Optional FAISS backend for BMU search

   .. grid-item::

      **Flexibility**

      - Multiple SOM variants (classical, PBC, growing, hierarchical)
      - Configurable topologies (rectangular, hexagonal)
      - Four distance metrics, four neighborhood functions
      - Multiple decay schedulers for learning rate and neighborhood width

   .. grid-item::

      **Visualization**

      - Nine visualization types for both rectangular and hexagonal topologies
      - Distance maps, hit maps, component planes, classification maps
      - Score, rank, and metric maps for supervised tasks
      - Clustering diagnostics with elbow plots and quality comparisons

   .. grid-item::

      **Developer-Friendly**

      - scikit-learn-style API (``fit``, ``build_map``, ``cluster``)
      - Pydantic-based configuration with validation
      - Comprehensive type hints throughout
      - 90%+ test coverage


Quick Start
-----------

Install TorchSOM:

.. code-block:: bash

   pip install torchsom

Train and visualize a SOM:

.. code-block:: python

   import torch
   from torchsom import SOM
   from torchsom.visualization import SOMVisualizer

   som = SOM(x=10, y=10, num_features=3, epochs=50)

   X = torch.randn(1000, 3)
   som.initialize_weights(data=X, mode="pca")
   q_errors, t_errors = som.fit(data=X)

   visualizer = SOMVisualizer(som=som)
   visualizer.plot_training_errors(
       quantization_errors=q_errors, topographic_errors=t_errors
   )
   visualizer.plot_distance_map()
   visualizer.plot_hit_map(data=X)


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


Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/architecture
   user_guide/benchmarks
   user_guide/visualization_help

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/utils
   api/visualization
   api/configs

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources

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
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
