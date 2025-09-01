.. TorchSOM documentation master file, created by
   sphinx-quickstart on Tue May  6 18:37:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TorchSOM Documentation
======================

.. .. image:: _static/assets/logo.png
..    :alt: TorchSOM Logo
..    :width: 200px
..    :align: center

**TorchSOM** is a modern PyTorch-based library for training and visualizing **Self-Organizing Maps (SOMs)**,
a powerful unsupervised learning algorithm used for clustering, dimensionality reduction, and data exploration.

Built with PyTorch at its core, TorchSOM seamlessly integrates with modern deep learning workflows while
providing GPU acceleration for high-performance computing.

.. note::
   🌟 If you find this project interesting, we would be grateful for your support by starring
   this `GitHub repository <https://github.com/michelin/TorchSOM>`_.

Key Features
------------

🚀 **Performance**
   - GPU-accelerated training with PyTorch
   - Efficient batch processing and memory management
   - Optimized for large-scale data

🎯 **Flexibility**
   - Multiple SOM variants (classical, growing, hierarchical, ...)
   - Configurable topologies (rectangular, hexagonal)
   - Extensive customization options

📊 **Visualization**
   - Rich visualization suite with matplotlib
   - Publication-ready figures

🔧 **Developer-Friendly**
   - Clean, modular architecture
   - Comprehensive API documentation
   - Type hints and validation with Pydantic

Quick Start
-----------

Install TorchSOM:

.. code-block:: bash

   pip install torchsom

Basic usage:

.. code-block:: python

   import torch
   from torchsom import SOM
   from torchsom.visualization import SOMVisualizer

   # Create a 10x10 map for 3D input
   som = SOM(x=10, y=10, num_features=3, epochs=50)

   # Train SOM for 50 epochs on 1000 samples
   X = torch.randn(1000, 3)
   som.initialize_weights(data=X, mode="pca")
   q_errors, t_errors = som.fit(data=X)

   # Visualize results
   visualizer = SOMVisualizer(som=som, config=None)
   visualizer.plot_training_errors(quantization_errors=q_errors, topographic_errors=t_errors, save_path=None)
   visualizer.plot_distance_map(save_path=None)
   visualizer.plot_hit_map(data=X, save_path=None)

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

Support & Community
-------------------

- 📖 **Documentation**: You're reading it!
- 🐛 **Bug Reports**: `GitHub Issues <https://github.com/michelin/TorchSOM/issues>`_

License
-------

TorchSOM is released under the Apache License 2.0 License. See the `LICENSE <https://github.com/michelin/TorchSOM/blob/main/LICENSE>`_ file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
