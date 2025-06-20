Changelog
=========

All notable changes to TorchSOM will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_, and this project adheres to `Semantic Versioning <https://semver.org/>`_.

.. ! Below are not used but keeping them could be relevant for the future

.. .. [Unreleased]
.. ------------

.. Added
.. ~~~~~
.. - Comprehensive documentation with tutorials and examples
.. - Advanced visualization capabilities with customizable configurations
.. - GPU acceleration support for training
.. - Multiple SOM variants (Growing SOM, Hierarchical SOM)
.. - Pydantic-based configuration management
.. - Type hints throughout the codebase
.. - Performance benchmarking tools

.. Changed
.. ~~~~~~~
.. - Improved API consistency with scikit-learn patterns
.. - Enhanced error handling and validation
.. - Optimized memory usage for large datasets

.. Fixed
.. ~~~~~
.. - Edge cases in distance calculations
.. - Memory leaks during long training sessions
.. - Visualization issues with hexagonal topologies

.. .. [0.1.0] - 2024-01-15
.. --------------------

.. Added
.. ~~~~~
.. - Initial release of TorchSOM
.. - Basic SOM implementation with PyTorch backend
.. - Core utilities for distance functions, neighborhoods, and decay
.. - Basic visualization with matplotlib
.. - Standard SOM training algorithms
.. - Support for rectangular and hexagonal topologies

.. Features
.. ~~~~~~~~
.. - **Core SOM Implementation**: Complete self-organizing map with customizable parameters
.. - **Multiple Distance Functions**: Euclidean, Cosine, Manhattan, and Chebyshev distances
.. - **Neighborhood Functions**: Gaussian, Mexican Hat, Bubble, and Triangle neighborhoods
.. - **Flexible Topologies**: Support for both rectangular and hexagonal grid layouts
.. - **Visualization Tools**: Basic plotting capabilities for distance maps and hit maps
.. - **GPU Support**: Automatic device detection and CUDA acceleration
.. - **Configuration Management**: Structured parameter validation with Pydantic

.. Technical Details
.. ~~~~~~~~~~~~~~~~~
.. - Minimum Python version: 3.8
.. - PyTorch dependency: >=1.10.0
.. - Full type annotation support
.. - Comprehensive test coverage
.. - CI/CD pipeline with GitHub Actions

.. Migration Guide
.. ---------------

.. From v0.0.x to v0.1.0
.. ~~~~~~~~~~~~~~~~~~~~~

.. Breaking Changes
.. ................

.. - **Import paths changed**: Update import statements

..   .. code-block:: python
  
..      # Old
..      from torchsom.som import SOM
     
..      # New  
..      from torchsom import SOM

.. - **Parameter names**: Some parameter names were standardized

..   .. code-block:: python
  
..      # Old
..      som = SOM(map_size=(10, 10), learning_rate=0.5)
     
..      # New
..      som = SOM(x=10, y=10, learning_rate=0.5)

.. - **Visualization API**: Updated method signatures

..   .. code-block:: python
  
..      # Old
..      som.plot_distance_map()
     
..      # New
..      from torchsom.visualization import SOMVisualizer
..      viz = SOMVisualizer(som)
..      viz.plot_distance_map()

.. From v0.1.0 to v0.2.0
.. ~~~~~~~~~~~~~~~~~~~~~

.. Breaking Changes
.. ................

.. Deprecated Features
.. ~~~~~~~~~~~~~~~~~~~

.. The following features are deprecated and will be removed in v0.2.0:

.. - XXX
.. - XXX

.. Upgrade Steps
.. ~~~~~~~~~~~~~

.. 1. XXX
.. 2. XXX

How to Contribute
-----------------

We welcome contributions! See our `contributing guide <https://github.com/michelin/TorchSOM/blob/main/CONTRIBUTING.md>`_ for:

Report Issues
~~~~~~~~~~~~~

Found a bug or have a feature request? Please:

1. Check existing `GitHub Issues <https://github.com/michelin/TorchSOM/issues>`_
2. Create a new issue with detailed information
3. Include minimal reproduction examples
4. Specify your environment details
