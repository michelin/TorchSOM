Utilities API
=============

The utils module provides supporting functions for SOM training and analysis.

Distance Functions
------------------

.. automodule:: torchsom.utils.distances
   :members:
   :undoc-members:
   :show-inheritance:

Available distance functions:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``euclidean``
     - Standard Euclidean distance (default)
   * - ``cosine``
     - Cosine distance (1 - cosine similarity)
   * - ``manhattan``
     - Manhattan (L1) distance
   * - ``chebyshev``
     - Chebyshev (L∞) distance


Neighborhood Functions
----------------------

.. automodule:: torchsom.utils.neighborhood
   :members:
   :undoc-members:
   :show-inheritance:

Available neighborhood functions:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``gaussian``
     - Gaussian neighborhood (default, smooth)
   * - ``mexican_hat``
     - Mexican hat (Ricker wavelet, inhibitory surround)
   * - ``bubble``
     - Bubble function (step function)
   * - ``triangle``
     - Triangular function (linear decay)

Decay Functions
---------------

.. automodule:: torchsom.utils.decay
   :members:
   :undoc-members:
   :show-inheritance:

Available decay functions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``asymptotic_decay``
     - General asymptotic decay (default)
   * - ``lr_inverse_decay_to_zero``
     - Learning rate inverse decay to zero
   * - ``lr_linear_decay_to_zero``
     - Learning rate linear decay to zero
   * - ``sig_inverse_decay_to_one``
     - Sigma inverse decay to one
   * - ``sig_linear_decay_to_one``
     - Sigma linear decay to one

Grid and Topology
-----------------

.. automodule:: torchsom.utils.grid
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: torchsom.utils.topology
   :members:
   :undoc-members:
   :show-inheritance:

Initialization
--------------

.. automodule:: torchsom.utils.initialization
   :members:
   :undoc-members:
   :show-inheritance:

Available initialization methods:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``random``
     - Random sampling from input data (default)
   * - ``pca``
     - PCA-based initialization for faster convergence

Metrics
-------

.. automodule:: torchsom.utils.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Quality metrics for evaluating SOM training:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Description
   * - ``quantization_error``
     - Average distance between input vectors and their Best Matching Units (BMUs)
   * - ``topographic_error``
     - Percentage of data vectors for which the first and second BMUs are not adjacent
