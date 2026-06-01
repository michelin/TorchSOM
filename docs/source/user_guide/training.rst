Training
========

This guide covers how to configure and monitor SOM training: initialization, the
decay schedules for the learning rate and neighborhood width, the key
hyperparameters, and the BMU search backend. The update rule itself is derived in
:doc:`../getting_started/basic_concepts`.


The training loop in one call
------------------------------

Training is two steps — initialize the weights, then fit:

.. code-block:: python

   import torch
   from torchsom import SOM

   data = torch.randn(2000, 8)

   som = SOM(x=25, y=15, num_features=8, epochs=100, batch_size=16)
   som.initialize_weights(data=data, mode="pca")
   q_errors, t_errors = som.fit(data=data)

``fit`` shuffles the data each epoch, processes it in batches, applies the
neighborhood-weighted update, decays the learning rate and neighborhood width, and
records the quantization error (QE) and topographic error (TE) per epoch. The two
returned lists are your convergence trace.


Initialization
--------------

``initialize_weights`` seeds the codebook before training:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Behavior
   * - ``"pca"``
     - Spread weights along the first two principal components of the data. Faster,
       more reproducible convergence — the recommended default.
   * - ``"random"``
     - Sample initial weights randomly from the data range.

.. code-block:: python

   som.initialize_weights(data=data, mode="pca")     # or "random"

Initialization quality strongly affects the final map; PCA initialization usually
reaches a lower QE/TE in fewer epochs.


Decay schedules
---------------

The learning rate :math:`\alpha(t)` and neighborhood width :math:`\sigma(t)` shrink
over training so that updates start broad (global ordering) and end local (fine
tuning). Pick a schedule per parameter:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Schedule
     - Learning rate (``lr_decay_function``)
     - Neighborhood width (``sigma_decay_function``)
   * - Asymptotic *(default)*
     - ``"asymptotic_decay"``
     - ``"asymptotic_decay"``
   * - Inverse
     - ``"lr_inverse_decay_to_zero"``
     - ``"sig_inverse_decay_to_one"``
   * - Linear
     - ``"lr_linear_decay_to_zero"``
     - ``"sig_linear_decay_to_one"``

The inverse and linear schedules guarantee :math:`\alpha(T) \to 0` and
:math:`\sigma(T) \to 1` by the final epoch — zero global drift and single-neuron
updates at the end, which is what gives the map its fine local structure. The exact
formulas are in :doc:`../getting_started/basic_concepts`.

.. code-block:: python

   som = SOM(
       x=25, y=15, num_features=8,
       learning_rate=0.95,                      # initial alpha
       sigma=1.75,                              # initial neighborhood width
       lr_decay_function="lr_linear_decay_to_zero",
       sigma_decay_function="sig_inverse_decay_to_one",
   )


Key hyperparameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 12 64

   * - Parameter
     - Default
     - Guidance
   * - ``epochs``
     - 10
     - Full passes over the data. Increase until QE/TE flatten.
   * - ``batch_size``
     - 5
     - Larger batches use the GPU more efficiently; raise it for big data.
   * - ``learning_rate``
     - 0.5
     - Initial step size, typically 0.1–1.0.
   * - ``sigma``
     - 1.0
     - Initial neighborhood radius. Scale with the grid size.
   * - ``neighborhood_order``
     - 1
     - Discrete neighborhood extent; also used by :doc:`jitl` retrieval.
   * - ``neighborhood_function``
     - ``"gaussian"``
     - Also ``"mexican_hat"``, ``"bubble"``, ``"triangle"``.
   * - ``distance_function``
     - ``"euclidean"``
     - Also ``"cosine"``, ``"manhattan"``, ``"chebyshev"``.
   * - ``random_seed``
     - 42
     - Fix for reproducible runs.

.. tip::

   Always standardize features before training (e.g. scikit-learn's
   ``StandardScaler``). The BMU search compares raw feature distances, so
   unscaled features let large-magnitude columns dominate.


BMU search backend
------------------

Finding the Best-Matching Unit is the per-step bottleneck. The ``search_backend``
argument selects the implementation:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Behavior
   * - ``"auto"`` *(default)*
     - Use FAISS if it is installed, otherwise the PyTorch backend.
   * - ``"torch"``
     - Full pairwise distance computation on GPU/CPU. No extra dependency.
   * - ``"faiss"``
     - Approximate nearest-neighbor search, faster for large maps and
       high-dimensional inputs. Install with ``uv add torchsom[faiss]``.

.. code-block:: python

   som = SOM(x=90, y=70, num_features=300, search_backend="auto")

For the default 25×15 grids, the PyTorch backend is already fast; FAISS pays off on
large maps (e.g. 90×70) or high-dimensional data.


Monitoring convergence
----------------------

Use the returned error traces to decide whether training was long enough:

.. code-block:: python

   q_errors, t_errors = som.fit(data=data)
   print(f"final QE = {q_errors[-1]:.4f},  final TE = {t_errors[-1]:.4f}")

Plot them with :meth:`~torchsom.visualization.SOMVisualizer.plot_training_errors`
(see :doc:`visualization_help`). Both curves should fall and then flatten; if either
is still dropping at the last epoch, raise ``epochs``.

You can also compute the metrics on held-out data:

.. code-block:: python

   qe = som.quantization_error(data=test_data)
   te = som.topographic_error(data=test_data)


Next steps
----------

- :doc:`topologies` — Grid choice and periodic boundary conditions
- :doc:`visualization_help` — Plotting the training curve and maps
- :doc:`../tutorials/index` — Full runs on real datasets
- :doc:`../api/core` — ``SOM`` and ``fit`` reference
