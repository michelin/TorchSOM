Visualization API
=================

The visualization module provides comprehensive plotting and analysis tools for SOMs.

SOM Visualizer
--------------

.. automodule:: torchsom.visualization.base
   :members:
   :undoc-members:
   :show-inheritance:

Example
~~~~~~~

.. code-block:: python

   import torch
   from torchsom import SOM, SOMVisualizer

   data = torch.randn(500, 3)
   som = SOM(x=12, y=10, num_features=3, epochs=20)
   som.initialize_weights(data=data, mode="pca")
   q_errors, t_errors = som.fit(data=data)

   viz = SOMVisualizer(som=som)
   viz.plot_distance_map()
   viz.plot_hit_map(data=data)

See the :doc:`../user_guide/visualization_help` gallery for every plot, including
``plot_all`` and the supervised maps.

Visualization Configuration
---------------------------

.. automodule:: torchsom.visualization.config
   :members:
   :undoc-members:
   :show-inheritance:

Supported Formats
-----------------

- PNG (default, web-friendly)
- PDF (vector, publication-ready)
- SVG (vector, scalable)
- JPEG (compressed, smaller files)

Notes
-----

- The visualizer is a factory that forwards to topology-specific implementations (hexagonal or rectangular).
- The supervised maps (metric/score/rank/classification) and ``plot_all`` require a
  pre-computed ``bmus_data_map = som.build_map("bmus_data", data=data)``; build it once
  and reuse it across plots.
