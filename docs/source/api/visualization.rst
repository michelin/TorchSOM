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

   from torchsom import SOM
   from torchsom.visualization import SOMVisualizer
   import torch

   data = torch.randn(500, 3)
   som = SOM(x=12, y=10, num_features=3, epochs=20)
   som.initialize_weights(data, mode="pca")
   q_errors, t_errors = som.fit(data)

   viz = SOMVisualizer(som)
   viz.plot_all(
       quantization_errors=q_errors,
       topographic_errors=t_errors,
       data=data,
       target=None,
       save_path=None,
   )

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
- For supervised maps (metric/score/rank/classification), you can either pass data and target directly or precompute ``bmus_data_map = som.build_map("bmus_data", data, return_indices=True)`` to speed up repeated plots.
