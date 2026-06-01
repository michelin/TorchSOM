Core API
========

The core module contains the main SOM classes and implementations.

Base Classes
------------

.. automodule:: torchsom.core.base_som
   :members:
   :undoc-members:
   :show-inheritance:

Classical SOM Implementation
-----------------------------

.. automodule:: torchsom.core.som
   :members:
   :undoc-members:
   :show-inheritance:

Example usage
~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torchsom import SOM

   X = torch.randn(1000, 4)
   som = SOM(x=10, y=10, num_features=4, epochs=20)
   som.initialize_weights(data=X, mode="pca")
   q_errors, t_errors = som.fit(X)

   # Build maps via unified API
   distance_map = som.build_map("distance")
   hit_map = som.build_map("hit", data=X)

   # Efficiently build multiple maps with shared BMUs
   results = som.build_multiple_maps(
       map_configs=[
           {"type": "hit"},
           {"type": "distance"},
       ],
       data=X,
   )

Periodic boundary conditions
----------------------------

Periodic boundary conditions are **not a separate class**. They are enabled with the
``pbc=True`` argument of :class:`~torchsom.core.SOM`, which wraps the grid into a
torus for both rectangular and hexagonal topologies, removing edge effects. See
:doc:`../user_guide/topologies` for when and how to use them.

Roadmap
-------

Growing and Hierarchical SOM variants are planned (see the paper's Conclusion). They
live under ``torchsom.core.growing`` and ``torchsom.core.hierarchical`` as
work-in-progress modules, are not yet part of the public API, and are therefore not
documented here. Track progress in the :doc:`../additional_resources/changelog`.
