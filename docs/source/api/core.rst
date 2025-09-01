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

SOM Variants (WORK IN PROGRESS)
-------------------------------

Periodic Boundary Conditioned SOM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Growing SOM
~~~~~~~~~~~

.. .. automodule:: torchsom.core.growing
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. .. automodule:: torchsom.core.growing.components
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. .. automodule:: torchsom.core.growing.growing_som
..    :members:
..    :undoc-members:
..    :show-inheritance:

Hierarchical SOM
~~~~~~~~~~~~~~~~

.. .. automodule:: torchsom.core.hierarchical
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. .. automodule:: torchsom.core.hierarchical.components
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. .. automodule:: torchsom.core.hierarchical.hierarchical_som
..    :members:
..    :undoc-members:
..    :show-inheritance:
