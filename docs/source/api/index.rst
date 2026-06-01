API Reference
=============

The public API is small by design. Most workflows use the :class:`~torchsom.core.SOM`
class, the :class:`~torchsom.visualization.SOMVisualizer`, and (optionally) the
:class:`~torchsom.configs.SOMConfig`. The pages below document every module in full.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Core
      :link: core
      :link-type: doc

      ``torchsom.core`` — the ``SOM`` class and its ``BaseSOM`` interface.

   .. grid-item-card:: Utils
      :link: utils
      :link-type: doc

      ``torchsom.utils`` — distances, neighborhoods, decay, clustering, metrics, search.

   .. grid-item-card:: Visualization
      :link: visualization
      :link-type: doc

      ``torchsom.visualization`` — ``SOMVisualizer`` and ``VisualizationConfig``.

   .. grid-item-card:: Configs
      :link: configs
      :link-type: doc

      ``torchsom.configs`` — the Pydantic ``SOMConfig`` model.


Public objects at a glance
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Object
     - Purpose
   * - :class:`torchsom.core.SOM`
     - The Self-Organizing Map: ``fit``, ``build_map``, ``cluster``, ``collect_samples``.
   * - :class:`torchsom.core.BaseSOM`
     - Abstract base defining the SOM interface and shared attributes.
   * - :class:`torchsom.visualization.SOMVisualizer`
     - Factory that renders every plot for the SOM's topology.
   * - :class:`torchsom.visualization.VisualizationConfig`
     - Styling options for figures (size, fonts, colormap, DPI, hex settings).
   * - :class:`torchsom.configs.SOMConfig`
     - Validated, serializable configuration for a ``SOM``.
   * - ``torchsom.DISTANCE_FUNCTIONS``
     - Registry of distance metrics (``euclidean``, ``cosine``, ``manhattan``, ``chebyshev``).
   * - ``torchsom.NEIGHBORHOOD_FUNCTIONS``
     - Registry of neighborhood kernels (``gaussian``, ``mexican_hat``, ``bubble``, ``triangle``).
   * - ``torchsom.DECAY_FUNCTIONS``
     - Registry of learning-rate and neighborhood-width decay schedules.


Import conventions
------------------

The most common objects are re-exported at the top level:

.. code-block:: python

   from torchsom import SOM, SOMVisualizer
   from torchsom.visualization import VisualizationConfig
   from torchsom.configs import SOMConfig

The function registries let you list or extend the available options:

.. code-block:: python

   from torchsom import DISTANCE_FUNCTIONS, NEIGHBORHOOD_FUNCTIONS, DECAY_FUNCTIONS

   print(list(DISTANCE_FUNCTIONS))       # available distance metrics
   print(list(NEIGHBORHOOD_FUNCTIONS))   # available neighborhood kernels


.. seealso::

   The :doc:`../user_guide/architecture` page explains how these modules fit together,
   and the :doc:`../user_guide/training` guide maps the constructor arguments to
   training behavior.
