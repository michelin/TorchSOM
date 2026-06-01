Iris — Classification
======================

The Iris dataset (150 samples, 4 features, 3 species) is the classic first SOM. This
tutorial trains a map, checks convergence, and reads the structure through the
U-matrix, hit map, classification map, and component planes.

.. note::

   Full runnable notebook:
   `notebooks/iris.ipynb <https://github.com/michelin/TorchSOM/blob/main/notebooks/iris.ipynb>`_.
   The figures below are its outputs.


1. Load and standardize the data
--------------------------------

The BMU search compares raw feature distances, so standardizing is essential.

.. code-block:: python

   import torch
   from sklearn.datasets import load_iris
   from sklearn.preprocessing import StandardScaler

   bunch = load_iris()
   features = torch.tensor(
       StandardScaler().fit_transform(bunch.data), dtype=torch.float32
   )
   targets = torch.tensor(bunch.target, dtype=torch.long)   # 0, 1, 2
   feature_names = list(bunch.feature_names)


2. Train the SOM
----------------

.. code-block:: python

   from torchsom import SOM

   som = SOM(
       x=25,
       y=15,
       num_features=features.shape[1],
       epochs=100,
       batch_size=16,
       sigma=1.45,
       learning_rate=0.95,
       neighborhood_order=3,
       topology="rectangular",
       initialization_mode="pca",
       random_seed=42,
   )
   som.initialize_weights(data=features, mode=som.initialization_mode)
   q_errors, t_errors = som.fit(data=features)


3. Check convergence
--------------------

.. code-block:: python

   from torchsom import SOMVisualizer

   viz = SOMVisualizer(som=som)
   viz.plot_training_errors(
       quantization_errors=q_errors, topographic_errors=t_errors
   )

.. image:: /_static/results/iris/rectangular/training_errors.png
   :width: 600px
   :align: center
   :alt: Iris training curve

Both errors fall and flatten — training is long enough.


4. Inspect the map structure
----------------------------

The U-matrix exposes cluster boundaries; the hit map shows where the data lands.

.. code-block:: python

   viz.plot_distance_map(
       distance_metric=som.distance_fn_name,
       neighborhood_order=som.neighborhood_order,
   )
   viz.plot_hit_map(data=features)

.. list-table::
   :widths: 50 50

   * - .. image:: /_static/results/iris/rectangular/distance_map.png
          :width: 100%
          :alt: Iris U-matrix
     - .. image:: /_static/results/iris/rectangular/hit_map.png
          :width: 100%
          :alt: Iris hit map


5. Classification map
---------------------

Build the BMU→sample map once, then color each neuron by its dominant class.

.. code-block:: python

   bmus_map = som.build_map("bmus_data", data=features)
   viz.plot_classification_map(
       bmus_data_map=bmus_map,
       data=features,
       target=targets,
       neighborhood_order=som.neighborhood_order,
   )

.. image:: /_static/results/iris/rectangular/classification_map.png
   :width: 600px
   :align: center
   :alt: Iris classification map

*Iris setosa* separates cleanly, while *versicolor* and *virginica* share a boundary —
exactly the overlap known in this dataset, recovered here without supervision.


6. Component planes
-------------------

One heat map per feature reveals which features drive the separation.

.. code-block:: python

   viz.plot_component_planes(component_names=feature_names)

.. list-table::
   :widths: 50 50

   * - .. image:: /_static/results/iris/rectangular/component_planes/Petal_Length.png
          :width: 100%
          :alt: Petal length component plane
     - .. image:: /_static/results/iris/rectangular/component_planes/Petal_Width.png
          :width: 100%
          :alt: Petal width component plane
   * - .. image:: /_static/results/iris/rectangular/component_planes/Sepal_Length.png
          :width: 100%
          :alt: Sepal length component plane
     - .. image:: /_static/results/iris/rectangular/component_planes/Sepal_Width.png
          :width: 100%
          :alt: Sepal width component plane

Petal length and width vary together across the grid and align with the class
regions, confirming they are the most discriminative features.


Hexagonal variant
-----------------

Set ``topology="hexagonal"`` for the same analysis on a hexagonal grid; the visualizer
renders hexagon cells automatically:

.. image:: /_static/results/iris/hexagonal/classification_map.png
   :width: 600px
   :align: center
   :alt: Iris classification map on a hexagonal grid


Next steps
----------

- :doc:`wine` — A higher-dimensional classification example
- :doc:`boston_housing` — From classification to regression
- :doc:`../user_guide/visualization_help` — Every plot explained
