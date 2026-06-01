Tutorials
=========

End-to-end walkthroughs on public datasets. Each tutorial loads and standardizes the
data, trains a SOM, and reads the result through the visualization suite. Every figure
on these pages is an actual output of the matching notebook in the
`notebooks/ <https://github.com/michelin/TorchSOM/tree/main/notebooks>`_ directory, so
you can reproduce them end to end.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Iris — Classification
      :link: iris
      :link-type: doc

      The classic 4-feature, 3-class dataset. Train a SOM and read the classification
      map and component planes.

   .. grid-item-card:: Wine — Classification
      :link: wine
      :link-type: doc

      13 chemical features, 3 cultivars. A higher-dimensional classification example.

   .. grid-item-card:: Boston Housing — Regression
      :link: boston_housing
      :link-type: doc

      Map a continuous target with mean, std, score, and rank maps.

   .. grid-item-card:: Energy Efficiency — Multi-target
      :link: energy_efficiency
      :link-type: doc

      Two regression targets (heating and cooling load) on one map.

   .. grid-item-card:: Clustering — Synthetic blobs
      :link: clustering_walkthrough
      :link-type: doc

      Cluster the neurons and use the elbow, silhouette, and comparison diagnostics.


.. list-table:: Datasets at a glance
   :header-rows: 1
   :widths: 26 14 14 46

   * - Tutorial
     - Task
     - Features
     - Visualizations highlighted
   * - :doc:`iris`
     - Classification
     - 4
     - Classification map, component planes
   * - :doc:`wine`
     - Classification
     - 13
     - Classification map, U-matrix
   * - :doc:`boston_housing`
     - Regression
     - 13
     - Mean / std / score / rank maps
   * - :doc:`energy_efficiency`
     - Regression (×2)
     - 8
     - Per-target metric maps
   * - :doc:`clustering_walkthrough`
     - Clustering
     - 4
     - Cluster map, elbow, silhouette, comparison
