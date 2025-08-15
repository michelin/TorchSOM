SOM Visualization Guide
=======================

This comprehensive guide covers all visualization capabilities available in TorchSOM for analyzing and interpreting Self-Organizing Maps effectively.

Overview
--------

TorchSOM provides a rich set of visualization tools through the :class:`SOMVisualizer` class, supporting both rectangular and hexagonal topologies. All visualizations are designed to help you understand:

- **Training Progress**: How well your SOM is learning over time
- **Data Distribution**: How input data maps onto the SOM grid
- **Topology Preservation**: Whether neighborhood relationships are maintained
- **Feature Representation**: How individual features are distributed across neurons
- **Cluster Structure**: Identification of natural groupings in your data

Quick Start
-----------

Basic Visualization Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchsom import SOM
   from torchsom.visualization import SOMVisualizer, VisualizationConfig
   import torch

   # Train a SOM
   data = torch.randn(1000, 4)
   som = SOM(x=20, y=15, num_features=4, epochs=50)
   som.initialize_weights(data=data, mode="pca")
   q_errors, t_errors = som.fit(data)

   # Create visualizer with default configuration
   visualizer = SOMVisualizer(som=som)

   # Generate all visualizations at once
   visualizer.plot_all(
       quantization_errors=q_errors,
       topographic_errors=t_errors,
       data=data,
       save_path="som_results"
   )

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

The :class:`VisualizationConfig` class provides comprehensive customization options:

.. code-block:: python

   config = VisualizationConfig(
       figsize=(12, 8),                    # Figure size in inches
       fontsize={                          # Font sizes for different elements
           "title": 16,
           "axis": 13,
           "legend": 11
       },
       fontweight={                        # Font weights
           "title": "bold",
           "axis": "normal"
       },
       cmap="viridis",                     # Default colormap
       dpi=150,                            # Resolution for saved figures
       grid_alpha=0.3,                     # Grid transparency
       colorbar_pad=0.01,                  # Colorbar padding
       save_format="png",                  # Save format (png, pdf, eps, svg)
       hexgrid_size=None                   # Hexagonal grid size (auto if None)
   )

Visualization Types
-------------------

Training Errors
~~~~~~~~~~~~~~~

Monitors SOM learning progress by plotting quantization and topographic errors over epochs.

.. code-block:: python

   visualizer.plot_training_errors(
       quantization_errors=q_errors,
       topographic_errors=t_errors,
       save_path="results"
   )

**Interpretation:**

- **Quantization Error**: Measures how well the SOM represents the input data (lower is better)
- **Topographic Error**: Measures topology preservation (lower percentage is better)
- **Convergence**: Both errors should generally decrease and stabilize during training

.. image:: ../../../assets/michelin_training_errors.png
   :width: 400px
   :align: center
   :alt: Training Errors Example

Distance Map (D-Matrix)
~~~~~~~~~~~~~~~~~~~~~~~

The unified distance matrix shows the distance between each neuron and its neighbors, revealing cluster boundaries.

.. code-block:: python

   visualizer.plot_distance_map(save_path="results")

**Interpretation:**

- **Dark Regions**: Large distances between neighboring neurons (cluster boundaries)
- **Light Regions**: Small distances between neighboring neurons (within clusters)
- **Topology**: Works with both rectangular and hexagonal grids

.. image:: ../../../assets/michelin_dmatrix.png
   :width: 400px
   :align: center
   :alt: Distance Matrix Example

Hit Map
~~~~~~~

Shows the frequency of neuron activation, indicating how often each neuron was selected as the Best Matching Unit (BMU).

.. code-block:: python

   visualizer.plot_hit_map(data=data, save_path="results")

**Interpretation:**
- **Bright Areas**: Frequently activated neurons (high data density)
- **Dark Areas**: Rarely activated neurons (low data density or dead neurons)
- **Usage**: Identifies data distribution patterns and potential dead neurons

.. image:: ../../../assets/michelin_hitmap.png
   :width: 400px
   :align: center
   :alt: Hit Map Example

Component Planes
~~~~~~~~~~~~~~~~

Individual visualizations for each input feature dimension, showing how feature weights are distributed across the map.

.. code-block:: python

   # Plot all component planes
   feature_names = ["Temperature", "Pressure", "Flow_Rate", "Quality"]
   visualizer.plot_component_planes(
       component_names=feature_names,
       save_path="results"
   )

**Interpretation:**
- **One Plane per Feature**: Shows weight values for each input dimension
- **Pattern Analysis**: Reveals feature importance in different map regions
- **Correlation Detection**: Similar patterns indicate correlated features

.. image:: ../../../assets/michelin_cp12.png
   :width: 400px
   :align: center
   :alt: Component Plane of feature 12

Supervised Maps
~~~~~~~~~~~~~~~

Visualizations for supervised learning tasks, including both classification and regression, help interpret how target information is distributed across the SOM map.

Classification Case
^^^^^^^^^^^^^^^^^^^

Displays the most frequent class label assigned to each neuron, providing insight into class separation and cluster structure.

.. code-block:: python

   # Example: Visualizing class assignments (labels must be > 0)
   labels = torch.randint(1, 4, (1000,))
   visualizer.plot_classification_map(
       data=data,
       target=labels,
       save_path="results"
   )

**Interpretation:**

- **Color Coding**: Each color represents a different class label.
- **Cluster Identification**: Reveals spatial organization of classes on the map.
- **Decision Boundaries**: Boundaries between colors indicate class separation.

.. image:: ../../../assets/wine_classificationmap.png
   :width: 400px
   :align: center
   :alt: Classification Map Example

Regression Case
^^^^^^^^^^^^^^^

Analyzes the distribution of continuous target values (e.g., for regression tasks) using statistical summaries per neuron.

Mean Map
""""""""

Shows the average target value for samples mapped to each neuron.

.. code-block:: python

   # Example: Visualizing mean target values
   target_values = torch.randn(1000) * 10 + 50
   visualizer.plot_metric_map(
       data=data,
       target=target_values,
       reduction_parameter="mean",
       save_path="results"
   )

**Interpretation:**

- **Color Scale**: Indicates the mean target value per neuron.
- **Smooth Transitions**: Suggest good topology preservation.
- **Hot Spots**: Highlight neurons with extreme target values.

.. image:: ../../../assets/michelin_meanmap.png
   :width: 400px
   :align: center
   :alt: Mean Map Example

Standard Deviation Map
""""""""""""""""""""""

Shows the variability of target values for each neuron, useful for assessing prediction reliability.

.. code-block:: python

   visualizer.plot_metric_map(
       data=data,
       target=target_values,
       reduction_parameter="std",
       save_path="results"
   )

**Interpretation:**

- **Low Values**: Neurons with consistent (low-variance) target values—good for prediction.
- **High Values**: Neurons with variable (high-variance) target values—less reliable.
- **Quality Assessment**: Helps identify the most reliable neurons for regression tasks.

Advanced Visualizations
~~~~~~~~~~~~~~~~~~~~~~~

Score Map
^^^^^^^^^

Evaluates neuron representativeness using a composite score combining standard error and sample distribution.

.. code-block:: python

   visualizer.plot_score_map(
       data=data,
       target=target_values,
       save_path="results"
   )

**Interpretation:**

- **Lower Scores**: Better neuron representativeness
- **Formula**: ``std_neuron / sqrt(n_neuron) * log(N_data/n_neuron)``
- **Usage**: Identifies most reliable neurons for analysis

Rank Map
^^^^^^^^

Ranks neurons based on their target value standard deviations.

.. code-block:: python

   visualizer.plot_rank_map(
       data=data,
       target=target_values,
       save_path="results"
   )

**Interpretation:**

- **Rank 1**: Lowest standard deviation (best predictive neurons)
- **Higher Ranks**: Increasing standard deviation
- **Selection**: Use top-ranked neurons for reliable predictions

Advanced Usage Examples
-----------------------

Batch Visualization Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate all visualizations with selective control
   visualizer.plot_all(
       quantization_errors=q_errors,
       topographic_errors=t_errors,
       data=data,
       target=target_values,
       component_names=["Feature_1", "Feature_2", "Feature_3", "Feature_4"],
       save_path="complete_analysis",
       training_errors=True,
       distance_map=True,
       hit_map=True,
       score_map=True,
       rank_map=True,
       metric_map=True,
       component_planes=True
   )

Custom Colormap Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Using custom colormaps for specific visualizations
   visualizer.plot_grid(
       map=som.build_distance_map(),
       title="Custom Distance Map",
       colorbar_label="Distance",
       filename="custom_dmatrix",
       save_path="results",
       cmap="RdYlBu_r",  # Red-Yellow-Blue reversed
       log_scale=False
   )

Troubleshooting
~~~~~~~~~~~~~~~

**White Cells in Visualizations**:
    - Indicates neurons with zero values or NaN
    - Check for dead neurons in hit map
    - Verify data preprocessing and normalization

**Memory Issues**:
    - Reduce batch size in visualization functions
    - Use CPU-only mode for very large SOMs
    - Clear GPU cache with ``torch.cuda.empty_cache()``

**Topology Preservation**:
    - High topographic error indicates poor topology preservation
    - Consider adjusting learning rate, sigma, or training epochs
    - Use PCA initialization for better convergence

References
----------

For more examples and detailed usage, see:

- `TorchSOM Examples <https://github.com/michelin/TorchSOM/tree/main/notebooks>`_
- `API Documentation <../api/visualization.html>`_
- `Getting Started Guide <../getting_started/quickstart.html>`_
