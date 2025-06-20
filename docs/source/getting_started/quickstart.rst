Quick Start Guide
=================

This guide will get you up and running with TorchSOM in just a few minutes!

Your First SOM
--------------

Let's create and train your first Self-Organizing Map:

.. code-block:: python

   import torch
   from torchsom import SOM
   
   # Generate some sample data (1000 samples, 4 features)
   data = torch.randn(1000, 4)
   
   # Create a 10x10 SOM
   som = SOM(
       x=10,                # Height of the map
       y=10,                # Width of the map
       num_features=4,      # Number of input features
       epochs=50,           # Training iterations
       learning_rate=0.5,   # Learning rate
       sigma=2.0            # Neighborhood radius
   )
   
   # Train the SOM
   som.initialize_weights(data=data, mode="pca")
   QE, TE = som.fit(data=data)
   
   print("Training completed!")
   print(f"Final quantization error: {QE[-1]:.4f}")
   print(f"Final topographic error: {TE[-1]:.4f}")

Basic Visualization
-------------------

Visualize your trained SOM:

.. code-block:: python

   from torchsom.visualization import SOMVisualizer
   
   # Create visualizer
   visualizer = SOMVisualizer(som)
   
   # Plot distance map (U-Matrix)
   visualizer.plot_distance_map()
   
   # Plot hit map (activation frequency)
   visualizer.plot_hit_map(data)
   
   # Plot training errors
   visualizer.plot_training_errors(quantization_errors, topographic_errors)

Common Parameters
-----------------

Quick reference for the most important SOM parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``x``
     - *Required*
     - Number of rows (neurons along the x-axis) in the SOM grid.
   * - ``y``
     - *Required*
     - Number of columns (neurons along the y-axis) in the SOM grid.
   * - ``num_features``
     - *Required*
     - Number of input features (dimensionality of the data).
   * - ``epochs``
     - 10
     - Number of training epochs (full passes over the data).
   * - ``batch_size``
     - 5
     - Number of samples per training batch.
   * - ``sigma``
     - 1.0
     - Initial neighborhood radius (spread of update influence).
   * - ``learning_rate``
     - 0.5
     - Initial learning rate (strength of weight updates, typically 0.1–1.0).
   * - ``neighborhood_order``
     - 1
     - Number of neighbors to consider for neighborhood updates.
   * - ``topology``
     - "rectangular"
     - Grid topology: ``"rectangular"`` or ``"hexagonal"``.
   * - ``lr_decay_function``
     - "asymptotic_decay"
     - Learning rate decay schedule (e.g., ``"asymptotic_decay"``, ``"linear_decay"``, ``"exponential_decay"``).
   * - ``sigma_decay_function``
     - "asymptotic_decay"
     - Sigma (neighborhood radius) decay schedule (same options as above).
   * - ``neighborhood_function``
     - "gaussian"
     - Function for neighborhood influence (e.g., ``"gaussian"``, ``"bubble"``, ``"mexican_hat"``).
   * - ``distance_function``
     - "euclidean"
     - Distance metric for BMU search (e.g., ``"euclidean"``, ``"manhattan"``, ``"cosine"``).
   * - ``initialization_mode``
     - "random"
     - Weight initialization method (``"random"``, ``"pca"``, or custom).
   * - ``device``
     - "cuda" if available, else "cpu"
     - Device for computation: ``"cpu"``, ``"cuda"``, or explicit device string.
   * - ``random_seed``
     - 42
     - Random seed for reproducibility.

Complete Example
----------------

Here's a complete example with data preprocessing and multiple visualizations:

.. code-block:: python

   import torch
   from sklearn.datasets import make_blobs
   from torchsom import SOM, SOMVisualizer

   # Check GPU availability
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using device: {device}")
   
   # Generate clustered data
   X, y = make_blobs(n_samples=1000, centers=4, n_features=4, 
                     cluster_std=1.5, random_state=42)
   data = torch.tensor(X, dtype=torch.float32, device=device)
   labels = torch.tensor(y, device=device)
   
   # Normalize data (recommended)
   data = (data - data.mean(dim=0)) / data.std(dim=0)
   
   # Create and configure SOM
   som = SOM(
      x=25,
      y=15,
      num_features=data.shape[1],
      sigma=1.75,
      learning_rate=0.95,
      neighborhood_order=3,
      epochs=100,
      batch_size=16,
      topology="rectangular",
      distance_function="euclidean",
      neighborhood_function="gaussian",      
      lr_decay_function="asymptotic_decay",
      sigma_decay_function="asymptotic_decay",
      initialization_mode="pca",
      device=device,
      random_seed=42,
  ) 
   
   # Train the SOM
   q_errors, t_errors = som.fit(data)
   
   # Create visualizer
   viz = SOMVisualizer(som)
   
   # Generate all visualizations
   viz.plot_all(
       quantization_errors=q_errors,
       topographic_errors=t_errors,
       data=data,
       target=labels,
       save_path="som_results"
   )

What's Next?
-----------

Now that you've created your first SOM, explore:

- :doc:`basic_concepts` - Understand how SOMs work
- :doc:`../user_guide/som_training` - Advanced training techniques
- :doc:`../user_guide/visualization` - Comprehensive visualization guide

.. Tips for Success
.. ---------------

.. 1. **Normalize your data**: Always normalize features to similar scales
.. 2. **Choose appropriate map size**: Roughly 5-10 times fewer neurons than data points
.. 3. **Tune learning parameters**: Start with defaults, then adjust based on results
.. 4. **Use GPU**: For large datasets, GPU acceleration provides significant speedup
.. 5. **Visualize results**: Always examine the trained map with visualizations 