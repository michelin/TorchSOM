Troubleshooting
===============

This guide helps you resolve common issues when using TorchSOM.

Installation Issues
-------------------

Package Issues
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    ImportError: No module named 'torchsom'

**Problem**: TorchSOM is not installed or not in Python path.

**Solutions**:

1. Install TorchSOM:
   
   .. code-block:: bash
   
      pip install torchsom

2. If using conda environment, make sure it's activated:
   
   .. code-block:: bash
   
      conda activate your_environment
      pip install torchsom

3. Check installation:
   
   .. code-block:: python
   
      import torchsom
      print(torchsom.__version__)

CUDA/GPU Issues
~~~~~~~~~~~~~~~

.. code-block:: bash

    RuntimeError: CUDA out of memory

**Problem**: GPU memory is exhausted during training.

**Solutions**:

1. **Reduce batch size**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, batch_size=16)  # Smaller batch

2. **Use CPU instead**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, device="cpu")

3. **Clear GPU cache**:
   
   .. code-block:: python
   
      import torch
      torch.cuda.empty_cache()

4. **Reduce map size**:
   
   .. code-block:: python
   
      som = SOM(x=8, y=8, num_features=4)  # Smaller SOM

CUDA not available
~~~~~~~~~~~~~~~~~~

**Problem**: ``torch.cuda.is_available()`` returns ``False``.

**Diagnostic steps**:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"PyTorch version: {torch.__version__}")

**Solutions**:

1. **Install CUDA-enabled PyTorch**:
   
   .. code-block:: bash
   
      # For CUDA 11.8
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2. **Check CUDA installation**:
   
   .. code-block:: bash
   
      nvidia-smi
      nvcc --version

3. **Use CPU if no GPU available**:
   
   .. code-block:: python
   
      device = "cuda" if torch.cuda.is_available() else "cpu"
      som = SOM(x=10, y=10, num_features=4, device=device)

Training Problems
-----------------

Training doesn't converge
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Quantization error doesn't decrease or fluctuates wildly.

**Diagnostic**:

.. code-block:: python

   # Monitor training progress
   q_errors, t_errors = som.fit(data)
   
   import matplotlib.pyplot as plt
   plt.plot(q_errors)
   plt.title('Quantization Error')
   plt.show()

**Common causes and solutions**:

1. **Learning rate too high**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, learning_rate=0.1)  # Lower LR

2. **Data not normalized**:
   
   .. code-block:: python
   
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      data_scaled = scaler.fit_transform(data)
      data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

3. **Poor initialization**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, initialization_mode="pca")

4. **Map too large**:
   
   .. code-block:: python
   
      # Rule of thumb: 5-10x fewer neurons than data points
      data_size = len(data)
      map_size = int(np.sqrt(data_size / 7))
      som = SOM(x=map_size, y=map_size, num_features=4)

Very slow training
~~~~~~~~~~~~~~~~~~

**Problem**: Training takes much longer than expected.

**Performance optimization**:

1. **Enable GPU acceleration**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, device="cuda")

2. **Increase batch size**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, batch_size=128)

3. **Use PCA initialization**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, initialization_mode="pca")

4. **Reduce epochs if acceptable**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, epochs=50)

5. **Profile your code**:
   
   .. code-block:: python
   
      import time
      start_time = time.time()
      som.fit(data)
      print(f"Training time: {time.time() - start_time:.2f} seconds")

NaN values in results
~~~~~~~~~~~~~~~~~~~~

**Problem**: Getting NaN values in errors or visualizations.

**Diagnostic**:

.. code-block:: python

   # Check for NaN in data
   print(f"NaN in data: {torch.isnan(data).any()}")
   
   # Check SOM weights
   print(f"NaN in weights: {torch.isnan(som.weights).any()}")

**Solutions**:

1. **Check input data**:
   
   .. code-block:: python
   
      # Remove NaN values
      data_clean = data[~torch.isnan(data).any(dim=1)]
      
      # Or impute missing values
      from sklearn.impute import SimpleImputer
      imputer = SimpleImputer(strategy='mean')
      data_imputed = imputer.fit_transform(data.numpy())
      data_clean = torch.tensor(data_imputed, dtype=torch.float32)

2. **Reduce learning rate**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, learning_rate=0.1)

3. **Check for inf values**:
   
   .. code-block:: python
   
      data = torch.clamp(data, min=-1e6, max=1e6)  # Clip extreme values

Visualization Issues
--------------------

Empty or white visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Visualizations appear blank or mostly white.

**Possible causes**:

1. **No data passed to visualization**:
   
   .. code-block:: python
   
      # Make sure to pass data to hit map
      viz.plot_hit_map(data=data_tensor)

2. **All neurons have same values**:
   
   .. code-block:: python
   
      # Check weight variance
      weights = som.weights.detach().cpu().numpy()
      print(f"Weight std: {np.std(weights)}")

3. **Colormap issues**:
   
   .. code-block:: python
   
      # Try different colormap
      from torchsom.visualization import VisualizationConfig
      config = VisualizationConfig(cmap="viridis")
      viz = SOMVisualizer(som, config=config)

Figures not displaying
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Plots don't show up in Jupyter notebooks or scripts.

**Solutions**:

1. **For Jupyter notebooks**:
   
   .. code-block:: python
   
      %matplotlib inline
      import matplotlib.pyplot as plt

2. **For scripts**:
   
   .. code-block:: python
   
      import matplotlib.pyplot as plt
      # ... create plots ...
      plt.show()  # Don't forget this

3. **Save figures instead**:
   
   .. code-block:: python
   
      viz.plot_distance_map(save_path="results", fig_name="distance_map")

Poor visualization quality
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Plots look pixelated or unclear.

**Solutions**:

1. **Increase resolution**:
   
   .. code-block:: python
   
      config = VisualizationConfig(dpi=300)
      viz = SOMVisualizer(som, config=config)

2. **Larger figure size**:
   
   .. code-block:: python
   
      config = VisualizationConfig(figsize=(12, 10))
      viz = SOMVisualizer(som, config=config)

3. **Better colormap**:
   
   .. code-block:: python
   
      config = VisualizationConfig(cmap="plasma")
      viz = SOMVisualizer(som, config=config)

Data Issues
-----------

Poor clustering results
~~~~~~~~~~~~~~~~~~~~~~

**Problem**: SOM doesn't find meaningful clusters.

**Diagnostic steps**:

1. **Visualize raw data**:
   
   .. code-block:: python
   
      from sklearn.decomposition import PCA
      from sklearn.manifold import TSNE
      
      # PCA visualization
      pca = PCA(n_components=2)
      data_pca = pca.fit_transform(data.numpy())
      plt.scatter(data_pca[:, 0], data_pca[:, 1])
      plt.title('Data in PCA space')
      plt.show()

2. **Check data distribution**:
   
   .. code-block:: python
   
      print(f"Data shape: {data.shape}")
      print(f"Data mean: {data.mean(dim=0)}")
      print(f"Data std: {data.std(dim=0)}")

3. **Compare with K-means**:
   
   .. code-block:: python
   
      from sklearn.cluster import KMeans
      kmeans = KMeans(n_clusters=3)
      kmeans_labels = kmeans.fit_predict(data.numpy())

**Solutions**:

1. **Better preprocessing**:
   
   .. code-block:: python
   
      # Remove outliers
      from sklearn.preprocessing import RobustScaler
      scaler = RobustScaler()
      data_scaled = scaler.fit_transform(data.numpy())

2. **Feature selection**:
   
   .. code-block:: python
   
      # Remove highly correlated features
      import pandas as pd
      df = pd.DataFrame(data.numpy())
      corr_matrix = df.corr().abs()
      # Remove features with correlation > 0.95

3. **Adjust SOM parameters**:
   
   .. code-block:: python
   
      som = SOM(
          x=15, y=15,  # Larger map
          num_features=data.shape[1],
          epochs=200,  # More training
          learning_rate=0.2,
          sigma=3.0  # Larger neighborhood
      )

Configuration Errors
--------------------

ValidationError from Pydantic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Configuration validation fails.

**Example error**:

.. code-block:: text

   ValidationError: 1 validation error for SOMConfig
   learning_rate
     ensure this value is greater than 0 (type=value_error.number.not_gt)

**Solution**:

.. code-block:: python

   from torchsom.configs import SOMConfig
   from pydantic import ValidationError
   
   try:
       config = SOMConfig(
           x=10, y=10,
           learning_rate=0.3,  # Must be > 0
           sigma=1.0,          # Must be > 0
           epochs=100          # Must be >= 1
       )
   except ValidationError as e:
       print("Configuration errors:")
       for error in e.errors():
           print(f"- {error['loc'][0]}: {error['msg']}")

Parameter compatibility issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Certain parameter combinations don't work.

**Common incompatibilities**:

1. **Sigma too large for map size**:
   
   .. code-block:: python
   
      # Problem: sigma=10 on 5x5 map
      som = SOM(x=5, y=5, num_features=4, sigma=2.0)  # Better

2. **Batch size larger than dataset**:
   
   .. code-block:: python
   
      batch_size = min(64, len(data))
      som = SOM(x=10, y=10, num_features=4, batch_size=batch_size)

Memory Issues
-------------

Memory usage too high
~~~~~~~~~~~~~~~~~~~~~

**Problem**: TorchSOM uses too much RAM or GPU memory.

**Memory usage breakdown**:
- SOM weights: ``x * y * num_features * 4 bytes`` (float32)
- Batch data: ``batch_size * num_features * 4 bytes``
- Distance calculations: ``batch_size * x * y * 4 bytes``

**Solutions**:

1. **Reduce map size**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4)  # Instead of 20x20

2. **Smaller batch size**:
   
   .. code-block:: python
   
      som = SOM(x=10, y=10, num_features=4, batch_size=32)

3. **Use CPU for large maps**:
   
   .. code-block:: python
   
      som = SOM(x=50, y=50, num_features=4, device="cpu")

4. **Process data in chunks**:
   
   .. code-block:: python
   
      # For very large datasets
      chunk_size = 1000
      for i in range(0, len(data), chunk_size):
           chunk = data[i:i+chunk_size]
           som.fit(chunk)  # Incremental training

Memory leaks
~~~~~~~~~~~

**Problem**: Memory usage increases over time.

**Solutions**:

1. **Clear GPU cache periodically**:
   
   .. code-block:: python
   
      import torch
      torch.cuda.empty_cache()

2. **Use context managers**:
   
   .. code-block:: python
   
      with torch.no_grad():
           # Inference operations
           bmus = som.identify_bmus(data)

3. **Delete large variables**:
   
   .. code-block:: python
   
      del large_data_tensor
      torch.cuda.empty_cache()

Getting Help
------------

Diagnostic Information
~~~~~~~~~~~~~~~~~~~~~

When reporting issues, please include:

.. code-block:: python

   import torchsom
   import torch
   import sys
   import platform
   
   print("=== Diagnostic Information ===")
   print(f"TorchSOM version: {torchsom.__version__}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"Python version: {sys.version}")
   print(f"Platform: {platform.platform()}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU count: {torch.cuda.device_count()}")
       for i in range(torch.cuda.device_count()):
           print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

Creating Minimal Examples
~~~~~~~~~~~~~~~~~~~~~~~~

For bug reports, create minimal reproducible examples:

.. code-block:: python

   import torch
   from torchsom import SOM
   
   # Minimal data
   data = torch.randn(100, 4)
   
   # Minimal SOM
   som = SOM(x=5, y=5, num_features=4, epochs=10)
   
   # Show the problem
   try:
       som.fit(data)
   except Exception as e:
       print(f"Error: {e}")
       raise

Where to Get Help
~~~~~~~~~~~~~~~~

1. **Documentation**: Check our comprehensive guides first
2. **FAQ**: Review the :doc:`faq` for common questions
3. **GitHub Issues**: Report bugs with minimal examples
4. **GitHub Discussions**: Ask questions and share experiences
5. **Stack Overflow**: Tag questions with ``torchsom`` and ``pytorch``

Debug Mode
~~~~~~~~~~

Enable debug logging for more detailed information:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Your TorchSOM code here
   som = SOM(x=10, y=10, num_features=4)
   som.fit(data) 