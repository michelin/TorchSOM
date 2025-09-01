Frequently Asked Questions
==========================

General Questions
-----------------

What is TorchSOM?
~~~~~~~~~~~~~~~~~

TorchSOM is a modern PyTorch-based implementation of Self-Organizing Maps (SOMs), designed for efficient training and comprehensive visualization of high-dimensional data clustering and analysis.

How does TorchSOM differ from other SOM implementations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TorchSOM offers several advantages:

- **GPU acceleration** through PyTorch
- **Modern Python practices** with type hints and Pydantic validation
- **Comprehensive visualization suite** with matplotlib integration
- **Flexible architecture** supporting multiple SOM variants

Installation and Setup
----------------------

Which Python versions are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using Python 3.9+.

Do I need a GPU to use TorchSOM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, TorchSOM works on both CPU and GPU.
However, GPU acceleration significantly improves training speed for large datasets and maps.
We recommend using a GPU for training.

Data Preprocessing
------------------

Should I always normalize my data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, normalization is crucial because:

- Features with larger scales dominate the distance calculation
- SOM learning is sensitive to feature magnitudes
- StandardScaler or MinMaxScaler from scikit-learn  both work well

What about categorical features?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SOMs operate exclusively on numerical data. Therefore, it is essential to convert any categorical features into a numerical format before using them with TorchSOM. Common strategies include:

1. **One-hot encoding** for nominal (unordered) categories
2. **Ordinal encoding** for ordered categories
3. **Target or frequency encoding** for high-cardinality categories

If your dataset contains a mix of numerical and categorical features, ensure all features are numerically encoded prior to training.

Similarly, when visualizing classification or label maps, assign numerical levels to each class or category to enable proper mapping and interpretation in the visualization outputs.

Performance and Optimization
----------------------------

My training is very slow. How can I speed it up?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try these optimizations:

1. **Enable GPU**: Use ``device="cuda"`` if available
2. **Increase batch size**: Try 64, 128, or 256
3. **Reduce map size**: Start smaller and scale up
4. **Use PCA initialization**: ``initialization_mode="pca"``
5. **Reduce epochs**: Monitor convergence and stop early

How much memory does TorchSOM use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory usage depends on:

- **Map size**: O(x × y × num_features)
- **Batch size**: Larger batches use more memory
- **Data size**: Keep datasets in reasonable sizes

For large datasets, consider:

- Processing in batches
- Using CPU instead of GPU
- Reducing precision (float32 vs float64)

Visualization Issues
--------------------

Why are some neurons white in my visualizations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

White neurons typically indicate:

- **Unactivated neurons**: No data points assigned as BMU
- **Zero values**: In some visualizations, zero values appear white
- **NaN values**: Missing or invalid calculations

This is normal for sparse data or oversized maps.

How do I interpret the distance map (D-Matrix)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the D-Matrix:

- **Light areas**: High distances between neighboring neurons (cluster boundaries)
- **Dark areas**: Low distances (within clusters)
- **Patterns**: Reveal cluster structure and boundaries

Can I customize the visualization colors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Use the VisualizationConfig:

.. code-block:: python

   from torchsom.visualization.config import VisualizationConfig

   config = VisualizationConfig(
       cmap="plasma",        # Use a different colormap
       figsize=(15, 10),     # Set larger figure size
       dpi=300               # Set higher resolution
   )

Advanced Topics
---------------

Can I use TorchSOM for time series data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TorchSOM is designed to work with tabular data, meaning any data type—such as time series, images, or text—can be used as long as it is represented in a tabular (2D array) format.
This typically means that each sample should be a fixed-length feature vector.

For time series or other complex data types, you can preprocess your data to obtain such representations.
Common approaches include extracting statistical features, flattening fixed-length windows, or generating embeddings (e.g., using autoencoders or other neural networks) before projecting them onto the SOM map.
As long as your data can be converted into a matrix of shape `[n_samples, n_features]`, it can be used with TorchSOM.

How do I implement custom distance functions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a function following the signature:

.. code-block:: python

   def custom_distance(data, weights):
       """
       Args:
           data: [batch_size, 1, 1, n_features]
           weights: [1, row_neurons, col_neurons, n_features]
       Returns:
           distances: [batch_size, row_neurons, col_neurons]
       """
       # Your custom distance calculation
       return distances

Can I save and load trained SOMs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, use PyTorch's standard mechanisms:

.. code-block:: python

   # Save
   torch.save(som.state_dict(), 'som_weights.pth')

   # Load
   som = SOM(x=10, y=10, num_features=4)
   som.load_state_dict(torch.load('som_weights.pth'))

Integration Questions
---------------------

How do I cite TorchSOM in my research?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please cite TorchSOM as:

.. code-block:: bibtex

    # Conference Paper
    @inproceedings{Berthier2025TorchSOM,
        title={torchsom: The Reference PyTorch Library for Self-Organizing Maps},
        author={Berthier, Louis},
        booktitle={Conference Name},
        year={2025}
    }

    # GitHub Repository
    @software{Berthier_TorchSOM_The_Reference_2025,
        author={Berthier, Louis},
        title={torchsom: The Reference PyTorch Library for Self-Organizing Maps},
        url={https://github.com/michelin/TorchSOM},
        version={1.0.0},
        year={2025}
    }

Getting Help
------------

Where can I get more help?
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Documentation**: Check our comprehensive guides `<https://opensource.michelin.io/TorchSOM/>`_
2. **GitHub Issues**: Report bugs and request features `<https://github.com/michelin/TorchSOM/issues>`_
3. **Notebooks**: See our tutorial notebooks `<https://github.com/michelin/TorchSOM/tree/main/notebooks>`_

How do I report a bug?
~~~~~~~~~~~~~~~~~~~~~~

Please include:

1. **TorchSOM version**: ``from torchsom.version import __version__``
2. **Python version**: ``python --version``
3. **PyTorch version**: ``torch.__version__``
4. **Operating system**: Linux/macOS/Windows
5. **Minimal reproduction example**
6. **Full error traceback**

Can I contribute to TorchSOM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! We welcome contributions:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add tests** for new functionality
4. **Submit** a pull request
5. **Follow** our coding standards

See our `contributing guide <https://github.com/michelin/TorchSOM/blob/main/CONTRIBUTING.md>`_ for details.
