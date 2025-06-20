Basic Concepts
==============

This page introduces the fundamental concepts behind Self-Organizing Maps (SOMs) and how they work.

What is a Self-Organizing Map?
------------------------------

A **Self-Organizing Map (SOM)**, also known as a Kohonen map, is an unsupervised neural network algorithm that:

- **Clusters** similar data points together
- **Reduces dimensionality** by mapping high-dimensional data to a lower-dimensional grid, usually 2D
- **Preserves topology** by keeping similar data points close together on the map
- **Visualizes patterns** in complex, high-dimensional datasets

**Key Characteristics**: 

- **Unsupervised**: No labeled data required
- **Competitive learning**: Neurons compete to represent input data
- **Topology preservation**: Maintains neighborhood relationships
- **Dimensionality reduction**: Maps N-dimensional data to 2D grid

How SOMs Work
-------------

The SOM Algorithm
~~~~~~~~~~~~~~~~~

1. **Initialize** weight vectors randomly for each neuron
2. **Present** input data to the network
3. **Find** the Best Matching Unit (BMU) - the neuron most similar to input
4. **Update** the BMU and its neighbors to be more similar to the input
5. **Repeat** until convergence or maximum iterations reached

.. .. image:: ../../../assets/som_algorithm_flow.png
..    :alt: SOM Algorithm Flow
..    :width: 600px
..    :align: center

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

**Distance Calculation**
The similarity between input **x** and neuron **w** is typically measured using Euclidean distance:

.. math::
   d(x, w_i) = \sqrt{\sum_{j=1}^{n} (x_j - w_{i,j})^2}

**Weight Update Rule**
The weight update follows:

.. math::
   w_i(t+1) = w_i(t) + \alpha(t) \cdot h_{BMU,i}(t) \cdot (x(t) - w_i(t))

Where:
- :math:`\alpha(t)` is the learning rate at time t
- :math:`h_{BMU,i}(t)` is the neighborhood function
- :math:`x(t)` is the input vector at time t

Core Components
---------------

1. Grid Topology
~~~~~~~~~~~~~~~~

SOMs organize neurons in a grid structure:

**Rectangular Grid**
   - Each neuron has up to 8 neighbors
   - Simple, intuitive visualization
   - Good for most applications

**Hexagonal Grid**
   - Each neuron has up to 6 neighbors
   - More uniform neighborhood distances
   - Better for circular/radial patterns

2. Neighborhood Function
~~~~~~~~~~~~~~~~~~~~~~~

Determines how much each neuron is affected by the BMU:

**Gaussian** (Most Common)
   .. math::
      h_{BMU,i}(t) = \exp\left(-\frac{d_{BMU,i}^2}{2\sigma(t)^2}\right)

**Bubble**
   Step function - neurons within radius are updated equally

**Triangle**
   Linear decay from BMU to neighborhood boundary

3. Learning Rate Decay
~~~~~~~~~~~~~~~~~~~~~

Controls how much weights change during training:

**Asymptotic Decay**
   .. math::
      \alpha(t) = \frac{\alpha_0}{1 + t/T}

**Linear Decay**
   .. math::
      \alpha(t) = \alpha_0 \cdot (1 - t/T)

4. Distance Functions
~~~~~~~~~~~~~~~~~~~~

Different ways to measure similarity:

- **Euclidean**: Standard geometric distance
- **Cosine**: Measures angle between vectors
- **Manhattan**: Sum of absolute differences
- **Chebyshev**: Maximum absolute difference

5. Quality Metrics
~~~~~~~~~~~~~~~~~~

- **Quantization Error**: Average distance between data points and their BMUs. Lower is better, measures how well the map represents the data.
- **Topographic Error**: Percentage of data points whose BMU and second-BMU are not neighbors. Lower is better, measures topology preservation.

Strengths and Weaknesses
------------------------

Advantages
~~~~~~~~~

- **No assumptions** about data distribution
- **Topology preservation** maintains relationships
- **Intuitive visualization** of complex data
- **Unsupervised learning** - no labels needed

Limitations
~~~~~~~~~~

- **Computationally expensive** for large datasets
- **Parameter sensitive** - requires tuning
- **Interpretation challenges** for very high dimensions

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~

1. **Normalize features** to similar scales
2. **Remove highly correlated** features
3. **Handle missing values** appropriately
4. **Consider dimensionality reduction** for very high dimensions

Parameter Selection
~~~~~~~~~~~~~~~~~~

1. **Experiment with different** topologies and functions
2. **Monitor training progress** with error curves to guide parameter choice

Interpretation
~~~~~~~~~~~~~

1. **Use multiple visualizations** to understand the map
2. **Combine with domain knowledge** for meaningful insights
3. **Validate findings** with other analysis methods
4. **Document parameter choices** for reproducibility

Next Steps
----------

Now that you understand the basics, explore:

- :doc:`../user_guide/visualization_help` - Visualization techniques 