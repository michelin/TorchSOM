Installation
============

Requirements
------------

Systems
~~~~~~~

TorchSOM requires:

- **Python**: 3.9 or higher
- **PyTorch**: 2.7 or higher
- **Operating System**: Linux, macOS, or Windows

GPU
~~~

For optimal performance, we recommend using a CUDA-compatible GPU with:

- **CUDA**: 11.0 or higher
- **GPU Memory**: 4GB or more recommended for large SOMs

Installation Methods
--------------------

Install from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install TorchSOM is using pip:

.. code-block:: bash

   pip install torchsom

This will install TorchSOM with all required dependencies.

Install from Source
~~~~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   # Clone the TorchSOM repository
   git clone https://github.com/michelin/TorchSOM.git
   cd TorchSOM

   # For standard users: install the main package
   pip install -e .

   # For contributors and developers: install with development dependencies
   pip install -e ".[dev]"

Ensure GPU Support
~~~~~~~~~~~~~~~~~~

To ensure GPU acceleration, install PyTorch with CUDA support:

.. code-block:: bash

   # For CUDA 11.8 (check your CUDA version)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install torchsom

Verification
------------

Verify your installation by running:

.. code-block:: python

   import torchsom
   print(f"TorchSOM version: {torchsom.__version__}")
   
   # Check GPU availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU device: {torch.cuda.get_device_name()}")

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

- **torch**: PyTorch deep learning framework
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **pydantic**: Data validation and configuration
- **tqdm**: Progress bars

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

- **rich**: Enhanced and colored console output (progress bars, tracebacks)
- **notebook**: Jupyter Notebook support for interactive examples and tutorials
- **pandas**: Advanced data manipulation and analysis (used in some tutorials)
- **scikit-learn**: Additional machine learning utilities and benchmarking tools
- **openpyxl**: Excel file support for data import/export in some examples
- **black**: Code formatting (for contributors)
- **isort**: Import sorting (for contributors)
- **pytest**, **pytest-cov**, **pytest-html**: Testing and coverage reporting
- **certifi**: Certificate management (dependency for some environments)
- **typing_extensions**: Backport of Python typing features for compatibility
- **sphinx**, **sphinx-rtd-theme**, **sphinx-autodoc-typehints**, **sphinx-copybutton**: Documentation building and formatting

Getting Help
------------

If you encounter installation issues:

1. Check the `troubleshooting guide <../troubleshooting.html>`_
2. Search existing `GitHub Issues <https://github.com/michelin/TorchSOM/issues>`_
3. Create a new issue with your system details and error message

Next Steps
----------

Once installed, continue with:

- :doc:`quickstart` - Your first SOM in 5 minutes
- :doc:`basic_concepts` - Understanding SOM fundamentals 