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
   pip install -e ".[all]"

Ensure GPU Support
~~~~~~~~~~~~~~~~~~

To ensure GPU acceleration, install a CUDA-enabled PyTorch per the official selector, then install TorchSOM:

.. code-block:: bash

   # Select the right command for your system at https://pytorch.org/get-started/locally/
   # Example (CUDA 11.8):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install torchsom

Verification
------------

Verify your installation by running:

.. code-block:: python

   import torchsom
   from torchsom.version import __version__
   print(f"TorchSOM version: {__version__}")

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
- **tqdm**: Progress bars
- **pydantic**: Data validation and configuration
- **hdbscan**: Hierarchical Density-Based Spatial Clustering of Applications with Noise
- **scikit-learn**: Machine learning utilities and benchmarking tools

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

The following optional dependency groups can be installed via pip, e.g.:

- ``pip install .[dev]``
- ``pip install .[tests]``
- ``pip install .[docs]``
- ``pip install .[security]``
- ``pip install .[linting]``
- ``pip install .[all]``

These are useful for development, testing, documentation, security, and linting.

Development Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

- **pandas**: Advanced data manipulation and analysis (used in some tutorials)
- **openpyxl**: Excel file support for data import/export in some examples
- **black**: Code formatting (for contributors)
- **isort**: Import sorting (for contributors)
- **rich**: Enhanced and colored console output (progress bars, tracebacks)
- **typing_extensions**: Backport of Python typing features for compatibility
- **build**: Build system for packaging
- **twine**: Package distribution
- **commitizen**: Commit message management
- **notebook**: Jupyter Notebook support for interactive examples and tutorials
- **pyyaml**: YAML file support

Testing Dependencies
^^^^^^^^^^^^^^^^^^^^

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-html**: HTML reporting
- **pytest-xdist**: Parallel/distributed testing
- **pytest-timeout**: Test timeout

Documentation Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **sphinx**: Documentation building and formatting
- **sphinx-rtd-theme**: ReadTheDocs theme for documentation
- **sphinx-autodoc-typehints**: Type hints for documentation
- **sphinx-copybutton**: Copy button for documentation
- **pydocstyle**: Documentation style checking
- **interrogate**: Documentation coverage checking

Security Dependencies
^^^^^^^^^^^^^^^^^^^^^
- **bandit[toml]**: Security vulnerability scanning
- **safety**: Dependency vulnerability scanning
- **pip-audit**: Dependency audit
- **pip-tools**: Dependency management

Linting Dependencies
^^^^^^^^^^^^^^^^^^^^

- **ruff**: Code formatting
- **mypy**: Type checking
- **radon**: Code complexity checking
- **certifi**: Certificate management (dependency for some environments)

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
