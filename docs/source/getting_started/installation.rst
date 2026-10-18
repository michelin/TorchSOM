Installation
============

Requirements
------------

Systems
~~~~~~~

TorchSOM requires:

- **Python**: 3.10 or higher
- **PyTorch**: installed automatically as a dependency (see below for a GPU build)
- **Operating System**: Linux, macOS, or Windows

GPU
~~~

TorchSOM runs on any device PyTorch supports. GPU acceleration needs a CUDA build of
PyTorch that matches your driver (see `Ensure GPU Support`_). For large maps, 4 GB or
more of GPU memory is recommended.

Installation Methods
--------------------

Install from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install torchsom

Or, in a `uv <https://docs.astral.sh/uv/>`_ project:

.. code-block:: bash

   uv add torchsom

This installs TorchSOM with all required dependencies. For the optional FAISS backend
for BMU search, quote the extra so your shell doesn't expand the brackets:

.. code-block:: bash

   pip install "torchsom[faiss]"

Install from Source
~~~~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   # Clone the TorchSOM repository
   git clone https://github.com/michelin/TorchSOM.git
   cd TorchSOM

   # For standard users: install the main package
   uv sync

   # For contributors and developers: install with all optional dependencies
   uv sync --all-extras

Ensure GPU Support
~~~~~~~~~~~~~~~~~~

On Linux, the default PyTorch wheels from PyPI already include CUDA support. On Windows,
or to target a specific CUDA version, install PyTorch first with the command that the
`official selector <https://pytorch.org/get-started/locally/>`_ gives for your system,
then install TorchSOM:

.. code-block:: bash

   # Example for CUDA 12.8; use the command the selector gives for your system.
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install torchsom

TorchSOM only needs ``torch``; ``torchvision`` and ``torchaudio`` are not required. In a
uv project, configure the PyTorch index in ``pyproject.toml`` as described in
`uv's PyTorch guide <https://docs.astral.sh/uv/guides/integration/pytorch/>`_, so that
``uv sync`` keeps the CUDA build.

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

Optional dependencies are grouped into extras. Install one with
``pip install "torchsom[faiss]"``, or all of them in a source checkout with
``uv sync --all-extras``. The exact package lists live in ``pyproject.toml``.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Extra
     - Purpose
   * - ``faiss``
     - FAISS backend for BMU search (``faiss-cpu``)
   * - ``dev``
     - Notebooks and tutorials, plus the MiniSom and Numba benchmark baselines
   * - ``tests``
     - pytest and its plugins (coverage, HTML reports, parallel runs, timeouts)
   * - ``docs``
     - Sphinx, the Furo theme, and the documentation extensions
   * - ``linting``
     - ruff (linting and formatting), mypy, and radon
   * - ``release``
     - build, twine, and commitizen
   * - ``security``
     - pip-audit
   * - ``all``
     - Every extra above

Getting Help
------------

If you encounter installation issues:

1. Check the :doc:`../additional_resources/troubleshooting`
2. Search existing `GitHub Issues <https://github.com/michelin/TorchSOM/issues>`_
3. Create a new issue with your system details and error message

Next steps
----------

Once installed, continue with:

- :doc:`quickstart` — Your first SOM in a few minutes
- :doc:`basic_concepts` — Understanding SOM fundamentals
