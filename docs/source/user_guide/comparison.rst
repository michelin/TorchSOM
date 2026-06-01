Comparison with Other Libraries
===============================

Several Python libraries implement Self-Organizing Maps. They differ in technical
architecture, maintenance, and built-in capabilities. The table below reproduces the
comparison from the paper (Table 1); it is the basis for TorchSOM's positioning.

Libraries compared:
`TorchSOM <https://github.com/michelin/TorchSOM>`_,
`MiniSom <https://github.com/JustGlowing/minisom>`_,
`SimpSOM <https://github.com/fcomitani/simpsom>`_,
`SOMPY <https://github.com/sevamoo/SOMPY>`_,
`somoclu <https://github.com/peterwittek/somoclu>`_, and
`som-pbc <https://github.com/alexarnimueller/som>`_.


Feature matrix
--------------

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 22 14 12 12 12 12 12

   * -
     - TorchSOM
     - MiniSom
     - SimpSOM
     - SOMPY
     - somoclu
     - som-pbc
   * - **Framework**
     - PyTorch
     - NumPy
     - NumPy
     - NumPy
     - C++
     - NumPy
   * - **GPU acceleration**
     - CUDA (PyTorch)
     - ✗
     - CuPy / CUML
     - ✗
     - CUDA C++
     - ✗
   * - **API design**
     - scikit-learn
     - Custom
     - Custom
     - MATLAB
     - Custom
     - Custom
   * - **Maintenance**
     - Active
     - Active
     - Minimal
     - Minimal
     - Minimal
     - ✗
   * - **Documentation**
     - Rich
     - Basic [#minisom-docs]_
     - Basic
     - ✗
     - Basic
     - Basic
   * - **Test coverage**
     - 90%
     - 98%
     - 53%
     - ✗
     - Minimal
     - ✗
   * - **PyPI distribution**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓
     - ✗
   * - **Visualization**
     - Advanced
     - ✗
     - Moderate
     - Moderate
     - Basic
     - Basic
   * - **Clustering (built-in)**
     - ✓
     - Examples only [#minisom-clust]_
     - ✓
     - ✗
     - ✗
     - ✗
   * - **JITL support**
     - ✓
     - ✗
     - ✗
     - ✗
     - ✗
     - ✗
   * - **SOM variants**
     - Multiple [#variants]_
     - ✗
     - PBC
     - ✗
     - PBC
     - PBC
   * - **Extensibility**
     - High
     - Moderate
     - Low
     - Low
     - Low
     - Low

.. [#minisom-docs] MiniSom ships example notebooks and partial in-code docstrings,
   but no narrative documentation site comparable to this one.
.. [#minisom-clust] Clustering is not a built-in MiniSom feature; it requires
   user-supplied code on top of MiniSom primitives. TorchSOM provides
   :meth:`~torchsom.core.SOM.cluster` directly.
.. [#variants] In the current release, "Multiple" means rectangular and hexagonal
   topologies, each optionally toroidal via periodic boundary conditions. Growing and
   Hierarchical variants are on the roadmap (see the paper's Conclusion).


Where TorchSOM fits
-------------------

Existing libraries each address a specific niche: MiniSom is a minimalist,
NumPy-based implementation well suited to teaching and prototyping, while somoclu
targets HPC environments through CUDA C++. TorchSOM is the only library in this
comparison that combines, in a single modular codebase:

- a native PyTorch backend with GPU acceleration,
- a scikit-learn-compatible API,
- an advanced built-in visualization suite,
- a built-in clustering interface,
- just-in-time-learning support, and
- multiple grid topologies with configurable neighborhood retrieval modes.

It is further supported by this narrative documentation site and a community-oriented
development process, making it a complete and scalable reference for both research and
production. The performance side of this comparison — quantization-error parity with
substantially lower topographic error and training time — is documented in
:doc:`benchmarks`.


Next steps
----------

- :doc:`benchmarks` — Quantitative speed and quality comparison with MiniSom
- :doc:`architecture` — How TorchSOM is organized
- :doc:`../getting_started/quickstart` — Try it yourself
