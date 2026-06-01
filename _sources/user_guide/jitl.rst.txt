Just-in-Time Learning
=====================

Just-in-time learning (JITL) builds a small, local model on demand: for each new
query, it retrieves the most relevant historical samples and trains (or predicts) on
just those. A SOM is a natural retrieval index for this — its topology already places
similar samples on neighboring neurons. TorchSOM exposes this through
:meth:`~torchsom.core.SOM.collect_samples`, which gathers the historical samples that
land on the query's BMU and its neighborhood.

This pattern is common in industrial soft sensing and adaptive monitoring, where the
process drifts and a single global model goes stale.


The retrieval index
-------------------

``collect_samples`` needs a BMU→sample-index map of the historical data — the same
``"bmus_data"`` map used by the visualizations. Build it once after training:

.. code-block:: python

   som.initialize_weights(data=historical_samples, mode="pca")
   som.fit(data=historical_samples)

   bmus_idx_map = som.build_map("bmus_data", data=historical_samples)

``bmus_idx_map`` maps each grid cell ``(i, j)`` to the list of historical-sample
indices whose BMU is that cell.


Retrieving samples for a query
------------------------------

.. code-block:: python

   data_buffer, output_buffer = som.collect_samples(
       query_sample=query,                 # tensor [num_features]
       historical_samples=historical_samples,   # [N, num_features]
       historical_outputs=historical_outputs,   # [N]
       bmus_idx_map=bmus_idx_map,
       retrieval_mode="bmu_neighborhood_knn",
       min_buffer_threshold=50,
   )

It returns the matched inputs and their outputs (``data_buffer``, ``output_buffer``),
ready to fit a local regressor. Pass ``return_indices=True`` to also get the indices
of the retrieved samples.


Retrieval modes
---------------

The ``retrieval_mode`` argument trades recall against locality. All modes start from
the query's BMU cell; they differ in how far they expand:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - ``retrieval_mode``
     - Expands?
     - Strategy
   * - ``"bmu_only"``
     - No
     - Only samples mapped to the query's BMU cell. Tightest, smallest buffer.
   * - ``"bmu_neighborhood"``
     - Topological
     - BMU plus its grid neighbors up to ``neighborhood_order`` hops. No fallback.
   * - ``"bmu_neighborhood_knn"`` *(default)*
     - Topological + KNN
     - Same as above, then a nearest-neighbor fallback in weight space when the
       buffer is still below ``min_buffer_threshold``.

The KNN fallback in the default mode guarantees a usable buffer size even in sparse
regions of the map: if the BMU and its neighbors hold too few samples, the nearest
remaining neurons (by codebook distance) are pulled in until
``min_buffer_threshold`` is exceeded. The neighborhood extent is the SOM's
``neighborhood_order``, and under :ref:`periodic boundary conditions <topologies-pbc>`
the neighborhood wraps across edges.


Typical workflow
----------------

.. code-block:: python

   from sklearn.linear_model import LinearRegression

   # 1. Train the SOM once on the historical buffer and index it
   som.initialize_weights(data=historical_samples, mode="pca")
   som.fit(data=historical_samples)
   bmus_idx_map = som.build_map("bmus_data", data=historical_samples)

   # 2. For each incoming query, retrieve a local set and fit a local model
   for query in stream:
       X_local, y_local = som.collect_samples(
           query_sample=query,
           historical_samples=historical_samples,
           historical_outputs=historical_outputs,
           bmus_idx_map=bmus_idx_map,
           retrieval_mode="bmu_neighborhood_knn",
       )
       local_model = LinearRegression().fit(
           X_local.cpu().numpy(), y_local.cpu().numpy()
       )
       prediction = local_model.predict(query.reshape(1, -1).cpu().numpy())


Choosing a mode
---------------

- Start with ``"bmu_neighborhood_knn"`` (the default) — it adapts the buffer size to
  local data density.
- Use ``"bmu_neighborhood"`` when you want strictly local samples and accept a
  variable, possibly small, buffer.
- Use ``"bmu_only"`` for the most local model, or to inspect exactly which samples a
  single neuron represents.
- Tune ``min_buffer_threshold`` to the minimum sample count your local model needs.


Next steps
----------

- :doc:`topologies` — How ``neighborhood_order`` and PBC shape retrieval
- :doc:`../getting_started/basic_concepts` — The latent representation behind retrieval
- :doc:`../api/core` — ``collect_samples`` and ``build_map`` reference
