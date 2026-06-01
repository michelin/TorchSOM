Configuration API
==================

The configs module provides structured configuration management using Pydantic models.

SOM Configuration
-----------------

.. automodule:: torchsom.configs.som_config
   :members:
   :undoc-members:
   :show-inheritance:

Loading and Saving
------------------

``SOMConfig`` is a Pydantic model, so it round-trips through dictionaries, JSON, and
YAML for reproducible experiments.

.. code-block:: python

   import yaml
   from torchsom.configs import SOMConfig

   # Load from a YAML file
   with open("som_config.yaml") as f:
       config = SOMConfig(**yaml.safe_load(f))

   # Export to a dict or a JSON string
   config_dict = config.model_dump()
   config_json = config.model_dump_json(indent=2)

   # Save back to YAML
   with open("exported_config.yaml", "w") as f:
       yaml.dump(config_dict, f, default_flow_style=False)
