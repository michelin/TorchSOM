Configuration API
==================

The configs module provides structured configuration management using Pydantic models.

SOM Configuration
-----------------

.. automodule:: torchsom.configs.som_config
   :members:
   :undoc-members:
   :show-inheritance:

Saving Configuration
--------------------

Loading Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import yaml
   from torchsom.configs import SOMConfig

   # Load from YAML file
   with open("som_config.yaml", "r") as f:
       config_dict = yaml.safe_load(f)
   config = SOMConfig(**config_dict)

Exporting Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import yaml

   # Export to dictionary and JSON
   config_dict = config.dict()
   config_json = config.json(indent=2)

   # Save to YAML file
   with open("exported_config.yaml", "w") as f:
       yaml.dump(config.dict(), f, default_flow_style=False)
