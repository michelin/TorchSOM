# Benchmark for JMLR-MLOSS submission

This repository is dedicated to compare the performances between [`torchsom`](https://github.com/michelin/TorchSOM) and [`Minisom`](https://github.com/JustGlowing/minisom), using dataset examples available at [data/benchmark](../data/benchmark/)

## Local setup

I assume you've already followed the instruction in the global [README](../README.md), having at disposal an environment <.torchsom_env>.
Now, using this environment:

```bash
# Activate it
source .torchsom_env/bin/activate
# Install latest MiniSom version used for benchmarking: v2.3.5
pip install git+https://github.com/JustGlowing/minisom.git@65b6ba6776f63db4536a85afa34bd7b2c6555960
```

Now you're free to experiment the notebook comparing both methods: notebook [benchmark.ipynb](benchmark.ipynb) or script [benchmark.py](benchmark.py)

<!-- ## Azure ML setup

```bash
# Install Azure client
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Connect to AZML
az login

# Install azure module
pip install azure-ai-ml
pip install azure-identity

# Provide keys to the environment
export AZUREML_SUBSCRIPTION="<key>"
export AZUREML_RESOURCE_GROUP="<key>"
export AZUREML_WORKSPACE_NAME="<key>"

# Create env on AZML
python environments/create_environment.py

# Run the raw command from the run_benchmark.yaml to run the job
``` -->
