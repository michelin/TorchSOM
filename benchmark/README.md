# Benchmark for JMLR-MLOSS submission

This repository is dedicated to compare the performances between [`torchsom`](https://github.com/michelin/TorchSOM) and [`Minisom`](https://github.com/JustGlowing/minisom), using dataset examples available at [data/benchmark](../data/benchmark/)

## Set up

I assume you've already followed the instruction in the global [README](../README.md), having at disposal an environment <.torchsom_env>.
Now, using this environment:

```bash
# Activate it
source .torchsom_env/bin/activate
# Install latest MiniSom version used for benchmarking: v2.3.5
pip install git+https://github.com/JustGlowing/minisom.git@65b6ba6776f63db4536a85afa34bd7b2c6555960
```

Now you're free to experiment the notebook comparing both methods: [benchmark](benchmark.ipynb)
