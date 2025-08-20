# `torchsom`: The Reference PyTorch Library for Self-Organizing Maps

<div align="center">

<!-- Global Package information -->
[![PyPI version](https://img.shields.io/pypi/v/torchsom.svg?color=blue)](https://pypi.org/project/torchsom/)
[![Python versions](https://img.shields.io/pypi/pyversions/torchsom.svg?color=blue)](https://pypi.org/project/torchsom/)
[![PyTorch versions](https://img.shields.io/badge/PyTorch-2.7-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/license/apache-2-0)

<!-- Sonar Qube badges -->
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=coverage)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)

<!-- [![SonarQube Cloud](https://sonarcloud.io/images/project_badges/sonarcloud-light.svg)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM) -->

<!-- GitHub Workflows -->
[![Tests](https://github.com/michelin/TorchSOM/workflows/Tests/badge.svg)](https://github.com/michelin/TorchSOM/actions/workflows/test.yml)
[![Code Quality](https://github.com/michelin/TorchSOM/workflows/Code%20Quality/badge.svg)](https://github.com/michelin/TorchSOM/actions/workflows/code-quality.yml)
[![Downloads](https://static.pepy.tech/badge/torchsom)](https://pepy.tech/project/torchsom)
[![GitHub stars](https://img.shields.io/github/stars/michelin/TorchSOM.svg?style=social&label=Star)](https://github.com/michelin/TorchSOM)
<!-- [![Security](https://github.com/michelin/TorchSOM/workflows/Security%20Scanning/badge.svg)](https://github.com/michelin/TorchSOM/actions/workflows/security.yml) -->

<!-- Code Style and Tools -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-6c757d.svg?color=blue)](https://pycqa.github.io/isort/)
[![Ruff](https://img.shields.io/badge/linter-ruff-6c757d.svg?color=blue)](https://github.com/astral-sh/ruff)

<p align="center">
    <img src="assets/logo.jpg" alt="TorchSOM_logo" width="400"/>
</p>

**The most comprehensive, scalable, and PyTorch-native implementation of Self-Organizing Maps**

[📚 Documentation](https://opensource.michelin.io/TorchSOM/)
| [🚀 Quick Start](#-quick-start)
| [📊 Examples](notebooks/)
| [🤝 Contributing](CONTRIBUTING.md)
<!-- | [📄 Paper (PDF)](assets/torchsom_jmlr.pdf) -->

**⭐ If you find [`torchsom`](https://github.com/michelin/TorchSOM) valuable, please consider starring this repository ⭐**

</div>

---

## 🎯 Why `torchsom`?

**`torchsom`** is the reference PyTorch library for Self-Organizing Maps (SOMs), purpose-built for seamless integration with modern deep learning and scientific workflows.
Unlike legacy SOM packages, `torchsom` is engineered from the ground up to fully leverage PyTorch’s ecosystem—offering native GPU acceleration, scalable performance, and compatibility with neural network pipelines.
Whether you are a researcher or practitioner, `torchsom` empowers you to efficiently incorporate SOMs into your machine learning projects, from exploratory data analysis to advanced model architectures.

`torchsom` is the official implementation accompanying the paper: [`torchsom`: The Reference PyTorch Library for Self-Organizing Maps](assets/torchsom_jmlr.pdf).
If you use `torchsom` in academic or industrial work, please cite the paper and the software (see [`CITATION`](CITATION.cff)).

> **Note**: Check the table below for a comprehensive comparison of how `torchsom` differs from existing SOM libraries, and explore our [Visualization Gallery](#-visualization-gallery) for examples of the rich visual outputs available.

### ⚡ Key Advantages

| Aspect | [torchsom](https://github.com/michelin/TorchSOM) | [MiniSom](https://github.com/JustGlowing/minisom) | [SimpSOM](https://github.com/fcomitani/simpsom) | [SOMPY](https://github.com/sevamoo/SOMPY) | [somoclu](https://github.com/peterwittek/somoclu) | [som-pbc](https://github.com/alexarnimueller/som) |
|---|---|---|---|---|---|---|
| Framework | PyTorch | NumPy | NumPy | NumPy | C++/CUDA | NumPy |
| GPU Acceleration |  CUDA | ❌ | CuPy/CUML | ❌ | CUDA | ❌ |
| API Design | scikit-learn | Custom | Custom | MATLAB | Custom | custom |
| Maintenance | Active | Active | Minimal | Minimal | Minimal | ❌ |
| Documentation | Rich | ❌ | Basic | ❌ | Basic | Basic |
| Test Coverage | ~86% | ❌ | ~53% | ❌ | Minimal | ❌ |
| PyPI Distribution | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| SOM Variants | Multiple | ❌ | PBC | ❌ | PBC | PBC (only) |
| Visualization | Advanced | ❌ | Moderate | Moderate | Basic | Basic |
| Extensibility | High | Moderate | Low | Low | Low | Low |

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [Tutorials](#-tutorials)
- [Installation](#-installation)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)
- [References](#-references)
<!-- - [Reproducibility](#-reproducibility) -->
<!-- - [Performance Benchmarks](#-performance-benchmarks) -->

---

## 🚀 Quick Start

Get started with `torchsom` in just a few lines of code:

```python
import torch
from torchsom.core import SOM
from torchsom.visualization import SOMVisualizer

# Create a 10x10 map for 3D input
som = SOM(x=10, y=10, num_features=3, epochs=50)

# Train SOM for 50 epochs on 1000 samples
X = torch.randn(1000, 3)
som.initialize_weights(data=X, mode="pca")
QE, TE = som.fit(data=X)

# Visualize results
visualizer = SOMVisualizer(som=som, config=None)
visualizer.plot_training_errors(quantization_errors=QE, topographic_errors=TE, save_path=None)
visualizer.plot_distance_map(save_path=None)
visualizer.plot_hit_map(data=X, save_path=None)
```

## 📓 Tutorials

Explore our comprehensive collection of Jupyter notebooks:

- 📊 [`iris.ipynb`](notebooks/iris.ipynb): Multiclass classification
- 🍷 [`wine.ipynb`](notebooks/wine.ipynb): Multiclass classification
- 🏠 [`boston_housing.ipynb`](notebooks/boston_housing.ipynb): Regression
- ⚡ [`energy_efficiency.ipynb`](notebooks/energy_efficiency.ipynb): Multi-output regression
- 🎯 [`clustering.ipynb`](notebooks/clustering.ipynb): SOM-based clustering analysis

### 🎨 Visualization Gallery

<p align="center">
  <table>
    <tr>
      <td align="center">
        <b>🗺️ D-Matrix Visualization</b><br>
        <p>Michelin production line (regression)</p>
        <img src="assets/michelin_dmatrix.png" alt="U-Matrix" width="280"/>
      </td>
      <td align="center">
        <b>📍 Hit Map Visualization</b><br>
        <p>Michelin production line (regression)</p>
        <img src="assets/michelin_hitmap.png" alt="Hit Map" width="280"/>
      </td>
      <td align="center">
        <b>📊 Mean Map Visualization</b><br>
        <p>Michelin production line (regression)</p>
        <img src="assets/michelin_meanmap.png" alt="Mean Map" width="280"/>
      </td>
    </tr>
    <tr>
      <td align="center" colspan="2">
        <b>🗺️ Component Planes Visualization</b><br>
        <p>Another Michelin line (regression)</p>
        <table>
          <tr>
            <td align="center">
              <img src="assets/michelin_cp12.png" alt="Component Plane 1" width="280"/>
            </td>
            <td align="center">
              <img src="assets/michelin_cp21.png" alt="Component Plane 2" width="280"/>
            </td>
          </tr>
        </table>
      </td>
      <td align="center">
        <b>🏷️ Classification Map</b><br>
        <p>Wine dataset (multi-classification)</p>
        <img src="assets/wine_classificationmap.png" alt="Classification Map" width="280"/>
      </td>
    </tr>
    <tr>
      <td align="center">
        <b>📊 Cluster Metrics</b><br>
        <p>Clustering analysis</p>
        <img src="assets/cluster_metrics.png" alt="Cluster Metrics" width="320"/>
      </td>
      <td align="center">
        <b>📈 K-Means Elbow</b><br>
        <p>Optimal cluster selection</p>
        <img src="assets/kmeans_elbow.png" alt="K-Means Elbow" width="300"/>
      </td>
      <td align="center">
        <b>🎯 HDBSCAN Cluster Map</b><br>
        <p>Cluster visualization</p>
        <img src="assets/hdbscan_cluster_map.png" alt="HDBSCAN Cluster Map" width="260"/>
      </td>
    </tr>
  </table>
</p>

---

## 💾 Installation

### 📦 PyPI

```bash
pip install torchsom
```

### 🔧 Development Version

```bash
git clone https://github.com/michelin/TorchSOM.git
cd TorchSOM
python3.9 -m venv .torchsom_env
source .torchsom_env/bin/activate
pip install -e ".[dev]"
```

---

## 📚 Documentation

Comprehensive documentation is available at [opensource.michelin.io/TorchSOM](https://opensource.michelin.io/TorchSOM/)

---

## 📝 Citation

If you use `torchsom` in your research, please cite both the paper and software:

```bibtex
@inproceedings{Berthier2025TorchSOM,
    title={torchsom: The Reference PyTorch Library for Self-Organizing Maps},
    author={Berthier, Louis},
    booktitle={Conference Name},
    year={2025}
}

@software{Berthier_TorchSOM_The_Reference_2025,
    author={Berthier, Louis},
    title={torchsom: The Reference PyTorch Library for Self-Organizing Maps},
    url={https://github.com/michelin/TorchSOM},
    version={1.0.0},
    year={2025}
}
```

For more details, please refer to the [CITATION](CITATION.cff) file.

---

## 🤝 Contributing

We welcome contributions from the community! See our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

- **GitHub Issues**: [Report bugs or request features](https://github.com/michelin/TorchSOM/issues)
<!-- - **GitHub Discussions**: [Ask questions and share ideas](https://github.com/michelin/TorchSOM/discussions) -->

---

## 🙏 Acknowledgments

- [Centre de Mathématiques Appliquées (CMAP)](https://cmap.ip-paris.fr/) at École Polytechnique
- [Manufacture Française des Pneumatiques Michelin](https://www.michelin.com/) for collaboration
- [Giuseppe Vettigli](https://github.com/JustGlowing) for [MiniSom](https://github.com/JustGlowing/minisom) inspiration
- The [PyTorch](https://pytorch.org/) team for the amazing framework
- Logo created using [DALL-E](https://openai.com/index/dall-e-3/)

---

## 📄 License

`torchsom` is licensed under the [Apache License 2.0](LICENSE). See the [LICENSE](LICENSE) file for details.

---

## 📚 References

### 📖 Core Papers

- Kohonen, T. (2001). [Self-Organizing Maps](https://link.springer.com/book/10.1007/978-3-642-56927-2). Springer.

### 🔗 Related Projects

- [MiniSom](https://github.com/JustGlowing/minisom): Minimalistic Python SOM
- [SimpSOM](https://github.com/fcomitani/simpsom):Simple Self-Organizing Maps
- [SOMPY](https://github.com/sevamoo/SOMPY): Python SOM library
- [somoclu](https://github.com/peterwittek/somoclu): Massively Parallel Self-Organizing Maps
- [som-pbc](https://github.com/alexarnimueller/som): A simple self-organizing map implementation in Python with periodic boundary conditions
- [SOM Toolbox](http://www.cis.hut.fi/projects/somtoolbox/): MATLAB implementation

---

<!-- ### 🎨 Core Capabilities

- **🏗️ Multiple SOM Variants**: Batch SOM, Growing SOM, Hierarchical SOM, and other variants
- **⚡ GPU Acceleration**: Full CUDA support with automatic device management
- **🔄 PyTorch Native**: Seamless integration with existing PyTorch workflows
- **📊 Rich Visualizations**: 10+ visualization types including U-Matrix, Component Planes, Hit Maps
- **🎯 Flexible Training**: Online and batch learning modes with customizable schedules

### 🧪 Advanced Features

- **🌐 Deep Learning Integration**: Use SOMs as layers, with autograd support
- **📈 Scalable Architecture**: Handle millions of samples efficiently
- **🔧 Customizable Components**: Plug-in architecture for neighborhoods, learning rates, distances
- **📱 Multi-GPU Support**: Distributed training for large-scale applications
- **🎮 Interactive Visualizations**: Real-time training monitoring and exploration

### 🛠️ Technical Specifications

- **✅ Type Safety**: Full type hints and runtime validation with Pydantic
- **📝 Comprehensive Docs**: Detailed API documentation with examples
- **🧪 Thoroughly Tested**: 95%+ test coverage with CI/CD
- **🔍 Reproducible**: Deterministic training with seed management -->

<!-- ---

## ⚡ Performance Benchmarks

### Training & Memory Performance

| Case | Samples | Features | Map Size | TorchSOM<br>CPU Time (s) | TorchSOM<br>CPU RAM (GB) | TorchSOM<br>GPU Time (s) | TorchSOM<br>GPU RAM (GB) | MiniSom<br>CPU Time (s) | MiniSom<br>CPU RAM (GB) |
|------|---------|----------|----------|--------------------------|--------------------------|--------------------------|--------------------------|-------------------------|-------------------------|
| 1    | 1,000   | 30       | 15×15    | 0.7 ± 0.1                | 0.6 ± 0.1                | 0.09 ± 0.01              | 0.09 ± 0.01              | 2.1 ± 0.2               | 0.7 ± 0.1               |
| 2    | 2,500   | 40       | 20×20    | 1.5 ± 0.2                | 0.8 ± 0.1                | 0.15 ± 0.01              | 0.12 ± 0.01              | 4.8 ± 0.3               | 1.0 ± 0.1               |
| 3    | 5,000   | 50       | 25×25    | 3.2 ± 0.3                | 1.2 ± 0.1                | 0.28 ± 0.02              | 0.18 ± 0.01              | 10.2 ± 0.5              | 1.6 ± 0.2               |
| 4    | 10,000  | 60       | 30×30    | 6.8 ± 0.5                | 1.8 ± 0.2                | 0.55 ± 0.03              | 0.25 ± 0.02              | 22.5 ± 1.0              | 2.5 ± 0.2               |
| 5    | 20,000  | 80       | 40×40    | 14.7 ± 0.8               | 2.9 ± 0.2                | 1.1 ± 0.05               | 0.42 ± 0.03              | 48.9 ± 2.0              | 4.2 ± 0.3               |
| 6    | 35,000  | 100      | 50×50    | 28.3 ± 1.2               | 4.5 ± 0.3                | 2.0 ± 0.08               | 0.65 ± 0.04              | 98.7 ± 3.5              | 7.0 ± 0.5               |
| 7    | 50,000  | 120      | 60×60    | 44.2 ± 1.8               | 6.2 ± 0.4                | 3.1 ± 0.10               | 0.92 ± 0.05              | 152.3 ± 5.0             | 9.8 ± 0.7               |
| 8    | 65,000  | 150      | 75×75    | 62.8 ± 2.5               | 8.7 ± 0.6                | 4.5 ± 0.15               | 1.3 ± 0.07               | 210.5 ± 7.0             | 13.5 ± 1.0              |
| 9    | 80,000  | 200      | 100×100  | 85.6 ± 3.0               | 12.1 ± 0.8               | 6.2 ± 0.20               | 1.8 ± 0.09               | 295.0 ± 10.0            | 18.7 ± 1.2              |
| 10   | 100,000 | 300      | 125×125  | 120.3 ± 4.0              | 18.5 ± 1.2               | 8.9 ± 0.25               | 2.7 ± 0.12               | 410.2 ± 15.0            | 27.5 ± 2.0              |

*RAM values are in GB. Times are in seconds. "OOM" = (out of memory).
Benchmarks performed on NVIDIA RTX 3090 (GPU), AMD Ryzen 9 5900X (CPU).*

### Training & Memory Speedup

| Case | TorchSOM CPU Speedup | TorchSOM GPU Speedup | TorchSOM CPU RAM Saving | TorchSOM GPU RAM Saving |
|------|------------------------|-------------------------|----------------------------|----------------------------|
| 1    | 3.0×                   | 23.3×                   | 1.2×                       | 7.8×                       |
| 2    | 3.2×                   | 32.0×                   | 1.3×                       | 8.3×                       |
| 3    | 3.2×                   | 36.4×                   | 1.3×                       | 8.9×                       |
| 4    | 3.3×                   | 40.9×                   | 1.4×                       | 10.0×                      |
| 5    | 3.3×                   | 44.5×                   | 1.4×                       | 10.0×                      |
| 6    | 3.5×                   | 49.4×                   | 1.6×                       | 10.8×                      |
| 7    | 3.4×                   | 49.2×                   | 1.6×                       | 10.7×                      |
| 8    | 3.4×                   | 46.8×                   | 1.6×                       | 10.4×                      |
| 9    | 3.4×                   | 47.6×                   | 1.5×                       | 10.4×                      |
| 10   | 3.4×                   | 46.1×                   | 1.5×                       | 10.2×                      |

*Speedup = (MiniSom CPU Time / TorchSOM Time) for each mode.*

**Summary:**

- TorchSOM (GPU) achieves up to ~Z× speedup and ~Z× lower memory usage compared to MiniSom (CPU) on large datasets.
- TorchSOM (CPU) is consistently Y–Z× faster and more memory efficient than MiniSom (CPU).
- All benchmarks use identical data and map initialization for fair comparison. -->
