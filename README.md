# `torchsom`: The Reference PyTorch Library for Self-Organizing Maps

<div align="center">

<!-- Global Package information -->
[![PyPI version](https://img.shields.io/pypi/v/torchsom.svg?color=blue)](https://pypi.org/project/torchsom/)
[![Python versions](https://img.shields.io/pypi/pyversions/torchsom.svg?color=blue)](https://pypi.org/project/torchsom/)
[![PyTorch versions](https://img.shields.io/badge/PyTorch-2.7-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/license/apache-2-0)

<!-- Sonar Qube badges -->
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=michelin_TorchSOM&metric=coverage)](https://sonarcloud.io/summary/new_code?id=michelin_TorchSOM)

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
    <img src="assets/logo.png" alt="TorchSOM_logo" width="400"/>
</p>

**A modern, comprehensive and GPU-accelerated PyTorch implementation of Self-Organizing Maps for scalable ML workflows**

[📄 Paper](https://arxiv.org/abs/2510.11147)
| [📚 Documentation](https://opensource.michelin.io/TorchSOM/)
| [🚀 Quick Start](#-quick-start)
| [📊 Examples](notebooks/)
| [🤝 Contributing](CONTRIBUTING.md)

**⭐ If you find [`torchsom`](https://github.com/michelin/TorchSOM) valuable, please consider starring this repository ⭐**

</div>

---

## 🎯 Overview

Self-Organizing Maps (SOMs) remain **highly relevant in modern machine learning** (ML) due to their **interpretability**, **topology preservation**, and **computational efficiency**. They excel and are widely used in domains such as energy systems, biology, internet of things (IoT), environmental science, and industrial applications.

Despite their utility, the SOM ecosystem is fragmented. Existing implementations are often **outdated**, **unmaintained**, and **lack GPU acceleration or modern deep learning** (DL) **framework integration**, limiting adoption and scalability.

**`torchsom`** addresses these gaps as a **reference PyTorch library** for SOMs. It provides:

- **GPU-accelerated training**
- **Advanced clustering capabilities**
- A **scikit-learn-style API** for ease of use
- **Rich visualization tools**
- **Robust software engineering practices**

`torchsom` enables researchers and practitioners to integrate SOMs seamlessly into workflows, from exploratory data analysis to advanced model architectures.

This library accompanies the paper: [`torchsom`: The Reference PyTorch Library for Self-Organizing Maps](https://arxiv.org/abs/2510.11147). If you use `torchsom` in academic or industrial work, please cite both the paper and the software (see [`CITATION`](CITATION.cff)).

> **Note**: See the comparison table below to understand how `torchsom` differs from other SOM libraries, and explore our [Visualization Gallery](#-visualization-gallery) for example outputs.

## ⚡ Why `torchsom`?

Unlike legacy implementations, `torchsom` is engineered from the ground up for modern ML workflows:

|  | [torchsom](https://github.com/michelin/TorchSOM) | [MiniSom](https://github.com/JustGlowing/minisom) | [SimpSOM](https://github.com/fcomitani/simpsom) | [SOMPY](https://github.com/sevamoo/SOMPY) | [somoclu](https://github.com/peterwittek/somoclu) | [som-pbc](https://github.com/alexarnimueller/som) |
|---|---|---|---|---|---|---|
| **Architecture Section** |  |  |  |  |  |  |
| Framework | PyTorch | NumPy | NumPy | NumPy | C++/CUDA | NumPy |
| GPU Acceleration |  ✅ CUDA | ❌ | ✅ CuPy/CUML | ❌ | ✅ CUDA | ❌ |
| API Design | scikit-learn | Custom | Custom | MATLAB | Custom | custom |
| **Development Quality Section** |  |  |  |  |  |  |
| Maintenance | ✅ Active | ✅ Active | ⚠️ Minimal | ⚠️ Minimal | ⚠️ Minimal | ❌ |
| Documentation | ✅ Rich | ❌ | ⚠️ Basic | ❌ | ⚠️ Basic | ⚠️ Basic |
| Test Coverage | ✅ 90% | ❌ | 🟠 ~53% | ❌ | ⚠️ Minimal | ❌ |
| PyPI Distribution | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Functionality Section** |  |  |  |  |  |  |
| Visualization | ✅ Advanced | ❌ | 🟠 Moderate | 🟠 Moderate | ⚠️ Basic | ⚠️ Basic |
| Clustering | ✅ Advanced | ❌ | ❌ | ❌ | ❌ | ❌ |
| JITL support | ✅ Built-in | ❌ | ❌ | ❌ | ❌ | ❌ |
| SOM Variants | 🚧 In development | ❌ | 🟠 PBC | ❌ | 🟠 PBC | 🟠 PBC |
| Extensibility | ✅ High | 🟠 Moderate | ⚠️ Low | ⚠️ Low | ⚠️ Low | ⚠️ Low |

> **Note**: `torchsom` supports **Just-In-Time Learning (JITL)**.
> Given an online query, JITL collects relevant datapoints to form a local buffer (selected first by topology, then by distance). A lightweight local model can then be trained on this buffer, enabling efficient supervised learning (regression or classification).

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
- [Related Work and References](#-related-work-and-references)

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
visualizer.plot_hit_map(data=X, batch_size=256, save_path=None)
visualizer.plot_distance_map(
  save_path=None,
  distance_metric=som.distance_fn_name,
  neighborhood_order=som.neighborhood_order,
  scaling="sum"
)
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
        <img src="assets/michelin_dmatrix.png" alt="U-Matrix" width="220"/>
      </td>
      <td align="center">
        <b>📍 Hit Map Visualization</b><br>
        <p>Michelin production line (regression)</p>
        <img src="assets/michelin_hitmap.png" alt="Hit Map" width="220"/>
      </td>
      <td align="center">
        <b>📊 Mean Map Visualization</b><br>
        <p>Michelin production line (regression)</p>
        <img src="assets/michelin_meanmap.png" alt="Mean Map" width="220"/>
      </td>
    </tr>
    <tr>
      <td align="center" colspan="2">
        <b>🗺️ Component Planes Visualization</b><br>
        <p>Another Michelin line (regression)</p>
        <table>
          <tr>
            <td align="center">
              <img src="assets/michelin_cp12.png" alt="Component Plane 1" width="220"/>
            </td>
            <td align="center">
              <img src="assets/michelin_cp21.png" alt="Component Plane 2" width="220"/>
            </td>
          </tr>
        </table>
      </td>
      <td align="center">
        <b>🏷️ Classification Map</b><br>
        <p>Wine dataset (multi-classification)</p>
        <img src="assets/wine_classificationmap.png" alt="Classification Map" width="220"/>
      </td>
    </tr>
    <tr>
      <td align="center">
        <b>📊 Cluster Metrics</b><br>
        <p>Clustering analysis</p>
        <img src="assets/cluster_metrics.png" alt="Cluster Metrics" width="220"/>
      </td>
      <td align="center">
        <b>📈 K-Means Elbow</b><br>
        <p>Optimal cluster selection</p>
        <img src="assets/kmeans_elbow.png" alt="K-Means Elbow" width="220"/>
      </td>
      <td align="center">
        <b>🎯 HDBSCAN Cluster Map</b><br>
        <p>Cluster visualization</p>
        <img src="assets/hdbscan_cluster_map.png" alt="HDBSCAN Cluster Map" width="220"/>
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
pip install -e ".[all]"
```

---

## 📚 Documentation

Comprehensive documentation is available at [opensource.michelin.io/TorchSOM](https://opensource.michelin.io/TorchSOM/)

---

## 📝 Citation

If you use `torchsom` in your academic, research or industrial work, please cite both the paper and software:

```bibtex
@misc{berthier2025torchsom,
    title={torchsom: The Reference PyTorch Library for Self-Organizing Maps},
    author = {Berthier, Louis and Shokry, Ahmed and Moreaud, Maxime and Ramelet, Guillaume and Moulines, Eric},
    year={2025},
    eprint={2510.11147},
    archivePrefix={arXiv},
    primaryClass={stat.ML},
    note={Preprint submitted to Journal of Machine Learning Research},
    url={https://arxiv.org/abs/2510.11147}
}

@software{berthier2025torchsom_software,
    author={Berthier, Louis},
    title={torchsom: The Reference PyTorch Library for Self-Organizing Maps},
    year={2025},
    version={1.1.1},
    url={https://github.com/michelin/TorchSOM},
    note = {Documentation available at \url{https://opensource.michelin.io/TorchSOM/}}
}
```

<!-- % If accepted by JMLR
@article{berthier2025torchsom,
    title={torchsom: The Reference PyTorch Library for Self-Organizing Maps},
    author={Louis Berthier and Ahmed Shokry and Maxime Moreaud and Guillaume Ramelet and Eric Moulines},
    journal={Journal of Machine Learning Research},
    year={2025},
    volume={26},
    number={123},
    pages={1--15},
    url={https://www.jmlr.org/papers/v26/25-11147.html},
    arxiv={2510.11147}
} -->

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

## 📚 Related Work and References

### 📖 Foundational Literature Papers

- Kohonen, T. (2001). [Self-Organizing Maps](https://link.springer.com/book/10.1007/978-3-642-56927-2). Springer.

### 🔗 Related Softwares

- [MiniSom](https://github.com/JustGlowing/minisom): Minimalistic Python SOM
- [SimpSOM](https://github.com/fcomitani/simpsom):Simple Self-Organizing Maps
- [SOMPY](https://github.com/sevamoo/SOMPY): Python SOM library
- [somoclu](https://github.com/peterwittek/somoclu): Massively Parallel Self-Organizing Maps
- [som-pbc](https://github.com/alexarnimueller/som): A simple self-organizing map implementation in Python with periodic boundary conditions
- [SOM Toolbox](http://www.cis.hut.fi/projects/somtoolbox/): MATLAB implementation

---
