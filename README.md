# TorchSOM: A Scalable PyTorch-Compatible Library for Self-Organizing Maps

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-red.svg)](https://opensource.org/license/apache-2-0)

<p align="center">
    <img src="logo.jpg" alt="TorchSOM_logo" width="350"/>
</p>

**`TorchSOM`** is the official code for
paper [PAPER ARTICLE AND LINK TO UPDATE](), @CONFERENCE @DATE.  
It provides an efficient and scalable implementation of **Self-Organizing Maps (SOMs)** using PyTorch, making it easy to integrate with deep learning workflows while benefiting from GPU acceleration.



## Table of Contents

- [TorchSOM: A Scalable PyTorch-Compatible Library for Self-Organizing Maps](#torchsom-a-scalable-pytorch-compatible-library-for-self-organizing-maps)
    - [Table of Contents](#table-of-contents)
    - [Introduction](#introduction)
    - [Installation](#installation)
    - [Documentation](#documentation)
    - [Citation](#citation)
    - [Acknowledgement](#acknowledgement)
    - [Contributions](#contributions)
    - [License](#license)
    - [References](#references)



## Introduction

**`TorchSOM`** is a PyTorch-based library for training Self-Organizing Maps (SOMs), a type of unsupervised learning algorithm used for clustering and dimensionality reduction. Designed for scalability and ease of use, this implementation is optimized for large-scale data.

Also, this **repository is highly documented and commented**, which makes it easy to use, understand, deploy, and which offers endless possibilities for improvements.  
To help you explore and experiment with **`TorchSOM`**, we provide Jupyter notebooks in the [`notebooks/`](notebooks) directory. There are multiples datasets and the corresponding results respectively in the [`data/`](data) and [`results/`](notebooks/results) directories.
- [`iris.ipynb`](notebooks/iris.ipynb): A multiclass classification example.
- [`wine.ipynb`](notebooks/wine.ipynb): Another multiclass classification example.
- [`boston_housing.ipynb`](notebooks/boston_housing.ipynb): A regression example.
- [`energy_efficiency.ipynb`](notebooks/energy_efficiency.ipynb): A multi-regression example.
- [`get_data.ipynb`](notebooks/get_data.ipynb): The notebook used to generated the datasets provided in the [`data/`](data) directory. 

If you find this project interesting, we would be grateful for your support by starring ⭐ this [`GitHub repository`](https://github.com/michelin/TorchSOM).

Here are some examples of visualizations you can obtain through the use of **`TorchSOM`**.



## Installation

You can install the package using PyPI (not available yet):

```bash
pip install torchsom
```

If you want to use the latest version, or if you prefer the command line interface, you can use it locally by cloning or forking this repository to your local machine. `TorchSOM` requires a recent version of Python, preferably **3.9 or higher**.  

```bash
git clone https://github.com/michelin/TorchSOM.git 
```

If you want to develop the package and run the notebooks after cloning the repository, make sure you have the required dependencies installed before using it:

```bash
python3.9 -m venv .torchsom_env # Create a virtual environment 
source .torchsom_env/bin/activate # Activate the environment 
pip install -e '.[dev]' # Install the required dependencies 
```



## Documentation

For more details on classes and functions, please refer to the `TorchSOM` [documentation](https://michelin.github.io/TorchSOM/index.html).

<!-- 

Generate the `.rst` files with

```bash
sphinx-apidoc -o docs/source torchsom
```

Then, to rebuild everything in [`docs/build/html`](docs/build/html):

```bash
cd docs/
make clean
make html
```

Open [`index.html`](docs/build/html/index.html) to preview locally the generated HTML. 

-->



## Citation

If you use `TorchSOM` in your research or work, please cite both the article and the software itself using the following entries:

1. **Paper**  
    ```bibtex
    @inproceedings{Berthier2025TorchSOM_paper,
        title        = {TorchSOM: A Scalable PyTorch-Compatible Library for Self-Organizing Maps},
        author       = {Berthier, Louis},
        year         = {2025},
    }
    ```

2. **Software**  
    ```bibtex
    @software{Berthier2025TorchSOM_software,
        author       = {Berthier, Louis},
        title        = {TorchSOM: A Scalable PyTorch-Compatible Library for Self-Organizing Maps},
        year         = {2025},
        url          = {https://github.com/michelin/TorchSOM},
        license      = {Apache-2.0},
    }
    ```

For more details, please refer to the [`CITATION.cff`](CITATION.cff) file.



## Acknowledgement

The [Centre de Mathématiques Appliquées](https://cmap.ip-paris.fr/) - CMAP - at the Ecole Polytechnique - X -, and [Manufacture Française des Pneumatiques Michelin](https://www.michelin.com/) for the joint collaboration and supervision during my PhD thesis.

[Giuseppe Vettigli](https://github.com/JustGlowing) for his GitHub repository [MiniSom](https://github.com/JustGlowing/minisom), which provided a first well-maintained and open-source implementation of Self-Organizing Maps.

Logo created using DALL-E.



## Contributions

We invite contributors of all backgrounds and experience levels to get involved and contribute to this library. Whether you have innovative ideas to propose or are eager to submit pull requests, we encourage your contributions!

Please take a moment to read our [`Code of Conduct`](CODE_OF_CONDUCT.md) and the [`Contributing guide`](CONTRIBUTING.md) if you're interested in contributing.



## License

TorchSOM is licensed under the [Apache License, Version 2.0](https://opensource.org/license/apache-2-0). Feel free to use and modify the code as per the terms of the [`LICENSE`](LICENSE).



## References

- **Related Papers**:
    - [Self-Organizing Maps](https://link.springer.com/book/10.1007/978-3-642-56927-2) by Teuvo Kohonen, 2001
    - [An Introduction to Self-Organizing Maps](https://link.springer.com/chapter/10.2991/978-94-91216-77-0_14) by Umut Asan & Secil Ercan, 2012
    - [Brief Review of Self-Organizing Maps](https://www.researchgate.net/publication/317339061_Brief_Review_of_Self-Organizing_Maps) by Dubravko Miljković, 2017
- **Code Inspiration**:  
    Built upon concepts from [MiniSom](https://github.com/JustGlowing/minisom) by [Giuseppe Vettigli](https://github.com/JustGlowing)
- **Repository Structure**:  
    Organized following the [CCSD](https://github.com/AdrienC21/CCSD) format by [Adrien Carrel](https://github.com/AdrienC21)





<!-- TO CHECK when swtiching from private to open repo

[![visitors](https://visitor-badge.laobi.icu/badge?page_id=LouisTier.TorchSOM&right_color=%23FFA500)](https://github.com/LouisTier/TorchSOM/)
[![Downloads](https://static.pepy.tech/badge/torch_som)](https://pepy.tech/project/torch_som)

-->

<!-- NOT SURE to keep

[![pypi version](https://img.shields.io/pypi/v/ccsd.svg)](https://pypi.python.org/pypi/ccsd)
[![Documentation Status](https://readthedocs.org/projects/ccsd/badge/?version=latest)](https://ccsd.readthedocs.io/en/latest/?badge=latest)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://pypi.python.org/pypi/ccsd)
[![Test](https://github.com/AdrienC21/CCSD/actions/workflows/test.yml/badge.svg)](https://github.com/AdrienC21/CCSD/actions/workflows/test.yml)
[![Lint](https://github.com/AdrienC21/CCSD/actions/workflows/lint.yml/badge.svg)](https://github.com/AdrienC21/CCSD/actions/workflows/lint.yml)
[![Codecov](https://codecov.io/gh/AdrienC21/CCSD/branch/main/graph/badge.svg)](https://app.codecov.io/gh/AdrienC21/CCSD) 
-->
