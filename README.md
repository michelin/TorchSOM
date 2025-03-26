# TorchSOM: A Scalable PyTorch-Compatible Library for Self-Organizing Maps

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

<p align="center">
    <img src="logos/logo4.jpg" alt="TorchSOM_logo" width="350"/>
</p>

**TorchSOM** is the official code for
paper [PAPER ARTICLE AND LINK TO UPDATE](), CONFERENCE NAME DATE.  
It provides an efficient and scalable implementation of **Self-Organizing Maps (SOMs)** using PyTorch, making it easy to integrate with deep learning workflows while benefiting from GPU acceleration.




## Table of Contents

- [TorchSOM: A Scalable PyTorch-Compatible Library for Self-Organizing Maps](#torchsom-a-scalable-pytorch-compatible-library-for-self-organizing-maps)
    - [Table of Contents](#table-of-contents)
    - [Introduction](#introduction)
    - [Installation](#installation)
        <!-- - [Using pip](#using-pip)
        - [Manually](#manually) -->
    - [Dependencies](#dependencies)
    <!-- - [Documentation](#documentation) -->
    - [Citation](#citation)
    - [Acknowledgement](#acknowledgement)
    - [Contributions](#contributions)
    - [License](#license)
    - [References](#references)



## Introduction

**TorchSOM** is a PyTorch-based library for training Self-Organizing Maps (SOMs), a type of unsupervised learning algorithm used for clustering and dimensionality reduction. Designed for scalability and ease of use, this implementation is optimized for large-scale data.

Also, this **repository is highly documented and commented**, which makes it easy to use, understand, deploy, and which offers endless possibilities for improvements.  
To help you explore and experiment with **TorchSOM**, we provide Jupyter notebooks in the [notebooks/](https://github.com/LouisTier/TorchSOM/tree/main/notebooks) directory.

If you find this project interesting, we would be grateful for your support by starring ⭐ this [GitHub repository](https://github.com/LouisTier/TorchSOM).



## Installation

You can install the package using pip following this command:

```bash
pip install torchsom
```

If you encounter, if you want to use the latest version, or if you prefer the command line interface, you can use it locally by cloning or forking this repository to your local machine.

```bash
git clone https://github.com/LouisTier/TorchSOM.git
```



## Dependencies

TorchSOM requires a recent version of Python, preferably **3.10 or higher**. Also, make sure you have the required dependencies installed before using it.  
You can these dependencies by running the command:
```bash
pip install -r requirements.txt
```



<!-- ## Documentation

Here is the link to the documentation of this library: [https://ccsd.readthedocs.io/en/latest/](https://ccsd.readthedocs.io/en/latest/). It contains more information regarding all the classes and functions of this package. -->



## Citation

If you use TorchSOM in your research or work, please consider citing it using the following BibTeX entry:

```bibtex
Berthier, L. (2025). TorchSOM: A Scalable PyTorch-Compatible Library for Self-Organizing Maps. (Version 1.0.0) [Computer software]. https://github.com/LouisTier/TorchSOM
```

``` bash
@inproceedings{Berthier2025TorchSOM,
  title        = {TorchSOM: A Scalable PyTorch-Compatible Library for Self-Organizing Maps},
  author       = {Berthier, Louis},
#   booktitle    = {25th International Conference on Artificial Intelligence and Statistics},
  year         = {2025},
#   organization = {PMLR}
}
```



## Acknowledgement

The [Centre de Mathématiques Appliquées](https://cmap.ip-paris.fr/) (CMAP) at the Ecole Polytechnique (X), and [Manufacture Française des Pneumatiques Michelin](https://www.michelin.fr/) for the joint collaboration and supervision during my PhD thesis.

[Giuseppe Vettigli](https://github.com/JustGlowing) for his GitHub repository [MiniSom](https://github.com/JustGlowing/minisom), which provided a first well-maintained and open-source implementation of Self-Organizing Maps.

Logo created by me using DALL-E.



## Contributions

We invite contributors of all backgrounds and experience levels to get involved and contribute to this library. Whether you have innovative ideas to propose or are eager to submit pull requests, we encourage your contributions!

Please take a moment to read our [Code of Conduct](https://github.com/LouisTier/TorchSOM/CODE_OF_CONDUCT.md) if you're interested in contributing.



## License

TorchSOM is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use and modify the code as per the terms of the license.



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



