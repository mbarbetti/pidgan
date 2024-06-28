<!--
<div align="center">
  <img alt="pidgan logo" src="https://raw.githubusercontent.com/mbarbetti/pidgan/main/.github/images/pidgan-logo.png" width="600"/>
</div>
-->

<h1 align="center">PIDGAN</h1>

<h2 align="center">
  <em>GAN-based models to flash-simulate the LHCb PID detectors</em>
</h2>

<p align="center">
  <a href="https://www.tensorflow.org/versions"><img alt="TensorFlow versions" src="https://img.shields.io/badge/TensorFlow-2.8–2.16-f57000?style=flat&logo=tensorflow&logoColor=white"></a>
  <a href="https://keras.io/keras_3"><img alt="Keras 3" src="https://img.shields.io/badge/Keras_3-compatible-d00000?style=flat&logo=keras&logoColor=white"></a>
  <a href="https://scikit-learn.org/stable/whats_new.html"><img alt="scikit-learn versions" src="https://img.shields.io/badge/sklearn-1.0–1.5-f89939?style=flat&logo=scikit-learn&logoColor=white"></a>
  <a href="https://www.python.org/downloads"><img alt="Python versions" src="https://img.shields.io/badge/python-3.7–3.11-blue?style=flat&logo=python&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/pidgan"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/pidgan"></a>
  <a href="LICENSE"><img alt="GitHub - License" src="https://img.shields.io/github/license/mbarbetti/pidgan"></a>
  <a href="https://zenodo.org/doi/10.5281/zenodo.10463727"><img alt="DOI" src="https://zenodo.org/badge/597088032.svg"></a>
</p>

<p align="center">
  <a href="https://github.com/mbarbetti/pidgan/actions/workflows/tests.yml"><img alt="GitHub - Tests" src="https://github.com/mbarbetti/pidgan/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/mbarbetti/pidgan"><img alt="Codecov" src="https://codecov.io/gh/mbarbetti/pidgan/branch/main/graph/badge.svg?token=ZLWDgWhnkq"></a>
</p>

<p align="center">
  <a href="https://github.com/mbarbetti/pidgan/actions/workflows/style.yml"><img alt="GitHub - Style" src="https://github.com/mbarbetti/pidgan/actions/workflows/style.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" style="max-width:100%;"></a>
</p>

<!--
[![Docker - Version](https://img.shields.io/docker/v/mbarbetti/pidgan?label=docker)](https://hub.docker.com/r/mbarbetti/pidgan)
-->

### Generative Adversarial Networks

| Algorithms* | Source | Avail | Test | Lipschitzianity** | Refs | Tutorial |
|:-----------:|:------:|:-----:|:----:|:-----------------:|:----:|:--------:|
| GAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/GAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/GAN.py) | ✅ | ✅ | ❌ | [1][1], [8][8], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-GAN-LHCb_RICH.ipynb) |
| BceGAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/BceGAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/BceGAN.py) | ✅ | ✅ | ❌ | [2][2], [8][8], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN-LHCb_RICH.ipynb) |
| LSGAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/LSGAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/LSGAN.py) | ✅ | ✅ | ❌ | [3][3], [8][8], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-LSGAN-LHCb_RICH.ipynb) |
| WGAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/WGAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/WGAN.py) | ✅ | ✅ | ✅ | [4][4], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN-LHCb_RICH.ipynb) |
| WGAN-GP | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/WGAN_GP.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/WGAN_GP.py) | ✅ | ✅ | ✅ | [5][5], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN_GP-LHCb_RICH.ipynb) |
| CramerGAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/CramerGAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/CramerGAN.py) | ✅ | ✅ | ✅ | [6][6], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-CramerGAN-LHCb_RICH.ipynb) |
| WGAN-ALP | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/WGAN_ALP.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/WGAN_ALP.py) | ✅ | ✅ | ✅ | [7][7], [9][9] |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN_ALP-LHCb_RICH.ipynb) |
| BceGAN-GP | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/BceGAN_GP.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/BceGAN_GP.py) | ✅ | ✅ | ✅ | [2][2], [5][5], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN_GP-LHCb_RICH.ipynb) |
| BceGAN-ALP | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/BceGAN_ALP.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/BceGAN_ALP.py) | ✅ | ✅ | ✅ | [2][2], [7][7], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN_ALP-LHCb_RICH.ipynb) |

*each GAN algorithm is designed to operate taking __conditions__ as input [[10][10]]

**the GAN training is regularized to ensure that the discriminator encodes a 1-Lipschitz function

### Generators

| Players | Source | Avail | Test | Skip conn | Refs |
|:-------:|:------:|:-----:|:----:|:---------:|:----:|
| Generator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/k2/Generator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/k3/Generator.py) | ✅ | ✅ | ❌ | [1][1], [10][10] |
| ResGenerator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/k2/ResGenerator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/k3/ResGenerator.py) | ✅ | ✅ | ✅ | [1][1], [10][10], [11][11] |

### Discriminators

| Players | Source | Avail | Test | Skip conn | Aux proc | Refs |
|:-------:|:------:|:-----:|:----:|:---------:|:--------:|:----:|
| Discriminator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k2/Discriminator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k3/Discriminator.py) | ✅ | ✅ | ❌ | ❌ | [1][1], [9][9], [10][10] |
| ResDiscriminator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k2/ResDiscriminator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k3/ResDiscriminator.py) | ✅ | ✅ | ✅ | ❌ | [1][1], [9][9], [10][10], [11][11] |
| AuxDiscriminator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k2/AuxDiscriminator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k3/AuxDiscriminator.py) | ✅ | ✅ | ✅ | ✅ | [1][1], [9][9], [10][10], [11][11], [12][12] |

### Other players

| Players | Source | Avail | Test | Skip conn | Aux proc | Multiclass |
|:-------:|:------:|:-----:|:----:|:---------:|:--------:|:---------:|
| Classifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/Classifier.py) | ✅ | ✅ | ❌ | ❌ | ❌ |
| ResClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/ResClassifier.py) | ✅ | ✅ | ✅ | ❌ | ❌ |
| AuxClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/AuxClassifier.py) | ✅ | ✅ | ✅ | ✅ | ❌ |
| MultiClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/MultiClassifier.py) | ✅ | ✅ | ❌ | ❌ | ✅ |
| MultiResClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/MultiResClassifier.py) | ✅ | ✅ | ✅ | ❌ | ✅ |
| AuxMultiClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/AuxMultiClassifier.py) | ✅ | ✅ | ✅ | ✅ | ✅ |

### References
1. I.J. Goodfellow _et al._, "Generative Adversarial Networks", [arXiv:1406.2661][1]
2. A. Radford, L. Metz, S. Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", [arXiv:1511.06434][2]
3. X. Mao _et al._, "Least Squares Generative Adversarial Networks", [arXiv:1611.04076][3]
4. M. Arjovsky, S. Chintala, L. Bottou, "Wasserstein GAN", [arXiv:1701.07875][4]
5. I. Gulrajani _et al._, "Improved Training of Wasserstein GANs", [arXiv:1704.00028][5]
6. M.G. Bellemare _et al._, "The Cramer Distance as a Solution to Biased Wasserstein Gradients", [arXiv:1705.10743][6]
7. D. Terjék, "Adversarial Lipschitz Regularization", [arXiv:1907.05681][7]
8. M. Arjovsky, L. Bottou, "Towards Principled Methods for Training Generative Adversarial Networks", [arXiv:1701.04862][8]
9. T. Salimans _et al._, "Improved Techniques for Training GANs", [arXiv:1606.03498][9]
10. M. Mirza, S. Osindero, "Conditional Generative Adversarial Nets", [arXiv:1411.1784][10]
11. K. He _et al._, "Deep Residual Learning for Image Recognition", [arXiv:1512.03385][11]
12. A. Rogachev, F. Ratnikov, "GAN with an Auxiliary Regressor for the Fast Simulation of the Electromagnetic Calorimeter Response", [arXiv:2207.06329][12]

[1]: https://arxiv.org/abs/1406.2661
[2]: https://arxiv.org/abs/1511.06434
[3]: https://arxiv.org/abs/1611.04076
[4]: https://arxiv.org/abs/1701.07875
[5]: https://arxiv.org/abs/1704.00028
[6]: https://arxiv.org/abs/1705.10743
[7]: https://arxiv.org/abs/1907.05681
[8]: https://arxiv.org/abs/1701.04862
[9]: https://arxiv.org/abs/1606.03498
[10]: https://arxiv.org/abs/1411.1784
[11]: https://arxiv.org/abs/1512.03385
[12]: https://arxiv.org/abs/2207.06329

### Credits
Most of the GAN algorithms are an evolution of what provided by the [mbarbetti/tf-gen-models](https://github.com/mbarbetti/tf-gen-models) repository. The `BceGAN` model is freely inspired by the TensorFlow tutorial [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan) and the Keras tutorial [Conditional GAN](https://keras.io/examples/generative/conditional_gan). The `WGAN_ALP` model is an adaptation of what provided by the [dterjek/adversarial_lipschitz_regularization](https://github.com/dterjek/adversarial_lipschitz_regularization) repository.
