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
  <a href="https://www.tensorflow.org/versions"><img alt="TensorFlow versions" src="https://img.shields.io/badge/tensorflow-2.7–2.15-f57000?style=flat"></a>
  <a href="https://scikit-learn.org/stable/whats_new.html"><img alt="scikit-learn versions" src="https://img.shields.io/badge/sklearn-1.0–1.3-f89939?style=flat"></a>
  <a href="https://www.python.org/downloads"><img alt="Python versions" src="https://img.shields.io/badge/python-3.7–3.11-blue?style=flat"></a>
  <a href="https://pypi.python.org/pypi/pidgan"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/pidgan"></a>
  <a href="LICENSE"><img alt="GitHub - License" src="https://img.shields.io/github/license/mbarbetti/pidgan"></a>
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

| Algorithms* | Avail | Test | Lipschitzianity** | Design inspired by | Tutorial |
|:-----------:|:-----:|:----:|:-----------------:|:------------------:|:--------:|
| [`GAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/GAN.py) | ✅ | ✅ | ❌ | [1][1], [8][8], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-GAN-LHCb_RICH.ipynb) |
| [`BceGAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/BceGAN.py) | ✅ | ✅ | ❌ | [2][2], [8][8], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN-LHCb_RICH.ipynb) |
| [`LSGAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/LSGAN.py) | ✅ | ✅ | ❌ | [3][3], [8][8], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-LSGAN-LHCb_RICH.ipynb) |
| [`WGAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/WGAN.py) | ✅ | ✅ | ✅ | [4][4], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN-LHCb_RICH.ipynb) |
| [`WGAN_GP`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/WGAN_GP.py) | ✅ | ✅ | ✅ | [5][5], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN_GP-LHCb_RICH.ipynb) |
| [`CramerGAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/CramerGAN.py) | ✅ | ✅ | ✅ | [6][6], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-CramerGAN-LHCb_RICH.ipynb) |
| [`WGAN_ALP`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/WGAN_ALP.py) | ✅ | ✅ | ✅ | [7][7], [9][9] |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN_ALP-LHCb_RICH.ipynb) |
| [`BceGAN_GP`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/BceGAN_GP.py) | ✅ | ✅ | ✅ | [2][2], [5][5], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN_GP-LHCb_RICH.ipynb) |
| [`BceGAN_ALP`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/BceGAN_ALP.py) | ✅ | ✅ | ✅ | [2][2], [7][7], [9][9] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN_ALP-LHCb_RICH.ipynb) |

*each GAN algorithm is designed to operate taking __conditions__ as input [[10][10]]

**the GAN training is regularized to ensure that the discriminator encodes a 1-Lipschitz function

### Generators

| Players | Avail | Test | Inherit from | Design inspired by |
|:-------:|:-----:|:----:|:------------:|:------------------:|
| [`Generator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/Generator.py) | ✅ | ✅ | [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) | [1][1], [10][10] |
| [`ResGenerator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/ResGenerator.py) | ✅ | ✅ | [`Generator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/Generator.py) | [1][1], [10][10], [11][11] |

### Discriminators

| Players | Avail | Test | Inherit from | Design inspired by |
|:-------:|:-----:|:----:|:------------:|:------------------:|
| [`Discriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/Discriminator.py) | ✅ | ✅ | [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) | [1][1], [9][9], [10][10] |
| [`ResDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/ResDiscriminator.py) | ✅ | ✅ | [`Discriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/Discriminator.py) | [1][1], [9][9], [10][10], [11][11] |
| [`AuxDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/AuxDiscriminator.py) | ✅ | ✅ | [`ResDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/ResDiscriminator.py) | [1][1], [9][9], [10][10], [11][11], [12][12] |

### Other players

| Players | Avail | Test | Inherit from |
|:-------:|:-----:|:----:|:------------:|
| [`Classifier`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/Classifier.py) | ✅ | ✅ | [`Discriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/Discriminator.py) |
| [`ResClassifier`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/ResClassifier.py) | ✅ | ✅ | [`ResDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/ResDiscriminator.py) |
| [`AuxClassifier`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/AuxClassifier.py) | ✅ | ✅ | [`AuxDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/AuxDiscriminator.py) |
| [`MultiClassifier`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/MultiClassifier.py) | ✅ | ✅ | [`Discriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/Discriminator.py) |
| [`MultiResClassifier`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/MultiResClassifier.py) | ✅ | ✅ | [`ResDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/ResDiscriminator.py) |
| [`AuxMultiClassifier`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/AuxMultiClassifier.py) | ✅ | ✅ | [`AuxDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/AuxDiscriminator.py) |

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
