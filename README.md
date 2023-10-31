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
  <a href="https://www.tensorflow.org/versions"><img alt="TensorFlow versions" src="https://img.shields.io/badge/tensorflow-2.7–2.14-f57000?style=flat"></a>
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
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<!--
[![Docker - Version](https://img.shields.io/docker/v/mbarbetti/pidgan?label=docker)](https://hub.docker.com/r/mbarbetti/pidgan)
-->

### Generative Adversarial Networks

| Algorithms* | Avail | Test | Lipschitzianity** | Design inspired by | Tutorial |
|:-----------:|:-----:|:----:|:-----------------:|:------------------:|:--------:|
| [`GAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/GAN.py) | ✅ | ✅ | ❌ | [1](https://arxiv.org/abs/1406.2661), [8](https://arxiv.org/abs/1701.04862), [9](https://arxiv.org/abs/1606.03498) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-GAN-LHCb_RICH.ipynb) |
| [`BceGAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/BceGAN.py) | ✅ | ✅ | ❌ | [2](https://arxiv.org/abs/1511.06434), [8](https://arxiv.org/abs/1701.04862), [9](https://arxiv.org/abs/1606.03498) | 🛠️ |
| [`BceGAN_GP`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/BceGAN_GP.py) | ✅ | ✅ | ✅ | [2](https://arxiv.org/abs/1511.06434), [5](https://arxiv.org/abs/1704.00028), [9](https://arxiv.org/abs/1606.03498) | 🛠️ |
| [`BceGAN_ALP`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/BceGAN_ALP.py) | ✅ | ✅ | ✅ | [2](https://arxiv.org/abs/1511.06434), [7](https://arxiv.org/abs/1907.05681), [9](https://arxiv.org/abs/1606.03498) | 🛠️ |
| [`LSGAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/LSGAN.py) | ✅ | ✅ | ❌ | [3](https://arxiv.org/abs/1611.04076), [8](https://arxiv.org/abs/1701.04862), [9](https://arxiv.org/abs/1606.03498) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-LSGAN-LHCb_RICH.ipynb) |
| [`WGAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/WGAN.py) | ✅ | ✅ | ✅ | [4](https://arxiv.org/abs/1701.07875), [9](https://arxiv.org/abs/1606.03498) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN-LHCb_RICH.ipynb) |
| [`WGAN_GP`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/WGAN_GP.py) | ✅ | ✅ | ✅ | [5](https://arxiv.org/abs/1704.00028), [9](https://arxiv.org/abs/1606.03498) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN_GP-LHCb_RICH.ipynb) |
| [`CramerGAN`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/CramerGAN.py) | ✅ | ✅ | ✅ | [6](https://arxiv.org/abs/1705.10743), [9](https://arxiv.org/abs/1606.03498) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-CramerGAN-LHCb_RICH.ipynb) |
| [`WGAN_ALP`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/WGAN_ALP.py) | ✅ | ✅ | ✅ | [7](https://arxiv.org/abs/1907.05681), [9](https://arxiv.org/abs/1606.03498) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN_ALP-LHCb_RICH.ipynb) |

*each GAN algorithm is designed to operate taking __conditions__ as input [[10](https://arxiv.org/abs/1411.1784)]

**the GAN training is regularized to ensure that the discriminator encodes a 1-Lipschitz function

### Generators

| Players | Avail | Test | Design inspired by |
|:-------:|:-----:|:----:|:------------------:|
| [`Generator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/Generator.py) | ✅ | ✅ | [1](https://arxiv.org/abs/1406.2661), [10](https://arxiv.org/abs/1411.1784) |

### Discriminators

| Players | Avail | Test | Design inspired by |
|:-------:|:-----:|:----:|:------------------:|
| [`Discriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/Discriminator.py) | ✅ | ✅ | [1](https://arxiv.org/abs/1406.2661), [9](https://arxiv.org/abs/1606.03498), [10](https://arxiv.org/abs/1411.1784) |
| [`AuxDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/AuxDiscriminator.py) | ✅ | ✅ | [1](https://arxiv.org/abs/1406.2661), [9](https://arxiv.org/abs/1606.03498), [10](https://arxiv.org/abs/1411.1784), [11](https://arxiv.org/abs/2207.06329) |

### Other players

| Players | Avail | Test | Inherit from |
|:-------:|:-----:|:----:|:------------:|
| [`Classifier`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/Classifier.py) | ✅ | ✅ | [`Discriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/Discriminator.py) |
| `AuxClassifier` | 🛠️ | ❌ | [`AuxDiscriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/AuxDiscriminator.py) |
| [`MultiClassifier`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/MultiClassifier.py) | ✅ | ✅ | [`Discriminator`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/Discriminator.py) |

### References
1. I.J. Goodfellow _et al._, "Generative Adversarial Networks", [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
2. A. Radford, L. Metz, S. Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
3. X. Mao _et al._, "Least Squares Generative Adversarial Networks", [arXiv:1611.04076](https://arxiv.org/abs/1611.04076)
4. M. Arjovsky, S. Chintala, L. Bottou, "Wasserstein GAN", [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
5. I. Gulrajani _et al._, "Improved Training of Wasserstein GANs", [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
6. M.G. Bellemare _et al._, "The Cramer Distance as a Solution to Biased Wasserstein Gradients", [arXiv:1705.10743](https://arxiv.org/abs/1705.10743)
7. D. Terjék, "Adversarial Lipschitz Regularization", [arXiv:1907.05681](https://arxiv.org/abs/1907.05681)
8. M. Arjovsky, L. Bottou, "Towards Principled Methods for Training Generative Adversarial Networks", [arXiv:1701.04862](https://arxiv.org/abs/1701.04862)
9. T. Salimans _et al._, "Improved Techniques for Training GANs", [arXiv:1606.03498](https://arxiv.org/abs/1606.03498)
10. M. Mirza, S. Osindero, "Conditional Generative Adversarial Nets", [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
11. A. Rogachev, F. Ratnikov, "GAN with an Auxiliary Regressor for the Fast Simulation of the Electromagnetic Calorimeter Response", [arXiv:2207.06329](https://arxiv.org/abs/2207.06329)

### Credits
Most of the GAN algorithms are an evolution of what provided by the [mbarbetti/tf-gen-models](https://github.com/mbarbetti/tf-gen-models) repository. The `BceGAN` model is freely inspired by the TensorFlow tutorial [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan) and the Keras tutorial [Conditional GAN](https://keras.io/examples/generative/conditional_gan). The `WGAN_ALP` model is an adaptation of what provided by the [dterjek/adversarial_lipschitz_regularization](https://github.com/dterjek/adversarial_lipschitz_regularization) repository.
