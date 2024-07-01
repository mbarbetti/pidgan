<div align="center">
  <img alt="pidgan logo" src="https://raw.githubusercontent.com/mbarbetti/pidgan/main/.github/images/pidgan-logo-h-plain.png" width="300">
</div>

<h2 align="center">
  <em>GAN-based models to flash-simulate the LHCb PID detectors</em>
</h2>

<p align="center">
  <a href="https://www.tensorflow.org/versions"><img alt="TensorFlow versions" src="https://img.shields.io/badge/TensorFlow-2.8–2.16-f57000?style=flat&logo=tensorflow&logoColor=white"></a>
  <a href="https://keras.io/keras_3"><img alt="Keras 3" src="https://img.shields.io/badge/Keras_3-compatible-d00000?style=flat&logo=keras&logoColor=white"></a>
  <a href="https://scikit-learn.org/stable/whats_new.html"><img alt="scikit-learn versions" src="https://img.shields.io/badge/sklearn-1.0–1.5-f89939?style=flat&logo=scikit-learn&logoColor=white"></a>
  <a href="https://www.python.org/downloads"><img alt="Python versions" src="https://img.shields.io/badge/python-3.7–3.12-blue?style=flat&logo=python&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/pidgan"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/pidgan"></a>
  <a href="https://pypi.python.org/pypi/pidgan"><img alt="PyPI - Status" src="https://img.shields.io/pypi/status/pidgan"></a>
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

### What is PIDGAN?

PIDGAN is a Python package built upon TensorFlow 2 to provide ready-to-use implementations for several GAN algorithms (listed in [this table](#generative-adversarial-networks)). The package was originally designed to simplify the training and optimization of GAN-based models for the _Particle Identification_ (PID) system of the LHCb experiment. Today, PIDGAN is a versatile package that can be employed in a wide range of _High Energy Physics_ (HEP) applications and, in general, whenever one has anything to do with tabular data and aims to learn the conditional probability distributions of a set of target features. This package is one of the building blocks to define a _Flash Simulation_ framework of the LHCb experiment [1].

#### PIDGAN is (almost) all you need (for flash-simulation)

Standard techniques for simulations consume tons of CPU hours in reproducing _all_ the radiation-matter interactions occurring within a HEP detector when traversed by primary and secondary particles. Directly transforming generated particles into analysis-level objects allows Flash Simulation strategies to speed up significantly the simulation production, up to x1000 [1]. Such transformations can be defined by using _Generative Adversarial Networks_ (GAN) [[2][gan]] trained to take into account the kinematics of the traversing particles and the detection conditions (e.g., magnet polarity, occupancy).

GANs rely on the simultaneous (adversarial) training of two neural networks called _generator_ and _discriminator_, whose competition ends once reached the [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium). At this point, the generator can be used as simulator to generate new data according to the conditional probability distributions learned during the training [[3][cgan]]. By relying on the TensorFlow and Keras APIs, PIDGAN allows to define and train a GAN model with no more than 20 code lines.

```python
from pidgan.players.generators import Generator
from pidgan.players.discriminators import Discriminator
from pidgan.algorithms import GAN

x = ... # conditions
y = ... # targets

G = Generator(
  output_dim=y.shape[1],
  latent_dim=64,
  output_activation="linear",
)

D = Discriminator(
  output_dim=1,
  output_activation="sigmoid",
)

model = GAN(generator=G, discriminator=D)
model.compile(
  metrics=["accuracy"],
  generator_optimizer="rmsprop",
  discriminator_optimizer="rmsprop",
)

model.fit(x, y, batch_size=256, epochs=100)
```

### Installation guide

#### First steps

Before installing PIDGAN, we suggest preparing a fully operational TensorFlow installation by following the instructions described in the [dedicated guide](https://www.tensorflow.org/install). If your device is equipped with one of the NVIDIA GPU cards supported by TensorFlow (see [Hardware requirements](https://www.tensorflow.org/install/pip#hardware_requirements)), do not forget to verify the correct installation of the libraries for hardware acceleration by running:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If the equipped GPU card is not included in the list printed by the previous command, your device and/or Python environment may be misconfigured. Please refer to [this table](https://www.tensorflow.org/install/source#gpu) for the correct configuration of CUDA Toolkit and cuDNN requested by the different TensorFlow versions.

#### How to install

PIDGAN has a minimal list of [requirements](https://github.com/mbarbetti/pidgan/blob/main/requirements/base.txt):

- Python >= 3.7, < 3.13
- TensorFlow >= 2.8, < 2.17
- scikit-learn >= 1.0, < 1.6
- NumPy < 2.0
- Hopaas client (https://github.com/landerlini/hopaas_client)

The easiest way to install PIDGAN is via `pip`:

```bash
pip install pidgan
```

In addition, since `hopaas_client` is not available on [PyPI](https://pypi.org), you need to install it manually to unlock the complete set of PIDGAN functionalities:

```bash
pip install git+https://github.com/landerlini/hopaas_client
```

#### Optional dependencies

Standard HEP applications may need additional packages for data management, results visualization/validation, and model export. PIDGAN and any [additional requirements](https://github.com/mbarbetti/pidgan/blob/main/requirements/hep.txt) potentially useful in HEP can be installed via `pip` in one shot:

```bash
pip install pidgan[hep]
```

### Models available

The main components of PIDGAN are the [`algorithms`](https://github.com/mbarbetti/pidgan/tree/main/src/pidgan/algorithms) and [`players`](https://github.com/mbarbetti/pidgan/tree/main/src/pidgan/players) modules that provide, respectively, implementations for several GAN algorithms and the so-called adversarial neural networks (e.g., generator, discriminator). The objects exposed by the `algorithms` and `players` modules are implemented by subclassing the Keras [Model class](https://keras.io/api/models/model) and customizing the training procedure that is executed when one calls the `fit()` method. With [PIDGAN v0.2.0](https://github.com/mbarbetti/pidgan/releases/tag/v0.2.0) the package has been massively rewritten to be also compatible with the new multi-backend [Keras 3](https://keras.io/keras_3). At the moment, the custom training procedures defined for the various GAN algorithms are only implemented for the TensorFlow backend, while relying also on the Pytorch and Jax backends is planned for a future release. The following tables report the complete set of `algorithms` and `players` classes currently available, together with a snapshot of their implementation details.

#### Generative Adversarial Networks

| Algorithms* | Source | Avail | Test | Lipschitz** | Refs | Tutorial |
|:-----------:|:------:|:-----:|:----:|:-----------:|:----:|:--------:|
| GAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/GAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/GAN.py) | ✅ | ✅ | ❌ | [2][gan], [10][pre-wgan], [11][gan-tricks] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-GAN-LHCb_RICH.ipynb) |
| BceGAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/BceGAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/BceGAN.py) | ✅ | ✅ | ❌ | [4][dcgan], [10][pre-wgan], [11][gan-tricks] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN-LHCb_RICH.ipynb) |
| LSGAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/LSGAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/LSGAN.py) | ✅ | ✅ | ❌ | [5][lsgan], [10][pre-wgan], [11][gan-tricks] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-LSGAN-LHCb_RICH.ipynb) |
| WGAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/WGAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/WGAN.py) | ✅ | ✅ | ✅ | [6][wgan], [11][gan-tricks] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN-LHCb_RICH.ipynb) |
| WGAN-GP | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/WGAN_GP.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/WGAN_GP.py) | ✅ | ✅ | ✅ | [7][wgan-gp], [11][gan-tricks] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN_GP-LHCb_RICH.ipynb) |
| CramerGAN | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/CramerGAN.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/CramerGAN.py) | ✅ | ✅ | ✅ | [8][cramer-gan], [11][gan-tricks] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-CramerGAN-LHCb_RICH.ipynb) |
| WGAN-ALP | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/WGAN_ALP.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/WGAN_ALP.py) | ✅ | ✅ | ✅ | [9][wgan-alp], [11][gan-tricks] |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-WGAN_ALP-LHCb_RICH.ipynb) |
| BceGAN-GP | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/BceGAN_GP.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/BceGAN_GP.py) | ✅ | ✅ | ✅ | [4][dcgan], [7][wgan-gp], [11][gan-tricks] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN_GP-LHCb_RICH.ipynb) |
| BceGAN-ALP | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k2/BceGAN_ALP.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/algorithms/k3/BceGAN_ALP.py) | ✅ | ✅ | ✅ | [4][dcgan], [9][wgan-alp], [11][gan-tricks] | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbarbetti/pidgan-notebooks/blob/main/tutorial-BceGAN_ALP-LHCb_RICH.ipynb) |

*each GAN algorithm is designed to operate taking __conditions__ as input [[3][cgan]]

**the GAN training is regularized to ensure that the discriminator encodes a 1-Lipschitz function

#### Generators

| Players | Source | Avail | Test | Skip conn | Refs |
|:-------:|:------:|:-----:|:----:|:---------:|:----:|
| Generator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/k2/Generator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/k3/Generator.py) | ✅ | ✅ | ❌ | [2][gan], [3][cgan] |
| ResGenerator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/k2/ResGenerator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/generators/k3/ResGenerator.py) | ✅ | ✅ | ✅ | [2][gan], [3][cgan], [12][skip-conn] |

#### Discriminators

| Players | Source | Avail | Test | Skip conn | Aux proc | Refs |
|:-------:|:------:|:-----:|:----:|:---------:|:--------:|:----:|
| Discriminator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k2/Discriminator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k3/Discriminator.py) | ✅ | ✅ | ❌ | ❌ | [2][gan], [3][cgan], [11][gan-tricks] |
| ResDiscriminator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k2/ResDiscriminator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k3/ResDiscriminator.py) | ✅ | ✅ | ✅ | ❌ | [2][gan], [3][cgan], [11][gan-tricks], [12][skip-conn] |
| AuxDiscriminator | [`k2`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k2/AuxDiscriminator.py)/[`k3`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/discriminators/k3/AuxDiscriminator.py) | ✅ | ✅ | ✅ | ✅ | [2][gan], [3][cgan], [11][gan-tricks], [12][skip-conn], [13][aux-feat] |

#### Other players

| Players | Source | Avail | Test | Skip conn | Aux proc | Multiclass |
|:-------:|:------:|:-----:|:----:|:---------:|:--------:|:---------:|
| Classifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/Classifier.py) | ✅ | ✅ | ❌ | ❌ | ❌ |
| ResClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/ResClassifier.py) | ✅ | ✅ | ✅ | ❌ | ❌ |
| AuxClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/AuxClassifier.py) | ✅ | ✅ | ✅ | ✅ | ❌ |
| MultiClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/MultiClassifier.py) | ✅ | ✅ | ❌ | ❌ | ✅ |
| MultiResClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/MultiResClassifier.py) | ✅ | ✅ | ✅ | ❌ | ✅ |
| AuxMultiClassifier | [`src`](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/players/classifiers/AuxMultiClassifier.py) | ✅ | ✅ | ✅ | ✅ | ✅ |

### References

1. M. Barbetti, "The flash-simulation paradigm and its implementation based on Deep Generative Models for the LHCb experiment at CERN", PhD thesis, University of Firenze, 2024
2. I.J. Goodfellow _et al._, "Generative Adversarial Networks", [arXiv:1406.2661][gan]
3. M. Mirza, S. Osindero, "Conditional Generative Adversarial Nets", [arXiv:1411.1784][cgan]
4. A. Radford, L. Metz, S. Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", [arXiv:1511.06434][dcgan]
5. X. Mao _et al._, "Least Squares Generative Adversarial Networks", [arXiv:1611.04076][lsgan]
6. M. Arjovsky, S. Chintala, L. Bottou, "Wasserstein GAN", [arXiv:1701.07875][wgan]
7. I. Gulrajani _et al._, "Improved Training of Wasserstein GANs", [arXiv:1704.00028][wgan-gp]
8. M.G. Bellemare _et al._, "The Cramer Distance as a Solution to Biased Wasserstein Gradients", [arXiv:1705.10743][cramer-gan]
9. D. Terjék, "Adversarial Lipschitz Regularization", [arXiv:1907.05681][wgan-alp]
10. M. Arjovsky, L. Bottou, "Towards Principled Methods for Training Generative Adversarial Networks", [arXiv:1701.04862][pre-wgan]
11. T. Salimans _et al._, "Improved Techniques for Training GANs", [arXiv:1606.03498][gan-tricks]
12. K. He _et al._, "Deep Residual Learning for Image Recognition", [arXiv:1512.03385][skip-conn]
13. A. Rogachev, F. Ratnikov, "GAN with an Auxiliary Regressor for the Fast Simulation of the Electromagnetic Calorimeter Response", [arXiv:2207.06329][aux-feat]

### Credits

Most of the GAN algorithms are an evolution of what provided by the [mbarbetti/tf-gen-models](https://github.com/mbarbetti/tf-gen-models) repository. The BceGAN model is freely inspired by the TensorFlow tutorial [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan) and the Keras tutorial [Conditional GAN](https://keras.io/examples/generative/conditional_gan). The WGAN-ALP model is an adaptation of what provided by the [dterjek/adversarial_lipschitz_regularization](https://github.com/dterjek/adversarial_lipschitz_regularization) repository.

### Citing PIDGAN

To cite this repository:

```bibtex
@software{pidgan:2023abc,
  author    = "Matteo Barbetti and Lucio Anderlini",
  title     = "{PIDGAN: GAN-based models to flash-simulate the LHCb PID detectors}",
  version   = "v0.2.0",
  url       = "https://github.com/mbarbetti/pidgan",
  doi       = "10.5281/zenodo.10463728",
  publisher = "Zenodo",
  year      = "2023",
}
```

In the above bibtex entry, the version number is intended to be that from [pidgan/version.py](https://github.com/mbarbetti/pidgan/blob/main/src/pidgan/version.py), while the year corresponds to the project's open-source release.

### License

PIDGAN has a GNU General Public License v3 (GPLv3), as found in the [LICENSE](https://github.com/mbarbetti/pidgan/blob/main/LICENSE) file.

[gan]: https://arxiv.org/abs/1406.2661
[dcgan]: https://arxiv.org/abs/1511.06434
[lsgan]: https://arxiv.org/abs/1611.04076
[wgan]: https://arxiv.org/abs/1701.07875
[wgan-gp]: https://arxiv.org/abs/1704.00028
[cramer-gan]: https://arxiv.org/abs/1705.10743
[wgan-alp]: https://arxiv.org/abs/1907.05681
[pre-wgan]: https://arxiv.org/abs/1701.04862
[gan-tricks]: https://arxiv.org/abs/1606.03498
[cgan]: https://arxiv.org/abs/1411.1784
[skip-conn]: https://arxiv.org/abs/1512.03385
[aux-feat]: https://arxiv.org/abs/2207.06329
