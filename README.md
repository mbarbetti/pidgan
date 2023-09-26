<!--
<div align="center">
  <img alt="pidgan logo" src="https://raw.githubusercontent.com/mbarbetti/pidgan/main/.github/images/pidgan-logo.png" width="600"/>
</div>
-->

<h1 align="center">PIDGAN</h1>

<h2 align="center">
  <em>GAN-based models to fast-simulate the LHCb PID detectors</em>
</h2>

<p align="center">
  <a href="https://www.tensorflow.org/versions"><img alt="TensorFlow versions" src="https://img.shields.io/badge/tensorflow-2.7–2.13-f57000?style=flat"></a>
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

| Algorithms* | Implementation |  Test  |                         Paper                        |
|:-----------:|:--------------:|:------:|:----------------------------------------------------:|
|    `GAN`    |       ✅       |   ✅   |  [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)  |
|  `BceGAN`   |       ✅       |   ✅   |                                                      |
|   `LSGAN`   |       ✅       |   ✅   | [arXiv:1611.04076](https://arxiv.org/abs/1611.04076) |
|   `WGAN`    |       ✅       |   ✅   | [arXiv:1701.07875](https://arxiv.org/abs/1701.07875) |
|  `WGAN_GP`  |       ✅       |   ✅   | [arXiv:1704.00028](https://arxiv.org/abs/1704.00028) |
| `CramerGAN` |       ✅       |   ✅   | [arXiv:1705.10743](https://arxiv.org/abs/1705.10743) |
| `WGAN_ALP`  |       ✅       |   ✅   | [arXiv:1907.05681](https://arxiv.org/abs/1907.05681) |

*Designed to operate according to the **conditional version** proposed in [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)

<!--
#### Some tricks:
* [arXiv:1606.03498](https://arxiv.org/abs/1606.03498)
* [arXiv:1701.04862](https://arxiv.org/abs/1701.04862)
-->