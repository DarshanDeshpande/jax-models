<h1> JAX Models </h1>

<!-- PROJECT SHIELDS -->
![license-shield]
![release-shield]
![python-shield]
![code-style]

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The <b>JAX Models</b> repository aims to provide open sourced JAX/Flax implementations for research papers originally without code or code written with frameworks other than JAX. The goal of this project is to make a collection of models, layers, activations and other utilities that are most commonly used for research. All papers and derived or translated code is cited in either the README or the docstrings. If you think that any citation is missed then please raise an issue.

All implementations provided here are available on <a href="https://www.paperswithcode.com">Papers With Code</a>.

<br>
Available model implementations for JAX are:

1. <a href="https://arxiv.org/abs/2111.11418">MetaFormer is Actually What You Need for Vision</a> (Weihao Yu et al., 2021)
2. <a href="https://arxiv.org/abs/2112.13692v1">Augmenting Convolutional networks with attention-based aggregation</a> (Hugo Touvron et al., 2021)
3. <a href="https://arxiv.org/abs/2112.11010">MPViT : Multi-Path Vision Transformer for Dense Prediction</a> (Youngwan Lee et al., 2021)
4. <a href="https://arxiv.org/abs/2105.01601v1">MLP-Mixer: An all-MLP Architecture for Vision</a> (Ilya Tolstikhin et al., 2021)
5. <a href="https://openreview.net/pdf?id=TVHS5Y4dNvM">Patches Are All You Need</a> (Anonymous et al., 2021)
6. <a href="https://arxiv.org/abs/2105.15203">SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers</a> (Enze Xie et al., 2021)
7. <a href="https://arxiv.org/abs/2201.03545">A ConvNet for the 2020s</a> (Zhuang Liu et al., 2021)
8. <a href="https://arxiv.org/abs/2111.06377v1">Masked Autoencoders Are Scalable Vision Learners</a> (Kaiming He et al., 2021)
9. <a href="https://arxiv.org/abs/2103.14030">Swin Transformer: Hierarchical Vision Transformer using Shifted Windows</a> (Ze Liu et al., 2021)
10. <a href="https://arxiv.org/abs/2102.12122">Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions</a> (Wenhai Wang et al., 2021)
11. <a href="https://arxiv.org/abs/2103.17239">Going deeper with Image Transformers</a> (Hugo Touvron et al., 2021)
12. <a href="https://arxiv.org/abs/2202.09741">Visual Attention Network</a> (Meng-Hao Guo et al., 2022)

<br>
Available layers for out-of-the-box integration:

1. <a href="https://arxiv.org/abs/1603.09382">DropPath (Stochastic Depth)</a> (Gao Huang et al., 2021)
2. <a href="https://arxiv.org/abs/1709.01507">Squeeze-and-Excitation Layer</a> (Jie Hu et al. 2019)
3. <a href="https://arxiv.org/abs/1610.02357v3"> Depthwise Convolution </a> (François Chollet, 2017)

<!-- PREREQUISITES -->
## Prerequisites

Prerequisites can be installed separately through the `requirements.txt` file in the main directory using:

```sh
pip install -r requirements.txt
```
The use of a virtual environment is highly recommended to avoid version incompatibilites.

<!-- INSTALLATION -->
## Installation

This project is built with Python 3 for the latest JAX/Flax versions and can be directly installed via pip.
```sh
pip install jax-models
```
If you wish to use the latest version then you can directly clone the repository too.
```sh
git clone https://github.com/DarshanDeshpande/jax-models.git
```

<!-- USAGE -->
## Usage

To see all model architectures available:

```py
from jax_models import list_models
from pprint import pprint

pprint(list_models())
```

To load your desired model:

```py
from jax_models import load_model
load_model('swin-tiny-224', attach_head=True, num_classes=1000, dropout=0.0, pretrained=True)
```

Note: It is necessary to pass `attach_head=True` and `num_classes` while loading pretrained models.


<!-- CONTRIBUTING -->
## Contributing

Please raise an issue if any implementation gives incorrect results, crashes unexpectedly during training/inference or if any citation is missing.

You can contribute to `jax_models` by supporting me with compute resources or by contributing your own resources to provide pretrained weights. 

If you wish to donate to this inititative then please drop me a mail <a href="https://mail.google.com/mail/u/0/?view=cm&fs=1&to=darshan.g.deshpande@gmail.com&tf=1">here</a>.
<br>

<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Feel free to reach out for any issues or requests related to these implementations

Darshan Deshpande - [Email](https://mail.google.com/mail/u/0/?view=cm&fs=1&to=darshan.g.deshpande@gmail.com&tf=1) | [Twitter](https://www.twitter.com/getdarshan) | [LinkedIn](https://www.linkedin.com/in/darshan-deshpande/) 





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/badge/LICENSE-Apache_2.0-magenta?style=for-the-badge
[python-shield]: https://img.shields.io/badge/PYTHON-3.6+-blue?style=for-the-badge
[release-shield]: https://img.shields.io/badge/Build-Alpha-red?style=for-the-badge
[code-style]: https://img.shields.io/badge/Code_Style-Black-black?style=for-the-badge
