import jax.numpy as jnp
import flax.linen as nn

from ..layers import DepthwiseConv2D
from .model_registry import register_model

from typing import Optional
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

__all__ = [
    "ConvMixer",
    "convmixer_512_12",
    "convmixer_768_32",
    "convmixer_1024_20",
    "convmixer_1536_20",
]


class ConvMixerLayer(nn.Module):
    features: int = 256
    filter_size: int = 3
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = inputs
        x = DepthwiseConv2D()(x)
        x = nn.gelu(x)
        x = nn.BatchNorm(deterministic)(x)

        add = x + inputs

        x = nn.Conv(self.features, (self.filter_size, self.filter_size))(add)
        x = nn.gelu(x)
        x = nn.BatchNorm(deterministic)(x)

        return x


class ConvMixer(nn.Module):
    """
    ConvMixer Module

    Attributes:
        features (int): Number of features. Default is 512.
        patch_size (int): Patch size. Default is 7.
        num_mixer_layers (int): Number of mixer layers. Default is 12.
        filter_size (int): Size of convolution filters. Default is 9.
        attach_head (bool): Whether to attach classification head. Default is True
        num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    features: int = 512
    patch_size: int = 7
    num_mixer_layers: int = 12
    filter_size: int = 9
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        extract_patches = nn.Conv(
            self.features, (self.patch_size, self.patch_size), self.patch_size
        )(inputs)
        x = nn.gelu(extract_patches)
        x = nn.BatchNorm(deterministic)(x)
        for _ in range(self.num_mixer_layers):
            x = ConvMixerLayer(self.features, self.filter_size)(x, deterministic)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


@register_model
def convmixer_1536_20(
    attach_head=False,
    num_classes=1000,
    dropout=None,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained model for ConvMixer isn't available. Loading un-trained model instead"
        )
    return ConvMixer(1536, 7, 20, 9, attach_head, num_classes, **kwargs)


@register_model
def convmixer_768_32(
    attach_head=False,
    num_classes=1000,
    dropout=None,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained model for ConvMixer isn't available. Loading un-trained model instead"
        )
    return ConvMixer(768, 7, 32, 7, attach_head, num_classes, **kwargs)


@register_model
def convmixer_512_12(
    attach_head=False,
    num_classes=1000,
    dropout=None,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained model for ConvMixer isn't available. Loading un-trained model instead"
        )
    return ConvMixer(512, 7, 12, 8, attach_head, num_classes, **kwargs)


@register_model
def convmixer_1024_20(
    attach_head=False,
    num_classes=1000,
    dropout=None,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained model for ConvMixer isn't available. Loading un-trained model instead"
        )
    return ConvMixer(1024, 14, 20, 9, attach_head, num_classes, **kwargs)
