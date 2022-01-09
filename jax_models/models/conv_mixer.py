import jax.numpy as jnp
import flax.linen as nn

from typing import Optional
from ..layers import DepthwiseConv2D


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
    features: int = 512  # h
    patch_size: int = 7
    num_mixer_layers: int = 12  # depth
    filter_size: int = 9
    attach_head: bool = False
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


def ConvMixer_1536_20(attach_head=False, num_classes=1000, dropout=None):
    return ConvMixer(1536, 7, 20, 9, attach_head, num_classes)


def ConvMixer_768_32(attach_head=False, num_classes=1000, dropout=None):
    return ConvMixer(768, 7, 32, 7, attach_head, num_classes)


def ConvMixer_512_12(attach_head=False, num_classes=1000, dropout=None):
    return ConvMixer(512, 7, 12, 8, attach_head, num_classes)


def ConvMixer_1024_20(attach_head=False, num_classes=1000, dropout=None):
    return ConvMixer(1024, 14, 20, 9, attach_head, num_classes)
