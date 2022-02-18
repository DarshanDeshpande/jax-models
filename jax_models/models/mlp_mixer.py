import jax.numpy as jnp

import flax.linen as nn
from ..layers import TransformerMLP
from .model_registry import register_model

from typing import Optional
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


__all__ = [
    "MLPMixer",
    "mlpmixer_s16",
    "mlpmixer_s32",
    "mlpmixer_b32",
    "mlpmixer_l16",
    "mlpmixer_l32",
    "mlpmixer_h14",
]


class MixerLayer(nn.Module):
    channels_dim: int = 256
    tokens_dim: int = 256
    dropout: float = 0.2
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        norm = nn.LayerNorm()(inputs)
        transpose = jnp.swapaxes(norm, 1, 2)

        mlp1 = TransformerMLP(self.tokens_dim, transpose.shape[-1], self.dropout)(
            transpose, deterministic
        )
        transpose = jnp.swapaxes(mlp1, 1, 2)
        skip = inputs + transpose

        norm = nn.LayerNorm()(skip)
        mlp2 = TransformerMLP(self.channels_dim, norm.shape[-1], self.dropout)(
            norm, deterministic
        )
        skip = skip + mlp2

        return skip


class MLPMixer(nn.Module):
    """
    MLP Mixer Module

    Attributes:
        patch_size (int): Patch size. Default is 32.
        num_mixers_layers (int): Number of mixer layers. Default is 2.
        hidden_size (int): Hidden embedding size. Default is 768.
        channels_dim (int): Channels dimension for MLP. Default is 256.
        tokens_dim (int): Embedding dimension for tokens. Default is 256.
        dropout (float): Dropout value. Default is 0.2.
        attach_head (bool): Whether to attach classification head. Default is True.
        num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    patch_size: int = 32
    num_mixers_layers: int = 2
    hidden_size: int = 768
    channels_dim: int = 256
    tokens_dim: int = 256
    dropout: float = 0.2
    attach_head: bool = False
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.Conv(
            self.hidden_size,
            (self.patch_size, self.patch_size),
            strides=self.patch_size,
        )(inputs)
        batch, height, width, channels = x.shape
        x = jnp.reshape(x, (batch, height * width, channels))

        for _ in range(self.num_mixers_layers):
            x = MixerLayer(self.channels_dim, self.tokens_dim, self.dropout)(
                x, deterministic
            )

        if self.attach_head:
            x = jnp.mean(x, -1)
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x, -1)
        return x


@register_model
def mlpmixer_s32(
    attach_head=False,
    num_classes=1000,
    dropout=0.2,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained MLPMixer models aren't available. Loading un-trained model."
        )

    return MLPMixer(32, 8, 512, 2048, 256, dropout, attach_head, num_classes, **kwargs)


@register_model
def mlpmixer_s16(
    attach_head=False,
    num_classes=1000,
    dropout=0.2,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained MLPMixer models aren't available. Loading un-trained model."
        )

    return MLPMixer(16, 8, 512, 2048, 256, dropout, attach_head, num_classes, **kwargs)


@register_model
def mlpmixer_b32(
    attach_head=False,
    num_classes=1000,
    dropout=0.2,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained MLPMixer models aren't available. Loading un-trained model."
        )

    return MLPMixer(32, 12, 768, 3072, 384, dropout, attach_head, num_classes, **kwargs)


@register_model
def mlpmixer_l32(
    attach_head=False,
    num_classes=1000,
    dropout=0.2,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained MLPMixer models aren't available. Loading un-trained model."
        )

    return MLPMixer(
        32, 24, 1024, 4096, 512, dropout, attach_head, num_classes, **kwargs
    )


@register_model
def mlpmixer_l16(
    attach_head=False,
    num_classes=1000,
    dropout=0.2,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained MLPMixer models aren't available. Loading un-trained model."
        )

    return MLPMixer(
        16, 24, 1024, 4096, 512, dropout, attach_head, num_classes, **kwargs
    )


@register_model
def mlpmixer_h14(
    attach_head=False,
    num_classes=1000,
    dropout=0.2,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained MLPMixer models aren't available. Loading un-trained model."
        )

    return MLPMixer(
        14, 32, 1280, 5120, 640, dropout, attach_head, num_classes, **kwargs
    )
