import jax.numpy as jnp

import flax.linen as nn
from layers import TransformerMLP
from typing import Optional


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


class Model(nn.Module):
    patch_size: int = 32
    num_mixers_layers: int = 2
    hidden_size: int = 768
    channels_dim: int = 256
    tokens_dim: int = 256
    dropout: float = 0.2
    attach_head: Optional[bool] = False
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


def MLPMixer_S32(attach_head=False, num_classes=1000, dropout=0.2):
    return Model(32, 8, 512, 2048, 256, dropout, attach_head, num_classes)


def MLPMixer_S16(attach_head=False, num_classes=1000, dropout=0.2):
    return Model(16, 8, 512, 2048, 256, dropout, attach_head, num_classes)


def MLPMixer_B32(attach_head=False, num_classes=1000, dropout=0.2):
    return Model(32, 12, 768, 3072, 384, dropout, attach_head, num_classes)


def MLPMixer_L32(attach_head=False, num_classes=1000, dropout=0.2):
    return Model(32, 24, 1024, 4096, 512, dropout, attach_head, num_classes)


def MLPMixer_L16(attach_head=False, num_classes=1000, dropout=0.2):
    return Model(16, 24, 1024, 4096, 512, dropout, attach_head, num_classes)


def MLPMixer_H14(attach_head=False, num_classes=1000, dropout=0.2):
    return Model(14, 32, 1280, 5120, 640, dropout, attach_head, num_classes)
