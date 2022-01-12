import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Iterable, Sequence

from ..layers import DepthwiseConv2D, DropPath

__all__ = [
    "ConvNeXt",
    "ConvNeXt_Tiny",
    "ConvNeXt_Small",
    "ConvNeXt_Base",
    "ConvNeXt_Large",
    "ConvNeXt_XLarge",
]

initializer = nn.initializers.variance_scaling(
    0.2, "fan_in", distribution="truncated_normal"
)


class ConvNeXtBlock(nn.Module):
    dim: int = 256
    layer_scale_init_value: float = 1e-6
    drop_path: float = 0.1
    deterministic: Optional[bool] = None

    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = DepthwiseConv2D((7, 7), weights_init=initializer)(inputs)
        x = nn.LayerNorm()(x)
        x = nn.Dense(4 * self.dim, kernel_init=initializer)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, kernel_init=initializer)(x)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma", self.init_fn, (self.dim,), self.layer_scale_init_value
            )
            x = gamma * x

        x = inputs + DropPath(self.drop_path)(x, deterministic)
        return x


class ConvNeXt(nn.Module):
    depths: Iterable = (3, 3, 9, 3)
    dims: Iterable = (96, 192, 384, 768)
    drop_path: float = 0.1
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attach_head: bool = False
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        dp_rates = [x.item() for x in jnp.linspace(0, self.drop_path, sum(self.depths))]
        curr = 0

        # Stem
        x = nn.Conv(self.dims[0], (4, 4), 4, kernel_init=initializer)(inputs)
        x = nn.LayerNorm()(x)
        for j in range(self.depths[0]):
            x = ConvNeXtBlock(
                self.dims[0],
                drop_path=dp_rates[curr + j],
                layer_scale_init_value=self.layer_scale_init_value,
            )(x, deterministic)
        curr += self.depths[0]

        # Downsample layers
        for i in range(3):
            x = nn.LayerNorm()(x)
            x = nn.Conv(self.dims[i + 1], (2, 2), 2, kernel_init=initializer)(x)

            for j in range(self.depths[i + 1]):
                x = ConvNeXtBlock(
                    self.dims[i + 1],
                    drop_path=dp_rates[curr + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                )(x, deterministic)

            curr += self.depths[i + 1]

        if self.attach_head:
            x = nn.LayerNorm()(x)
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes, kernel_init=initializer)(x)
        return x


def ConvNeXt_Tiny(attach_head=False, num_classes=1000, dropout=0.1, **kwargs):
    return ConvNeXt(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )


def ConvNeXt_Small(attach_head=False, num_classes=1000, dropout=0.1, **kwargs):
    return ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )


def ConvNeXt_Base(attach_head=False, num_classes=1000, dropout=0.1, **kwargs):
    return ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )


def ConvNeXt_Large(attach_head=False, num_classes=1000, dropout=0.1, **kwargs):
    return ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )


def ConvNeXt_XLarge(attach_head=False, num_classes=1000, dropout=0.1, **kwargs):
    return ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(256, 512, 1024, 2048),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )
