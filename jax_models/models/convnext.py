import jax.random as random
import jax.numpy as jnp
import flax.linen as nn

from ..layers import DepthwiseConv2D, DropPath
from .helper import load_trained_params, download_checkpoint_params
from .model_registry import register_model

from typing import Optional, Iterable
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

__all__ = [
    "ConvNeXt",
    "convnext_tiny",
    "convnext_small",
    "convnext_base_224_1K",
    "convnext_base_224_22K",
    "convnext_base_224_22K_1K",
    "convnext_base_384_1K",
    "convnext_base_384_22K_1K",
    "convnext_large_224_1K",
    "convnext_large_224_22K",
    "convnext_large_224_22K_1K",
    "convnext_large_384_1K",
    "convnext_large_384_22K_1K",
    "convnext_xlarge_224_22K",
    "convnext_xlarge_224_22K_1K",
    "convnext_xlarge_384_22K_1K",
]

pretrained_cfgs = {
    "convnext-tiny-224-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_tiny_224_1k.weights",
    "convnext-small-224-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_small_224_1k.weights",
    "convnext-base-224-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_base_224_1k.weights",
    "convnext-base-224-22k-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_base_224_22k_1k.weights",
    "convnext-base-224-22k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_base_224_22k.weights",
    "convnext-base-384-22k-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_base_384_22k_1k.weights",
    "convnext-base-384-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_base_384_1k.weights",
    "convnext-large-224-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_large_224_1k.weights",
    "convnext-large-224-22k-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_large_224_22k_1k.weights",
    "convnext-large-224-22k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_large_224_22k.weights",
    "convnext-large-384-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_large_384_1k.weights",
    "convnext-large-384-22k-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_large_384_22k_1k.weights",
    "convnext-xlarge-224-22k-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_xlarge_224_22k_1k.weights",
    "convnext-xlarge-224-22k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_xlarge_224_22k.weights",
    "convnext-xlarge-384-22k-1k": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.2-convnext/convnext_xlarge_384_22k_1k.weights",
}

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
        x = DepthwiseConv2D((7, 7), weights_init=initializer, name="dwconv")(inputs)
        x = nn.LayerNorm(name="norm")(x)
        x = nn.Dense(4 * self.dim, kernel_init=initializer, name="pwconv1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, kernel_init=initializer, name="pwconv2")(x)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma", self.init_fn, (self.dim,), self.layer_scale_init_value
            )
            x = gamma * x

        x = inputs + DropPath(self.drop_path)(x, deterministic)
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt Module

    Attributes:

        depths (list or tuple): Depths for every block
        dims (list or tuple): Embedding dimension for every stage.
        drop_path (float): Dropout value for DropPath. Default is 0.1
        layer_scale_init_value (float): Initialization value for scale. Default is 1e-6.
        head_init_scale (float): Initialization value for head. Default is 1.0.
        attach_head (bool): Whether to attach classification head. Default is False.
        num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    depths: Iterable = (3, 3, 9, 3)
    dims: Iterable = (96, 192, 384, 768)
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        dp_rates = jnp.linspace(0, self.drop_path, sum(self.depths))
        curr = 0

        # Stem
        x = nn.Conv(
            self.dims[0], (4, 4), 4, kernel_init=initializer, name="downsample_layers00"
        )(inputs)
        x = nn.LayerNorm(name="downsample_layers01")(x)

        for j in range(self.depths[0]):
            x = ConvNeXtBlock(
                self.dims[0],
                drop_path=dp_rates[curr + j],
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"stages0{j}",
            )(x, deterministic)
        curr += self.depths[0]

        # Downsample layers
        for i in range(3):
            x = nn.LayerNorm(name=f"downsample_layers{i+1}0")(x)
            x = nn.Conv(
                self.dims[i + 1],
                (2, 2),
                2,
                kernel_init=initializer,
                name=f"downsample_layers{i+1}1",
            )(x)

            for j in range(self.depths[i + 1]):
                x = ConvNeXtBlock(
                    self.dims[i + 1],
                    drop_path=dp_rates[curr + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    name=f"stages{i+1}{j}",
                )(x, deterministic)

            curr += self.depths[i + 1]

        if self.attach_head:
            x = nn.LayerNorm(name="norm")(jnp.mean(x, [1, 2]))
            x = nn.Dense(self.num_classes, kernel_init=initializer, name="head")(x)
        return x


@register_model
def convnext_tiny(
    attach_head=False,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = ConvNeXt(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-tiny-224-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_small(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):

    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-small-224-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_base_224_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-base-224-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_base_224_22K_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-base-224-22k-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_base_224_22K(
    attach_head=True,
    num_classes=21841,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-base-224-22k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_base_384_22K_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-base-384-22k-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_base_384_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-base-384-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_large_224_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-large-224-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_large_224_22K_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-large-224-22k-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_large_224_22K(
    attach_head=True,
    num_classes=21841,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-large-224-22k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_large_384_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-large-384-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_large_384_22K_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-large-384-22k-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_xlarge_224_22K_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(256, 512, 1024, 2048),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-xlarge-224-22k-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_xlarge_224_22K(
    attach_head=True,
    num_classes=21841,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(256, 512, 1024, 2048),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-xlarge-224-22k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_xlarge_224_22K_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(256, 512, 1024, 2048),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-xlarge-224-22k-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def convnext_xlarge_384_22K_1K(
    attach_head=True,
    num_classes=1000,
    dropout=0.0,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    model = ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(256, 512, 1024, 2048),
        drop_path=dropout,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["convnext-xlarge-384-22k-1k"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model
