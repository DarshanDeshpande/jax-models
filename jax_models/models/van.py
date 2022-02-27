import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze

from ..layers import DropPath, DepthwiseConv2D
from ..initializers import trunc_norm_init
from .helper import download_checkpoint_params, load_trained_params
from .model_registry import register_model

from typing import Optional, Iterable, Callable

__all__ = [
    "VAN",
    "van_tiny",
    "van_small",
    "van_base",
    "van_large",
]

pretrained_cfgs = {
    "van-tiny": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.5-van/van_tiny.weights",
    "van-small": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.5-van/van_small.weights",
    "van-base": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.5-van/van_base.weights",
    "van-large": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.5-van/van_large.weights",
}

conv_init = nn.initializers.variance_scaling(
    scale=2.0, mode="fan_out", distribution="normal"
)
default_init = nn.initializers.variance_scaling(
    scale=1.0, mode="fan_in", distribution="normal"
)


class OverlapPatchEmbed(nn.Module):
    emb_dim: int = 768
    patch_size: int = 16
    stride: int = 4
    kernel_init: Callable = nn.initializers.xavier_normal()
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        conv = nn.Conv(
            self.emb_dim,
            (self.patch_size, self.patch_size),
            self.stride,
            padding=(
                (self.patch_size // 2, self.patch_size // 2),
                (self.patch_size // 2, self.patch_size // 2),
            ),
            kernel_init=self.kernel_init,
            name="proj",
        )(inputs)
        norm = nn.BatchNorm(
            momentum=0.9, use_running_average=deterministic, name="norm"
        )(conv)
        return norm


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.2
    use_dwconv: bool = False
    conv_kernel_init: Callable = nn.initializers.xavier_normal()
    linear: bool = False
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.Conv(self.dim, (1, 1), kernel_init=self.conv_kernel_init, name="fc1")(
            inputs
        )

        x = DepthwiseConv2D((3, 3), name="dwconv", weights_init=self.conv_kernel_init)(
            x
        )

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Conv(
            self.out_dim, (1, 1), kernel_init=self.conv_kernel_init, name="fc2"
        )(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x


class Attention(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(
            self.dim,
            (5, 5),
            1,
            padding=[[2, 2], [2, 2]],
            feature_group_count=self.dim,
            kernel_init=default_init,
            name="conv0",
        )(inputs)
        x = nn.Conv(
            self.dim,
            (7, 7),
            1,
            padding=[[9, 9], [9, 9]],
            feature_group_count=self.dim,
            kernel_dilation=3,
            kernel_init=default_init,
            name="conv_spatial",
        )(x)
        x = nn.Conv(self.dim, (1, 1), kernel_init=default_init, name="conv1")(x)
        return inputs * x


class SpatialAttention(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(self.d_model, (1, 1), kernel_init=default_init, name="proj_1")(
            inputs
        )
        x = nn.gelu(x)
        x = Attention(self.d_model, name="spatial_gating_unit")(x)
        x = nn.Conv(self.d_model, (1, 1), kernel_init=default_init, name="proj_2")(x)
        return x + inputs


class Block(nn.Module):
    dim: int
    mlp_hidden_dim: int
    droppath: float = 0.0
    dropout: float = 0.0
    init_value: float = 1e-2
    deterministic: Optional[bool] = None

    def scale_init(self, key, shape, value):
        return jnp.full(shape, value)

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        layer_scale_1 = self.param(
            "layer_scale_1", self.scale_init, (self.dim,), self.init_value
        )
        layer_scale_2 = self.param(
            "layer_scale_2", self.scale_init, (self.dim,), self.init_value
        )

        norm1 = nn.BatchNorm(
            momentum=0.9, epsilon=1e-5, name="norm1", use_running_average=deterministic
        )(inputs)
        attn = SpatialAttention(self.dim, name="attn")(norm1)
        scaled = jnp.expand_dims(jnp.expand_dims(layer_scale_1, 0), 0)
        scaled = scaled * attn
        drop_path = DropPath(self.droppath)(scaled, deterministic)
        inputs = inputs + drop_path

        norm2 = nn.BatchNorm(
            momentum=0.9, epsilon=1e-5, name="norm2", use_running_average=deterministic
        )(inputs)
        mlp = TransformerMLP(
            self.mlp_hidden_dim,
            self.dim,
            self.dropout,
            use_dwconv=True,
            conv_kernel_init=conv_init,
            name="mlp",
        )(norm2, deterministic)

        scaled = jnp.expand_dims(jnp.expand_dims(layer_scale_2, 0), 0) * mlp
        drop_path = DropPath(self.droppath)(scaled, deterministic)
        out = inputs + drop_path
        return out


class VAN(nn.Module):
    embed_dims: Iterable = (64, 128, 256, 512)
    mlp_ratios: Iterable = (4, 4, 4, 4)
    dropout: float = 0.0
    drop_path: float = 0.0
    depths: Iterable = (3, 4, 6, 3)
    num_stages: int = 4
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        dpr = [x.item() for x in jnp.linspace(0, self.drop_path, sum(self.depths))]
        cur = 0

        x = inputs

        for i in range(self.num_stages):
            x = OverlapPatchEmbed(
                self.embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                kernel_init=conv_init,
                name=f"patch_embed{i+1}",
            )(x, deterministic)

            batch, height, width, channels = x.shape

            for j in range(self.depths[i]):
                x = Block(
                    self.embed_dims[i],
                    self.mlp_ratios[i] * self.embed_dims[i],
                    dpr[cur + j],
                    self.dropout,
                    name=f"block{i+1}{j}",
                )(x, deterministic)

            cur += self.depths[i]

            x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = nn.LayerNorm(name=f"norm{i+1}")(x)

            if i != self.num_stages - 1:
                x = x.reshape(batch, height, width, -1)

        x = jnp.mean(x, axis=1)

        if self.attach_head:
            x = nn.Dense(self.num_classes, kernel_init=trunc_norm_init, name="head")(x)

        return x


@register_model
def van_tiny(
    attach_head=True, num_classes=1000, pretrained=False, download_dir=None, **kwargs
):
    model = VAN(
        embed_dims=(32, 64, 160, 256),
        mlp_ratios=(8, 8, 4, 4),
        depths=(3, 3, 5, 2),
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["van-tiny"], download_dir
        )
        variables = load_trained_params(file_path)
        params, batch_stats = variables["params"], variables["batch_stats"]
        return model, freeze(params), freeze(batch_stats)

    else:
        return model


@register_model
def van_small(
    attach_head=True, num_classes=1000, pretrained=False, download_dir=None, **kwargs
):
    model = VAN(
        embed_dims=(64, 128, 320, 512),
        mlp_ratios=(8, 8, 4, 4),
        depths=(2, 2, 4, 2),
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["van-small"], download_dir
        )
        variables = load_trained_params(file_path)
        params, batch_stats = variables["params"], variables["batch_stats"]
        return model, freeze(params), freeze(batch_stats)

    else:
        return model


@register_model
def van_base(
    attach_head=True, num_classes=1000, pretrained=False, download_dir=None, **kwargs
):
    model = VAN(
        embed_dims=(64, 128, 320, 512),
        mlp_ratios=(8, 8, 4, 4),
        depths=(3, 3, 12, 3),
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["van-base"], download_dir
        )
        variables = load_trained_params(file_path)
        params, batch_stats = variables["params"], variables["batch_stats"]
        return model, freeze(params), freeze(batch_stats)

    else:
        return model


@register_model
def van_large(
    attach_head=True, num_classes=1000, pretrained=False, download_dir=None, **kwargs
):
    model = VAN(
        embed_dims=(64, 128, 320, 512),
        mlp_ratios=(8, 8, 4, 4),
        depths=(3, 5, 27, 3),
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["van-large"], download_dir
        )
        variables = load_trained_params(file_path)
        params, batch_stats = variables["params"], variables["batch_stats"]
        return model, freeze(params), freeze(batch_stats)

    else:
        return model
