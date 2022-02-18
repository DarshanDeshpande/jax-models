import jax.numpy as jnp
import flax.linen as nn

from typing import Optional, Iterable

from ..layers import OverlapPatchEmbed, TransformerMLP, AdaptiveAveragePool2D, DropPath
from ..models.helper import download_checkpoint_params, load_trained_params
from .model_registry import register_model

__all__ = [
    "PyramidViT",
    "pvit_b0",
    "pvit_b1",
    "pvit_b2",
    "pvit_b2_linear",
    "pvit_b3",
    "pvit_b4",
    "pvit_b5",
]

pretrained_cfgs = {
    "pvit_b0": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.3-pvit/pvit_b0.weights",
    "pvit_b1": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.3-pvit/pvit_b1.weights",
    "pvit_b2": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.3-pvit/pvit_b2.weights",
    "pvit_b2_linear": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.3-pvit/pvit_b2_linear.weights",
    "pvit_b3": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.3-pvit/pvit_b3.weights",
    "pvit_b4": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.3-pvit/pvit_b4.weights",
    "pvit_b5": "https://github.com/DarshanDeshpande/jax-models/releases/download/v0.3-pvit/pvit_b5.weights",
}


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    use_att_bias: bool = False
    att_drop: float = 0.0
    proj_drop: float = 0.0
    sr_ratio: int = 1
    linear: bool = False
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, height, width, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        assert self.dim % self.num_heads == 0
        head_dim = self.dim // self.num_heads

        batch, n, channels = inputs.shape
        q = nn.Dense(self.dim, use_bias=self.use_att_bias, name="q")(inputs).reshape(
            batch, n, self.num_heads, channels // self.num_heads
        )
        q = jnp.transpose(q, (0, 2, 1, 3))

        if not self.linear:
            if self.sr_ratio > 1:
                x = inputs.reshape(batch, height, width, channels)
                x = nn.Conv(
                    self.dim, (self.sr_ratio, self.sr_ratio), self.sr_ratio, name="sr"
                )(x)
                x = nn.LayerNorm(name="norm")(x)
                kv = nn.Dense(self.dim * 2, use_bias=self.use_att_bias, name="kv")(x)
                kv = kv.reshape(
                    batch, -1, 2, self.num_heads, channels // self.num_heads
                )
                kv = jnp.transpose(kv, (2, 0, 3, 1, 4))
            else:
                kv = nn.Dense(self.dim * 2, use_bias=self.use_att_bias, name="kv")(
                    inputs
                )
                kv = kv.reshape(
                    batch, -1, 2, self.num_heads, channels // self.num_heads
                )
                kv = jnp.transpose(kv, (2, 0, 3, 1, 4))
        else:
            x = inputs.reshape(batch, height, width, channels)
            x = AdaptiveAveragePool2D(7, name="pool")(x)
            x = nn.Conv(self.dim, (1, 1), 1, name="sr")(x)
            x = nn.LayerNorm(name="norm")(x)
            x = nn.gelu(x)
            kv = nn.Dense(self.dim * 2, use_bias=self.use_att_bias, name="kv")(x)
            kv = kv.reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
            kv = jnp.transpose(kv, (2, 0, 3, 1, 4))

        k, v = kv[0], kv[1]

        att = (q @ jnp.swapaxes(k, -2, -1)) * (head_dim ** -0.5)
        att = nn.softmax(att, -1)
        att = nn.Dropout(self.att_drop)(att, deterministic)

        x = jnp.swapaxes(att @ v, 1, 2).reshape(batch, n, channels)
        x = nn.Dense(self.dim, name="proj")(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic)

        return x


class Block(nn.Module):
    dim: int
    num_heads: int = 8
    use_att_bias: bool = False
    att_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path: float = 0.0
    sr_ratio: int = 1
    mlp_ratio: int = 4
    linear: bool = False
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, height, width, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        hidden_dim = self.dim * self.mlp_ratio

        x = nn.LayerNorm(name="norm1")(inputs)
        x = Attention(
            self.dim,
            self.num_heads,
            self.use_att_bias,
            self.att_drop,
            self.proj_drop,
            self.sr_ratio,
            self.linear,
            name="attn",
        )(x, height, width, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        inputs = inputs + x

        x = nn.LayerNorm(name="norm2")(inputs)
        x = TransformerMLP(
            hidden_dim, self.dim, self.proj_drop, True, linear=self.linear, name="mlp"
        )(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        x = inputs + x

        return x


class PyramidViT(nn.Module):
    """
    Pyramid Vision Transformer Module

    Attributes:
    patch_size (int): Patch size. Default is 4.
    emb_dims (list or tuple): Embedding dimensions at every stage.
    num_heads (list or tuple): Number of attention heads at every stage.
    mlp_ratios (list or tuple): Multiplier for hidden dimension in transformer MLP block. Default is 4 for every block.
    use_att_bias (bool): Whether to use bias for linear qkv projection. Default is True.
    drop (float): Dropout value to use for projection layers. Default is 0.
    att_drop (float): Dropout value to use for attention. Default is 0.
    drop_path (float): Dropout value to use for DropPath. Default is 0.
    depths (list or tuple): Depths for every block.
    sr_ratios (list or tuple): sr ratio for every stage.
    num_stages (int): Number of stages in the model. Default is 4.
    linear (bool): Whether to use the linear approach to the build the model. Default is False.
    attach_head (bool): Whether to attach classification head. Default is True
    num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
    deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.
    """

    patch_size: int = 4
    embed_dims: Iterable[int] = (64, 128, 256, 512)
    num_heads: Iterable[int] = (1, 2, 4, 8)
    mlp_ratios: Iterable[int] = (4, 4, 4, 4)
    use_att_bias: bool = True
    drop: float = 0.0
    att_drop: float = 0.0
    drop_path: float = 0.0
    depths: Iterable[int] = (3, 4, 6, 3)
    sr_ratios: Iterable[int] = (8, 4, 2, 1)
    num_stages: int = 4
    linear: bool = False
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        dpr = [x.item() for x in jnp.linspace(0, self.drop_path, sum(self.depths))]
        cur = 0

        for i in range(self.num_stages):
            x = OverlapPatchEmbed(
                self.embed_dims[i],
                7 if i == 0 else 3,
                4 if i == 0 else 2,
                name=f"patch_embed{i+1}",
            )(x)

            batch, n, channels = x.shape
            height = width = int(n ** 0.5)

            for j in range(self.depths[i]):
                x = Block(
                    self.embed_dims[i],
                    self.num_heads[i],
                    self.use_att_bias,
                    self.att_drop,
                    self.drop,
                    dpr[cur + j],
                    self.sr_ratios[i],
                    self.mlp_ratios[i],
                    linear=self.linear,
                    name=f"block{i+1}{j}",
                )(x, height, width, deterministic)

            x = nn.LayerNorm(name=f"norm{i+1}")(x)
            if i != self.num_stages - 1:
                x = x.reshape(batch, height, width, -1)
            cur += self.depths[i]

        x = jnp.mean(x, axis=1)

        if self.attach_head:
            x = nn.Dense(self.num_classes, name="head")(x)

        return x


@register_model
def pvit_b0(
    attach_head=True, num_classes=1000, dropout=0.0, pretrained=False, download_dir=None
):
    model = PyramidViT(
        patch_size=4,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        use_att_bias=True,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
    )

    if pretrained:
        file_path = download_checkpoint_params(pretrained_cfgs["pvit_b0"], download_dir)
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def pvit_b1(
    attach_head=True, num_classes=1000, dropout=0.0, pretrained=False, download_dir=None
):
    model = PyramidViT(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        use_att_bias=True,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
    )

    if pretrained:
        file_path = download_checkpoint_params(pretrained_cfgs["pvit_b1"], download_dir)
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def pvit_b2(
    attach_head=True, num_classes=1000, dropout=0.0, pretrained=False, download_dir=None
):
    model = PyramidViT(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        use_att_bias=True,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
    )

    if pretrained:
        file_path = download_checkpoint_params(pretrained_cfgs["pvit_b2"], download_dir)
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def pvit_b3(
    attach_head=True, num_classes=1000, dropout=0.0, pretrained=False, download_dir=None
):
    model = PyramidViT(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        use_att_bias=True,
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
    )

    if pretrained:
        file_path = download_checkpoint_params(pretrained_cfgs["pvit_b3"], download_dir)
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def pvit_b4(
    attach_head=True, num_classes=1000, dropout=0.0, pretrained=False, download_dir=None
):
    model = PyramidViT(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        use_att_bias=True,
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
    )

    if pretrained:
        file_path = download_checkpoint_params(pretrained_cfgs["pvit_b4"], download_dir)
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def pvit_b5(
    attach_head=True, num_classes=1000, dropout=0.0, pretrained=False, download_dir=None
):
    model = PyramidViT(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        use_att_bias=True,
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
    )

    if pretrained:
        file_path = download_checkpoint_params(pretrained_cfgs["pvit_b5"], download_dir)
        params = load_trained_params(file_path)
        return model, params

    else:
        return model


@register_model
def pvit_b2_linear(
    attach_head=True, num_classes=1000, dropout=0.0, pretrained=False, download_dir=None
):
    model = PyramidViT(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        use_att_bias=True,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        linear=True,
        attach_head=attach_head,
        num_classes=num_classes,
        drop=dropout,
    )

    if pretrained:
        file_path = download_checkpoint_params(
            pretrained_cfgs["pvit_b2_linear"], download_dir
        )
        params = load_trained_params(file_path)
        return model, params

    else:
        return model
