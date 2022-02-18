import jax.numpy as jnp
from jax.image import resize
import flax.linen as nn

from ..layers import DropPath, TransformerMLP, OverlapPatchEmbed
from .model_registry import register_model

from typing import Optional, Iterable, Any
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

__all__ = [
    "Segformer",
    "segformer_b0",
    "segformer_b1",
    "segformer_b2",
    "segformer_b3",
    "segformer_b4",
    "segformer_b5",
]


class EfficientSelfAttention(nn.Module):
    dim: int = 256
    num_heads: int = 8
    sr_ratio: int = 1
    att_drop: float = 0.1
    proj_drop: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        assert self.dim % self.num_heads == 0

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5

        batch, n, channels = inputs.shape
        height = width = int(n ** 0.5)

        q = nn.Dense(self.dim, use_bias=True)(inputs)
        q = jnp.reshape(q, (batch, n, self.num_heads, channels // self.num_heads))
        q = jnp.transpose(q, (0, 2, 1, 3))

        if self.sr_ratio > 1:
            inputs = jnp.transpose(inputs, (0, 2, 1)).reshape(
                batch, height, width, channels
            )
            conv = nn.Conv(self.dim, (self.sr_ratio, self.sr_ratio), self.sr_ratio)(
                inputs
            )
            conv = jnp.reshape(conv, (batch, -1, channels))
            norm = nn.LayerNorm()(conv)
            kv = nn.Dense(self.dim * 2, use_bias=True)(norm).reshape(
                batch, -1, 2, self.num_heads, channels // self.num_heads
            )
            kv = jnp.transpose(kv, (2, 0, 3, 1, 4))

        else:
            kv = nn.Dense(self.dim * 2, use_bias=True)(inputs)
            kv = jnp.reshape(
                kv, (batch, -1, 2, self.num_heads, channels // self.num_heads)
            )
            kv = jnp.transpose(kv, (2, 0, 3, 1, 4))

        k, v = kv[0], kv[1]

        att = q @ jnp.swapaxes(k, -2, -1) * scale
        att = nn.softmax(att, -1)
        att = nn.Dropout(self.att_drop)(att, deterministic)

        x = jnp.swapaxes(att @ v, 1, 2).reshape(batch, n, channels)
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic)

        return x


class Block(nn.Module):
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    sr_ratio: int = 1
    att_drop: float = 0.1
    drop: float = 0.1
    drop_path: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.LayerNorm()(inputs)
        x = EfficientSelfAttention(
            self.dim, self.num_heads, self.sr_ratio, self.att_drop, self.drop
        )(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        inputs = inputs + x

        x = nn.LayerNorm()(inputs)
        x = TransformerMLP(
            self.dim * self.mlp_ratio, self.dim, self.drop, use_dwconv=True
        )(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        return inputs + x


class MixTransformer(nn.Module):
    patch_size: int = 16
    emb_dims: Iterable = (64, 128, 256, 512)
    num_heads: Iterable = (1, 2, 4, 8)
    mlp_ratios = (4, 4, 4, 4)
    drop: float = 0.1
    att_drop: float = 0.1
    drop_path: float = 0.1
    depths: Iterable = (3, 4, 6, 3)
    sr_ratios: Iterable = (8, 4, 2, 1)
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        batch = inputs.shape[0]
        outs = []

        dpr = [x.item() for x in jnp.linspace(0, self.drop_path, sum(self.depths))]
        cur = 0

        x = OverlapPatchEmbed(self.emb_dims[0], 7, 4)(inputs)
        for i in range(self.depths[0]):
            x = Block(
                self.emb_dims[0],
                self.num_heads[0],
                self.mlp_ratios[0],
                self.sr_ratios[0],
                self.att_drop,
                self.drop,
                dpr[cur + i],
            )(x, deterministic)
        x = nn.LayerNorm()(x)
        height = width = int(x.shape[1] ** 0.5)
        x = jnp.reshape(x, (batch, height, width, -1))
        outs.append(x)

        cur += self.depths[0]
        x = OverlapPatchEmbed(self.emb_dims[1], 3, 2)(x)
        for i in range(self.depths[1]):
            x = Block(
                self.emb_dims[1],
                self.num_heads[1],
                self.mlp_ratios[1],
                self.sr_ratios[1],
                self.att_drop,
                self.drop,
                dpr[cur + i],
            )(x, deterministic)
        x = nn.LayerNorm()(x)
        height = width = int(x.shape[1] ** 0.5)
        x = jnp.reshape(x, (batch, height, width, -1))
        outs.append(x)

        cur += self.depths[1]
        x = OverlapPatchEmbed(self.emb_dims[2], 3, 2)(x)
        for i in range(self.depths[2]):
            x = Block(
                self.emb_dims[2],
                self.num_heads[2],
                self.mlp_ratios[2],
                self.sr_ratios[2],
                self.att_drop,
                self.drop,
                dpr[cur + i],
            )(x, deterministic)
        x = nn.LayerNorm()(x)
        height = width = int(x.shape[1] ** 0.5)
        x = jnp.reshape(x, (batch, height, width, -1))
        outs.append(x)

        cur += self.depths[2]
        x = OverlapPatchEmbed(self.emb_dims[3], 3, 2)(x)
        for i in range(self.depths[3]):
            x = Block(
                self.emb_dims[3],
                self.num_heads[3],
                self.mlp_ratios[3],
                self.sr_ratios[3],
                self.att_drop,
                self.drop,
                dpr[cur + i],
            )(x, deterministic)
        x = nn.LayerNorm()(x)
        height = width = int(x.shape[1] ** 0.5)
        x = jnp.reshape(x, (batch, height, width, -1))
        outs.append(x)

        return outs


class MLPHead(nn.Module):
    emb_dim: int = 768

    @nn.compact
    def __call__(self, inputs):
        batch, height, width, channels = inputs.shape
        x = jnp.reshape(inputs, (batch, height * width, channels))
        x = nn.Dense(self.emb_dim)(x)
        return x


class SegFormerHead(nn.Module):
    emb_dim: int = 768
    kernel_size: Iterable[int] = (1, 1)
    stride: int = 1
    num_classes: int = 3
    drop: float = 0.1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        c1, c2, c3, c4 = inputs
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        h1 = MLPHead(self.emb_dim)(c1)
        h1 = jnp.reshape(h1, (c1.shape[0], c1.shape[1], c1.shape[2], -1))

        h2 = MLPHead(self.emb_dim)(c2)
        h2 = jnp.reshape(h2, (c2.shape[0], c2.shape[1], c2.shape[2], -1))
        h2 = resize(
            h2, (c1.shape[0], c1.shape[1], c1.shape[2], h2.shape[-1]), "bilinear"
        )

        h3 = MLPHead(self.emb_dim)(c3)
        h3 = jnp.reshape(h3, (c3.shape[0], c3.shape[1], c3.shape[2], -1))
        h3 = resize(
            h3, (c1.shape[0], c1.shape[1], c1.shape[2], h3.shape[-1]), "bilinear"
        )

        h4 = MLPHead(self.emb_dim)(c4)
        h4 = jnp.reshape(h4, (c4.shape[0], c4.shape[1], c4.shape[2], -1))
        h4 = resize(
            h4, (c1.shape[0], c1.shape[1], c1.shape[2], h4.shape[-1]), "bilinear"
        )

        concat = jnp.concatenate([h1, h2, h3, h4], axis=-1)

        conv = nn.Conv(self.emb_dim, self.kernel_size, self.stride)(concat)
        norm = nn.BatchNorm(
            deterministic,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
        )(conv)
        act = nn.relu(norm)

        dropout = nn.Dropout(self.drop)(act, deterministic)
        linear_pred = nn.Conv(self.num_classes, (1, 1))(dropout)

        return linear_pred


class SegFormer(nn.Module):
    """
    SegFormer Module

    Attributes:
        patch_size (int): Patch size. Default is 4.
        emb_dims (list or tuple): Embedding dimension for every block.
        num_heads (list or tuple): Number of attention heads for every stage.
        mlp_ratios (int): Multiplier for hidden dimension in transformer MLP block at every stage. Default is 4 at every stage.
        drop (float): Dropout value. Default is 0.
        att_dropout (float): Dropout value for attention Default is 0.
        drop_path (float): Dropout value for DropPath. Default is 0.1.
        depths (list or tuple): Depths for every block.
        sr_ratios (list or tuple): sr ratio for every block.
        num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 19.
        decoder_emb (int): Embedding dimension for decoder. Default is 256.
        axis_name (str): Optional str, Used when parallelizing training over multiple devices. Default is None.
        axis_index_groups: Index groups used when parallelizing training over multiple devices. Default is None.
        attach_head (bool): Whether to attach classification head. Default is False
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    patch_size: int = 4
    emb_dims: Iterable = (64, 128, 256, 512)
    num_heads: Iterable = (1, 2, 4, 8)
    mlp_ratios = (4, 4, 4, 4)
    drop: float = 0.0
    att_drop: float = 0.0
    drop_path: float = 0.1
    depths: Iterable = (3, 4, 6, 3)
    sr_ratios: Iterable = (8, 4, 2, 1)
    num_classes: int = 19
    decoder_emb: int = 256
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    attach_head: bool = False
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        out_list = MixTransformer(
            self.patch_size,
            self.emb_dims,
            self.num_heads,
            self.drop,
            self.drop,
            self.drop_path,
            self.depths,
            self.sr_ratios,
        )(inputs, deterministic)
        if self.attach_head:
            return SegFormerHead(
                emb_dim=self.decoder_emb,
                drop=self.drop,
                num_classes=self.num_classes,
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups,
            )(out_list, deterministic)
        else:
            return out_list


@register_model
def segformer_b0(
    attach_head=False,
    num_classes=19,
    dropout=0.0,
    drop_path=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained SegFormer_B0 isn't available. Loading un-trained model instead"
        )

    return SegFormer(
        4,
        (32, 64, 160, 256),
        (1, 2, 5, 8),
        dropout,
        dropout,
        drop_path,
        (2, 2, 2, 2),
        (8, 4, 2, 1),
        attach_head=attach_head,
        num_classes=num_classes,
        decoder_emb=256,
        **kwargs,
    )


@register_model
def segformer_b1(
    attach_head=False,
    num_classes=19,
    dropout=0.0,
    drop_path=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained SegFormer_B1 isn't available. Loading un-trained model instead"
        )

    return SegFormer(
        4,
        (64, 128, 320, 512),
        (1, 2, 5, 8),
        dropout,
        dropout,
        drop_path,
        (2, 2, 2, 2),
        (8, 4, 2, 1),
        attach_head=attach_head,
        num_classes=num_classes,
        decoder_emb=256,
        **kwargs,
    )


@register_model
def segformer_b2(
    attach_head=False,
    num_classes=19,
    dropout=0.0,
    drop_path=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained SegFormer_B2 isn't available. Loading un-trained model instead"
        )

    return SegFormer(
        4,
        (64, 128, 320, 512),
        (1, 2, 5, 8),
        dropout,
        dropout,
        drop_path,
        (3, 4, 6, 3),
        (8, 4, 2, 1),
        attach_head=attach_head,
        num_classes=num_classes,
        decoder_emb=768,
        **kwargs,
    )


@register_model
def segformer_b3(
    attach_head=False,
    num_classes=19,
    dropout=0.0,
    drop_path=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained SegFormer_B3 isn't available. Loading un-trained model instead"
        )

    return SegFormer(
        4,
        (64, 128, 320, 512),
        (1, 2, 5, 8),
        dropout,
        dropout,
        drop_path,
        (3, 4, 18, 3),
        (8, 4, 2, 1),
        attach_head=attach_head,
        num_classes=num_classes,
        decoder_emb=768,
        **kwargs,
    )


@register_model
def segformer_b4(
    attach_head=False,
    num_classes=19,
    dropout=0.0,
    drop_path=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained SegFormer_B4 isn't available. Loading un-trained model instead"
        )

    return SegFormer(
        4,
        (64, 128, 320, 512),
        (1, 2, 5, 8),
        dropout,
        dropout,
        drop_path,
        (3, 8, 27, 3),
        (8, 4, 2, 1),
        attach_head=attach_head,
        num_classes=num_classes,
        decoder_emb=768,
        **kwargs,
    )


@register_model
def segformer_b5(
    attach_head=False,
    num_classes=19,
    dropout=0.0,
    drop_path=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained SegFormer_B5 isn't available. Loading un-trained model instead"
        )

    return SegFormer(
        4,
        (64, 128, 320, 512),
        (1, 2, 5, 8),
        dropout,
        dropout,
        drop_path,
        (3, 6, 40, 3),
        (8, 4, 2, 1),
        attach_head=attach_head,
        num_classes=num_classes,
        decoder_emb=768,
        **kwargs,
    )
