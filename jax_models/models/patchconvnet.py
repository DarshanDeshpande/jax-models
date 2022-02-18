import jax.numpy as jnp
import flax.linen as nn

from ..layers import SqueezeAndExcitation, DropPath, TransformerMLP
from .model_registry import register_model

from typing import Optional
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

__all__ = [
    "PatchConvNet",
    "patchconvnet_s60",
    "patchconvnet_s120",
    "patchconvnet_b60",
    "patchconvnet_b120",
    "patchconvnet_l60",
    "patchconvnet_l120",
]


class ConvolutionalStem(nn.Module):
    emb_dim: int = 768

    @nn.compact
    def __call__(self, inputs):
        conv = nn.Conv(
            self.emb_dim // 8, (3, 3), strides=2, padding="SAME", use_bias=False
        )(inputs)
        conv = nn.gelu(conv)
        conv = nn.Conv(
            self.emb_dim // 4, (3, 3), strides=2, padding="SAME", use_bias=False
        )(conv)
        conv = nn.gelu(conv)
        conv = nn.Conv(
            self.emb_dim // 2, (3, 3), strides=2, padding="SAME", use_bias=False
        )(conv)
        conv = nn.gelu(conv)
        conv = nn.Conv(self.emb_dim, (3, 3), strides=2, padding="SAME", use_bias=False)(
            conv
        )
        batch, height, width, channels = conv.shape
        return jnp.reshape(conv, (batch, height * width, channels))


class TrunkBlock(nn.Module):
    conv_dim: int

    @nn.compact
    def __call__(self, inputs):
        batch, proj_dim, channels = inputs.shape
        height = width = int(proj_dim ** 0.5)
        reshaped = jnp.reshape(inputs, (batch, height, width, channels))

        norm = nn.LayerNorm()(reshaped)
        conv1 = nn.Conv(self.conv_dim, (1, 1), strides=1)(norm)
        conv1 = nn.gelu(conv1)
        conv2 = nn.Conv(self.conv_dim, (3, 3), strides=1)(conv1)
        conv2 = nn.gelu(conv2)
        sae = SqueezeAndExcitation()(conv2)
        conv3 = nn.Conv(self.conv_dim, (1, 1), strides=1)(sae)
        conv3 = jnp.reshape(conv3, (batch, proj_dim, channels))

        gamma = self.param("gamma", nn.ones, (self.conv_dim,))
        scaled_output = gamma * conv3
        return inputs + scaled_output


class Aggregation(nn.Module):
    dim: int = 768
    dropout: float = 0.5
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        batch, proj_dim, channels = inputs.shape
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        q = nn.Dense(self.dim, use_bias=False)(inputs[:, 0])
        q = jnp.transpose(
            jnp.expand_dims(q, 1).reshape(batch, 1, 1, channels), (0, 2, 1, 3)
        )

        k = nn.Dense(self.dim, use_bias=False)(inputs)
        k = jnp.reshape(k, (batch, proj_dim, 1, channels))
        k = jnp.transpose(k, (0, 2, 1, 3))

        q = q * (self.dim ** -0.5)
        v = nn.Dense(self.dim, use_bias=False)(inputs)
        v = jnp.transpose(jnp.reshape(v, (batch, proj_dim, 1, channels)), (0, 2, 1, 3))

        att = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        att = nn.softmax(att, -1)
        att_drop = nn.Dropout(self.dropout)(att, deterministic=deterministic)

        x_cls = jnp.transpose(jnp.matmul(att_drop, v), (0, 2, 1, 3)).reshape(
            batch, 1, channels
        )
        x_cls = nn.Dense(self.dim)(x_cls)
        x_cls = nn.Dropout(self.dropout)(x_cls, deterministic=deterministic)

        return x_cls


class AttentionPoolingBlock(nn.Module):
    dim: int = 768
    dropout: float = 0.5
    mlp_ratio: int = 4
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        input, token = inputs
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        hidden_dim = int(self.dim * self.mlp_ratio)
        gamma_1 = self.param("gamma_1", nn.zeros, (self.dim,))
        gamma_2 = self.param("gamma_2", nn.zeros, (self.dim,))

        concat = jnp.concatenate((token, input), axis=1)

        norm = nn.LayerNorm()(concat)
        att = Aggregation(self.dim, self.dropout)(norm, deterministic=deterministic)
        scaled_att = gamma_1 * att
        drop_att = DropPath(self.dropout)(scaled_att, deterministic=deterministic)
        token = token + drop_att

        norm = nn.LayerNorm()(token)
        x = TransformerMLP(dim=hidden_dim, out_dim=self.dim, dropout=self.dropout)(
            norm, deterministic
        )
        scaled_mlp = gamma_2 * x
        drop_mlp = DropPath(self.dropout)(scaled_mlp, deterministic=deterministic)
        token = token + drop_mlp

        return token


class PatchConvNet(nn.Module):
    """
    PatchConvNet Module

    Attributes:
        depth (int): Depth for PatchConvNet. Default is 20.
        dim (int): Embedding dimension. Default is 768.
        dropout (float): Dropout value. Default is 0.5.
        mlp_ratio (int): Multiplier for hidden dimension in transformer MLP block. Default is 4.
        attach_head (bool): Whether to attach classification head. Default is True.
        num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    depth: int = 20
    dim: int = 768
    dropout: float = 0.5
    mlp_ratio: int = 4
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        stem = ConvolutionalStem(emb_dim=self.dim)(inputs)
        cls_token = self.param(
            "cls_token", nn.zeros, (inputs.shape[0], 1, int(self.dim))
        )

        trunk = stem
        for _ in range(self.depth):
            trunk = TrunkBlock(conv_dim=self.dim)(trunk)

        cls_token = AttentionPoolingBlock(
            dim=self.dim, dropout=self.dropout, mlp_ratio=self.mlp_ratio
        )([trunk, cls_token], deterministic=deterministic)

        x = jnp.concatenate((cls_token, trunk), axis=1)
        x = nn.LayerNorm()(x)
        x = x[:, 0]
        if self.attach_head:
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)
        return x


@register_model
def patchconvnet_s60(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    model = PatchConvNet(
        depth=60,
        dim=384,
        dropout=dropout,
        mlp_ratio=3,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained:
        logging.info(
            "Pretrained PatchConvNet models aren't available. Loading un-trained model instead"
        )

    return model


@register_model
def patchconvnet_s120(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    model = PatchConvNet(
        depth=120,
        dim=384,
        dropout=dropout,
        mlp_ratio=3,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained:
        logging.info(
            "Pretrained PatchConvNet models aren't available. Loading un-trained model instead"
        )

    return model


@register_model
def patchconvnet_b60(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    model = PatchConvNet(
        depth=60,
        dim=768,
        dropout=dropout,
        mlp_ratio=4,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained:
        logging.info(
            "Pretrained PatchConvNet models aren't available. Loading un-trained model instead"
        )

    return model


@register_model
def patchconvnet_b120(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    model = PatchConvNet(
        depth=120,
        dim=768,
        dropout=dropout,
        mlp_ratio=4,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained:
        logging.info(
            "Pretrained PatchConvNet models aren't available. Loading un-trained model instead"
        )

    return model


@register_model
def patchconvnet_l60(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    model = PatchConvNet(
        depth=60,
        dim=1024,
        dropout=dropout,
        mlp_ratio=3,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained:
        logging.info(
            "Pretrained PatchConvNet models aren't available. Loading un-trained model instead"
        )

    return model


@register_model
def patchconvnet_l120(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    model = PatchConvNet(
        depth=120,
        dim=1024,
        dropout=dropout,
        mlp_ratio=3,
        attach_head=attach_head,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained:
        logging.info(
            "Pretrained PatchConvNet models aren't available. Loading un-trained model instead"
        )

    return model
