import jax.numpy as jnp
import flax.linen as nn

from ..layers import TransformerMLP
from .model_registry import register_model

from typing import Optional
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

__all__ = [
    "poolformer_s12",
    "poolformer_s24",
    "poolformer_s36",
    "poolformer_m36",
    "poolformer_m48",
]


class TransformerEncoder(nn.Module):
    mlp_dim: int
    pool_size: int
    stride: int
    dropout: float
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        norm = nn.LayerNorm()(inputs)
        att = nn.avg_pool(
            norm,
            (self.pool_size, self.pool_size),
            strides=(self.stride, self.stride),
            padding="SAME",
        )
        att = att - norm
        add = inputs + att
        x = nn.LayerNorm()(add)
        x = TransformerMLP(self.mlp_dim, self.mlp_dim, self.dropout)(x, deterministic)
        return add + x


class AddPositionEmbs(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2], inputs.shape[3])
        pe = self.param(
            "pos_embedding", nn.initializers.normal(stddev=0.02), pos_emb_shape
        )
        return inputs + pe


class S12(nn.Module):
    """
    S12 Module

    attach_head (bool): Whether to attach classification head. Default is True.
    num_classes (int): Number of classification classes. Default is 1000.
    dropout (float): Dropout value. Default is 0.1.
    deterministic (bool): Optional argument, if True, netowrk becomes deterministic and dropout is not applied.

    """

    attach_head: bool = False
    num_classes: int = 1000
    dropout: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(64, 3, 1, self.dropout)(x, deterministic)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(128, 3, 1, self.dropout)(x, deterministic)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(320, 3, 1, self.dropout)(x, deterministic)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(512, 3, 1, self.dropout)(x, deterministic)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)
        return x


class S24(nn.Module):
    """
    S24 Module

    attach_head (bool): Whether to attach classification head. Default is True.
    num_classes (int): Number of classification classes. Default is 1000.
    dropout (float): Dropout value. Default is 0.1.
    deterministic (bool): Optional argument, if True, netowrk becomes deterministic and dropout is not applied.

    """

    attach_head: bool = False
    num_classes: int = 1000
    dropout: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(64, 3, 1, self.dropout)(x, deterministic)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(128, 3, 1, self.dropout)(x, deterministic)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(12):
            x = TransformerEncoder(320, 3, 1, self.dropout)(x, deterministic)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(512, 3, 1, self.dropout)(x, deterministic)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


class S36(nn.Module):
    """
    S36 Module

    attach_head (bool): Whether to attach classification head. Default is True.
    num_classes (int): Number of classification classes. Default is 1000.
    dropout (float): Dropout value. Default is 0.1.
    deterministic (bool): Optional argument, if True, netowrk becomes deterministic and dropout is not applied.

    """

    attach_head: bool = False
    num_classes: int = 1000
    dropout: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(64, 3, 1, self.dropout)(x, deterministic)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(128, 3, 1, self.dropout)(x, deterministic)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(18):
            x = TransformerEncoder(320, 3, 1, self.dropout)(x, deterministic)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(512, 3, 1, self.dropout)(x, deterministic)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


class M36(nn.Module):
    """
    M36 Module

    attach_head (bool): Whether to attach classification head. Default is True.
    num_classes (int): Number of classification classes. Default is 1000.
    dropout (float): Dropout value. Default is 0.1.
    deterministic (bool): Optional argument, if True, netowrk becomes deterministic and dropout is not applied.

    """

    attach_head: bool = False
    num_classes: int = 1000
    dropout: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        # H/4, W/4
        x = nn.Conv(96, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(96, 3, 1, self.dropout)(x, deterministic)

        # H/8, W/8
        x = nn.Conv(192, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(192, 3, 1, self.dropout)(x, deterministic)

        # H/16, W/16
        x = nn.Conv(384, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(18):
            x = TransformerEncoder(384, 3, 1, self.dropout)(x, deterministic)

        # H/32, W/32
        x = nn.Conv(768, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(768, 3, 1, self.dropout)(x, deterministic)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


class M48(nn.Module):
    """
    M48 Module

    attach_head (bool): Whether to attach classification head. Default is True.
    num_classes (int): Number of classification classes. Default is 1000.
    dropout (float): Dropout value. Default is 0.1.
    deterministic (bool): Optional argument, if True, netowrk becomes deterministic and dropout is not applied.

    """

    attach_head: bool = False
    num_classes: int = 1000
    dropout: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        # H/4, W/4
        x = nn.Conv(96, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(96, 3, 1, self.dropout)(x, deterministic)

        # H/8, W/8
        x = nn.Conv(192, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(192, 3, 1, self.dropout)(x, deterministic)

        # H/16, W/16
        x = nn.Conv(384, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(24):
            x = TransformerEncoder(384, 3, 1, self.dropout)(x, deterministic)

        # H/32, W/32
        x = nn.Conv(768, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(768, 3, 1, self.dropout)(x, deterministic)

        if self.attach_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)

        return x


@register_model
def poolformer_s12(
    attach_head=False,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained PoolFormer_S12 isn't available. Loading un-trained model instead"
        )

    return S12(attach_head, num_classes, dropout, **kwargs)


@register_model
def poolformer_s24(
    attach_head=False,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained PoolFormer_S24 isn't available. Loading un-trained model instead"
        )

    return S24(attach_head, num_classes, dropout, **kwargs)


@register_model
def poolformer_s36(
    attach_head=False,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained PoolFormer_S36 isn't available. Loading un-trained model instead"
        )

    return S36(attach_head, num_classes, dropout, **kwargs)


@register_model
def poolformer_m36(
    attach_head=False,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained PoolFormer_M36 isn't available. Loading un-trained model instead"
        )

    return M36(attach_head, num_classes, dropout, **kwargs)


@register_model
def poolformer_m48(
    attach_head=False,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs
):
    if pretrained:
        logging.info(
            "Pretrained PoolFormer_M48 isn't available. Loading un-trained model instead"
        )

    return M48(attach_head, num_classes, dropout, **kwargs)
