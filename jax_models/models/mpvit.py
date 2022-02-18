import jax.numpy as jnp
import flax.linen as nn

from ..layers import DepthwiseConv2D, SeparableDepthwiseConv2D
from ..activations import hardswish
from .model_registry import register_model

from typing import Optional, Union, Sequence, Iterable
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


__all__ = ["MPViT", "mpvit_tiny", "mpvit_xsmall", "mpvit_small", "mpvit_base"]


class ConvolutionalStem(nn.Module):
    emb_dim: int = 128
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        conv = nn.Conv(
            self.emb_dim // 2, (3, 3), strides=2, padding="SAME", use_bias=False
        )(inputs)
        conv = nn.BatchNorm(deterministic)(conv)
        conv = hardswish(conv)

        conv = nn.Conv(self.emb_dim, (3, 3), strides=2, padding="SAME", use_bias=False)(
            conv
        )
        conv = nn.BatchNorm(deterministic)(conv)
        conv = hardswish(conv)
        b, h, w, c = conv.shape
        return jnp.reshape(conv, (b, h * w, c))


class MultiScalePatchEmbedding(nn.Module):
    features: int = 64
    kernel_size: Union[int, Sequence[int]] = (3, 3)
    strides: Union[int, Sequence[int]] = 1
    padding: Optional[Union[int, str]] = "SAME"
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        batch, dim, channels = inputs.shape
        height = width = int(dim ** 0.5)
        inputs = jnp.reshape(inputs, (batch, height, width, channels))

        conv3x3 = SeparableDepthwiseConv2D(
            self.features, 1, (3, 3), (self.strides, self.strides)
        )(inputs)
        conv3x3 = hardswish(conv3x3)
        conv3x3 = nn.BatchNorm(deterministic)(conv3x3)
        conv3x3 = jnp.reshape(conv3x3, (batch, -1, self.features))

        conv5x5 = SeparableDepthwiseConv2D(self.features, 1, (3, 3), (1, 1))(inputs)
        conv5x5 = hardswish(conv5x5)
        conv5x5 = nn.BatchNorm(deterministic)(conv5x5)
        conv5x5 = SeparableDepthwiseConv2D(
            self.features, 1, (3, 3), (self.strides, self.strides)
        )(conv5x5)
        conv5x5 = hardswish(conv5x5)
        conv5x5 = nn.BatchNorm(deterministic)(conv5x5)
        conv5x5 = jnp.reshape(conv5x5, (batch, -1, self.features))

        conv7x7 = SeparableDepthwiseConv2D(self.features, 1, (3, 3), (1, 1))(inputs)
        conv7x7 = hardswish(conv7x7)
        conv7x7 = nn.BatchNorm(deterministic)(conv7x7)
        conv7x7 = SeparableDepthwiseConv2D(self.features, 1, (3, 3), (1, 1))(conv7x7)
        conv7x7 = hardswish(conv7x7)
        conv7x7 = nn.BatchNorm(deterministic)(conv7x7)
        conv7x7 = SeparableDepthwiseConv2D(
            self.features, 1, (3, 3), (self.strides, self.strides)
        )(conv7x7)
        conv7x7 = hardswish(conv7x7)
        conv7x7 = nn.BatchNorm(deterministic)(conv7x7)
        conv7x7 = jnp.reshape(conv7x7, (batch, -1, self.features))

        return [conv3x3, conv5x5, conv7x7]


class ConvolutionalLocalFeature(nn.Module):
    features: int = 64
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        batch, n, channels = inputs.shape
        height = width = int(n ** 0.5)
        inputs = jnp.reshape(inputs, (batch, height, width, channels))

        skip = inputs
        conv = SeparableDepthwiseConv2D(self.features, 1, (1, 1), (1, 1), "SAME")(
            inputs
        )
        conv = nn.BatchNorm(use_running_average=deterministic)(conv)
        conv = hardswish(conv)

        conv = DepthwiseConv2D((3, 3), (1, 1), "SAME", 1)(conv)
        conv = nn.BatchNorm(use_running_average=deterministic)(conv)
        conv = hardswish(conv)

        conv = SeparableDepthwiseConv2D(self.features, 1, (1, 1), (1, 1), "SAME")(conv)
        conv = nn.BatchNorm(use_running_average=deterministic)(conv)
        conv = hardswish(conv)

        residual = conv + skip
        batch, height, width, channels = residual.shape
        return jnp.reshape(residual, (batch, height * width, channels))


class ConvPosEnc(nn.Module):
    """
    Implementation translated from the official repository of CoaT: https://github.com/mlpc-ucsd/CoaT
    """

    dim: int

    @nn.compact
    def __call__(self, inputs):
        batch, n, channels = inputs.shape
        height = width = int((n - 1) ** 0.5)
        cls_token, img_tokens = inputs[:, :1], inputs[:, 1:]
        features = jnp.reshape(img_tokens, [batch, height, width, channels])
        dwconv = DepthwiseConv2D((3, 3))(features)
        x = dwconv + features
        x = jnp.reshape(x, [batch, height * width, channels])

        x = jnp.concatenate([cls_token, x], axis=1)
        return x


class ConvRelPosEnc(nn.Module):
    """
    Implementation translated from the official repository of CoaT: https://github.com/mlpc-ucsd/CoaT
    """

    window_size: int

    @nn.compact
    def __call__(self, q, v, size: tuple):
        batch, num_heads, n, channels_per_head = q.shape
        height, width = size
        assert n == 1 + height * width

        windows = {self.window_size: num_heads}
        conv_list = []
        head_splits = []

        for window, head_split in windows.items():
            conv_list.append(DepthwiseConv2D((window, window), channel_multiplier=1))
            head_splits.append(head_split)
        channel_splits = [x * channels_per_head for x in head_splits]

        q_img = q[:, :, 1:, :]
        v_img = v[:, :, 1:, :]

        v_img = jnp.reshape(
            v_img, (batch, num_heads * channels_per_head, height, width)
        )
        v_img_list = jnp.array_split(v_img, channel_splits, axis=1)
        conv_v_img_list = [
            conv(jnp.transpose(x, (0, 2, 3, 1)))
            for conv, x in zip(conv_list, v_img_list)
        ]
        conv_v_img = jnp.concatenate(conv_v_img_list, axis=-1)
        conv_v_img = jnp.reshape(
            conv_v_img, (batch, num_heads, height * width, channels_per_head)
        )

        EV_hat_img = q_img * conv_v_img
        zero = jnp.zeros([batch, num_heads, 1, channels_per_head])
        EV_hat = jnp.concatenate((zero, EV_hat_img), axis=2)
        return EV_hat


class FactorizedAttention(nn.Module):
    """
    Implementation translated from the official repository of CoaT: https://github.com/mlpc-ucsd/CoaT
    """

    dim: int
    num_heads: int
    drop_prob: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5

        batch, n, channels = inputs.shape
        height = width = int((n - 1) ** 0.5)
        size = (height, width)

        # Conv Positional Encoding
        pos_enc = ConvPosEnc(channels)(inputs)
        assert pos_enc.shape == inputs.shape

        # Generate Q, K, V.
        qkv = nn.Dense(self.dim * 3, use_bias=False)(pos_enc)
        qkv = qkv.reshape(batch, n, 3, self.num_heads, channels // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.
        k_softmax = nn.softmax(k, axis=2)
        k_softmax_T_dot_v = jnp.einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = jnp.einsum("b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v)

        # ConvRel Positional Encoding
        crpe = ConvRelPosEnc(1)(q, v, size)

        # Merge and reshape.
        x = scale * factor_att + crpe
        x = x.transpose(0, 1, 2, 3).reshape(batch, n, channels)

        # Output projection.
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(self.drop_prob)(x, deterministic)

        return x


class TransformerEncoder(nn.Module):
    dim: int = 256
    num_heads: int = 2
    att_drop: float = 0.2
    proj_drop: float = 0.2
    mlp_ratio: int = 2
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        skip = inputs
        norm = nn.LayerNorm()(inputs)
        mhsa = FactorizedAttention(
            self.dim, self.num_heads, self.att_drop, deterministic
        )(norm)
        mhsa = mhsa + skip
        norm = nn.LayerNorm()(mhsa)
        x = nn.Dense(self.dim * self.mlp_ratio)(norm)
        x = nn.gelu(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic=deterministic)
        x = nn.Dense(self.dim)(x)
        x = x + mhsa
        return x


class MultiPathTransformerBlock(nn.Module):
    deterministic: Optional[bool] = None
    features: int = 64
    dim: int = 64
    num_heads: int = 2
    att_drop: float = 0.2
    proj_drop: float = 0.2
    mlp_ratio: int = 2
    num_encoder_layers: int = 1

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x1, x2, x3 = inputs
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        clf = ConvolutionalLocalFeature(self.features)(x1, deterministic=deterministic)

        for index in range(self.num_encoder_layers):
            batch, n, channels = x1.shape

            # Add cls token
            x1 = jnp.concatenate(
                [
                    self.param(
                        f"param_1_{index}", nn.initializers.zeros, (batch, 1, channels)
                    ),
                    x1,
                ],
                axis=1,
            )
            x2 = jnp.concatenate(
                [
                    self.param(
                        f"param_2_{index}", nn.initializers.zeros, (batch, 1, channels)
                    ),
                    x2,
                ],
                axis=1,
            )
            x3 = jnp.concatenate(
                [
                    self.param(
                        f"param_3_{index}", nn.initializers.zeros, (batch, 1, channels)
                    ),
                    x3,
                ],
                axis=1,
            )

            x1 = TransformerEncoder(
                self.dim, self.num_heads, self.att_drop, self.proj_drop, self.mlp_ratio
            )(x1, deterministic)
            x2 = TransformerEncoder(
                self.dim, self.num_heads, self.att_drop, self.proj_drop, self.mlp_ratio
            )(x2, deterministic)
            x3 = TransformerEncoder(
                self.dim, self.num_heads, self.att_drop, self.proj_drop, self.mlp_ratio
            )(x3, deterministic)

            # Remove cls token
            x1, x2, x3 = x1[:, 1:, :], x2[:, 1:, :], x3[:, 1:, :]

        concat = jnp.concatenate([clf, x1, x2, x3], axis=-1)
        conv = nn.Conv(self.features, [1, 1], 1)(concat)
        return conv


class MPViT(nn.Module):
    """
    MPViT Module

    Attributes:
        mlp_ratio (int): Multiplier for hidden dimension in transformer MLP block. Default is 2.
        channels_list (list or tuple): Number of channels for each stage.
        num_layers_list (list or tuple): Number of layers for each stage.
        att_drop (float): Dropout value for attention Default is 0.2.
        proj_drop (float): Dropout value for attention projection. Default is 0.2.
        attach_head (bool): Whether to attach classification head. Default is True.
        num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    mlp_ratio: int = 2
    channels_list: Iterable[int] = (64, 96, 176, 216)
    num_layers_list: Iterable[int] = (1, 2, 4, 1)
    att_drop: float = 0.2
    proj_drop: float = 0.2
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        stem = ConvolutionalStem(64)(inputs, deterministic=deterministic)
        mptb = stem
        for i, j in zip(self.channels_list, self.num_layers_list):
            mspe = MultiScalePatchEmbedding(features=i, strides=2)(mptb, deterministic)
            mptb = MultiPathTransformerBlock(
                features=i,
                dim=i,
                num_heads=8,
                num_encoder_layers=j,
                mlp_ratio=self.mlp_ratio,
                att_drop=self.att_drop,
                proj_drop=self.proj_drop,
            )(mspe, deterministic)

        if self.attach_head:
            # Global avg pooling
            x = jnp.mean(mptb, 1)
            x = nn.Dense(self.num_classes)(x)
            x = nn.softmax(x)
            return x

        return mptb


@register_model
def mpvit_tiny(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained model for MPViT tiny isn't available. Loading un-trained model instead"
        )

    return MPViT(
        mlp_ratio=2,
        channels_list=(64, 96, 176, 216),
        num_layers_list=(1, 2, 4, 1),
        attach_head=attach_head,
        num_classes=num_classes,
        att_drop=dropout,
        proj_drop=dropout,
        **kwargs,
    )


@register_model
def mpvit_xsmall(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained model for MPViT XSmall isn't available. Loading un-trained model instead"
        )

    return MPViT(
        mlp_ratio=4,
        channels_list=(64, 128, 192, 256),
        num_layers_list=(1, 2, 4, 1),
        attach_head=attach_head,
        num_classes=num_classes,
        att_drop=dropout,
        proj_drop=dropout,
        **kwargs,
    )


@register_model
def mpvit_small(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained model for MPViT Small isn't available. Loading un-trained model instead"
        )

    return MPViT(
        mlp_ratio=4,
        channels_list=(64, 128, 216, 288),
        num_layers_list=(1, 3, 6, 3),
        attach_head=attach_head,
        num_classes=num_classes,
        att_drop=dropout,
        proj_drop=dropout,
        **kwargs,
    )


@register_model
def mpvit_base(
    attach_head=True,
    num_classes=1000,
    dropout=0.1,
    pretrained=False,
    download_dir=None,
    **kwargs,
):
    if pretrained:
        logging.info(
            "Pretrained model for MPViT Base isn't available. Loading un-trained model instead"
        )

    return MPViT(
        mlp_ratio=4,
        channels_list=(128, 224, 368, 480),
        num_layers_list=(1, 3, 8, 3),
        attach_head=attach_head,
        num_classes=num_classes,
        att_drop=dropout,
        proj_drop=dropout,
        **kwargs,
    )
