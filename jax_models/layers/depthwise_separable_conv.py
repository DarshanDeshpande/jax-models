import jax
import jax.numpy as jnp

import flax.linen as nn
from typing import Optional, Union, Tuple, Callable, Sequence


class DepthwiseConv2D(nn.Module):
    kernel_shape: Union[int, Sequence[int]] = (1, 1)
    stride: Union[int, Sequence[int]] = (1, 1)
    padding: str or Sequence[Tuple[int, int]] = "SAME"
    channel_multiplier: int = 1
    use_bias: bool = True
    weights_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Optional[Callable] = nn.initializers.zeros

    @nn.compact
    def __call__(self, input):
        w = self.param(
            "kernel",
            self.weights_init,
            self.kernel_shape + (1, self.channel_multiplier * input.shape[-1]),
        )
        if self.use_bias:
            b = self.param(
                "bias", self.bias_init, (self.channel_multiplier * input.shape[-1],)
            )

        conv = jax.lax.conv_general_dilated(
            input,
            w,
            self.stride,
            self.padding,
            (1,) * len(self.kernel_shape),
            (1,) * len(self.kernel_shape),
            ("NHWC", "HWIO", "NHWC"),
            input.shape[-1],
        )
        if self.use_bias:
            bias = jnp.broadcast_to(b, conv.shape)
            return conv + bias
        else:
            return conv


class SeparableDepthwiseConv2D(nn.Module):
    """Separable 2-D Depthwise Convolution Module."""

    features: int
    channel_multiplier: int
    kernel_shape: Union[int, Sequence[int]]
    stride: Union[int, Sequence[int]] = 1
    padding: Union[str, Sequence[Tuple[int, int]]] = "SAME"
    use_bias: bool = True
    weights_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Optional[Callable] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        depthwise_conv = DepthwiseConv2D(
            channel_multiplier=self.channel_multiplier,
            kernel_shape=self.kernel_shape,
            stride=self.stride,
            padding=self.padding,
            use_bias=False,
            weights_init=self.weights_init,
            bias_init=self.bias_init,
        )

        pointwise_conv = nn.Conv(
            self.features,
            kernel_size=[1, 1],
            strides=1,
            padding=self.padding,
            use_bias=False,
            kernel_init=self.weights_init,
            bias_init=self.bias_init,
        )

        sep_conv = pointwise_conv(depthwise_conv(inputs))

        if self.use_bias:
            b = self.param("bias", nn.zeros, (sep_conv.shape[-1]))

            return sep_conv + b

        return sep_conv
