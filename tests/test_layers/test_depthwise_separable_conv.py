import unittest
import jax.random as random
import jax.numpy as jnp

from jax_models.layers import DepthwiseConv2D, SeparableDepthwiseConv2D


class ParameterTest(unittest.TestCase):
    def test_params_shape_dw_conv(self):
        channel_multiplier = 2
        kernel_shape = (3, 3)
        strides = (2, 2)

        dw = DepthwiseConv2D(
            kernel_shape, strides, padding="SAME", channel_multiplier=channel_multiplier
        )
        x = jnp.zeros([1, 32, 32, 3])
        params = dw.init({"params": random.PRNGKey(0)}, x)["params"]
        self.assertEqual(params["kernel"].shape, (3, 3, 1, 6))
        self.assertEqual(params["bias"].shape, (6,))

    def test_param_shape_sep_conv(self):
        channel_multiplier = 2
        kernel_shape = (3, 3)
        strides = (2, 2)
        features = 256

        dw = SeparableDepthwiseConv2D(
            features,
            kernel_shape=kernel_shape,
            stride=strides,
            padding="SAME",
            channel_multiplier=channel_multiplier,
        )
        x = jnp.zeros([1, 32, 32, 3])
        params = dw.init({"params": random.PRNGKey(0)}, x)["params"]
        self.assertEqual(params["DepthwiseConv2D_0"]["kernel"].shape, (3, 3, 1, 6))
        self.assertEqual(params["Conv_0"]["kernel"].shape, (1, 1, 6, 256))
