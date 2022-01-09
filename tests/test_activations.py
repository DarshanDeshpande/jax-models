import unittest
import jax.numpy as jnp
import jax.random as random
import numpy as np

from jax_models.activations import PReLU, mish, hardswish, relu6


class TestActivations(unittest.TestCase):
    def test_mish(self):
        out = mish(jnp.asarray([1, 7, 0, -1, -6]))
        np.testing.assert_allclose(
            out, np.asarray([0.8650985, 6.999989, 0.0, -0.30340144, -0.01485408])
        ) is None

    def test_hardswish(self):
        out = hardswish(jnp.asarray([1, 7, 0, -1, -6]))
        np.testing.assert_allclose(
            out, np.asarray([0.6666667, 7.0, 0.0, -0.33333334, -0.0])
        ) is None

    def test_relu6(self):
        out = relu6(jnp.asarray([1, 7, 0, -1, -6]))
        np.testing.assert_allclose(out, np.asarray([1, 6, 0, 0, 0])) is None

    def test_prelu(self):
        prelu = PReLU()
        x = jnp.asarray([[1, 7, 0, -1, -6]])
        params = prelu.init({"params": random.PRNGKey(0)}, x)["params"]
        out = prelu.apply({"params": params}, x)
        np.testing.assert_allclose(out, np.asarray([[1, 7, 0, 0, 0]])) is None
