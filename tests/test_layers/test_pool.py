import unittest

import jax.numpy as jnp
import jax.random as random

from jax_models.layers import AdaptiveAveragePool1D


class TestPoolLayers(unittest.TestCase):
    def test_adavgpool1d(self):
        adap = AdaptiveAveragePool1D(1)
        x = jnp.zeros([1, 196, 128])
        params = adap.init({"params": random.PRNGKey(0)}, x)
        x = adap.apply({"params": params}, x)
        self.assertEqual(x.shape, (1, 1, 128))
