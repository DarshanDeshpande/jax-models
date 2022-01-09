import unittest
import jax.random as random
import jax.numpy as jnp

from jax_models.layers import SqueezeAndExcitation


class TestSqueezeAndExcitation(unittest.TestCase):
    def test_output_shape(self):
        sae = SqueezeAndExcitation()
        x = jnp.zeros([1, 32, 32, 256])
        params = sae.init({"params": random.PRNGKey(0)}, x)["params"]
        self.assertEqual(sae.apply({"params": params}, x).shape, x.shape)
