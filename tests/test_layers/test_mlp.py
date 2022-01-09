import unittest
import jax.random as random
import jax.numpy as jnp

from jax_models.layers import TransformerMLP


class TestMLP(unittest.TestCase):
    def test_shape(self):
        drop, key = random.split(random.PRNGKey(0), 2)

        dim, out_dim = 64, 256
        tmlp = TransformerMLP(dim, out_dim, dropout=0.1)
        x = jnp.zeros([1, 32, 64])
        params = tmlp.init({"params": key, "dropout": drop}, x, False)["params"]
        out = tmlp.apply({"params": params}, x, False, rngs={"dropout": drop})

        self.assertEqual(out.shape, x.shape[:-1] + (out_dim,))
