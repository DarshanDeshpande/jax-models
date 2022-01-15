import unittest

import jax.numpy as jnp
import jax.random as random

from jax_models.layers import Attention


class TestAttention(unittest.TestCase):
    def test_attention_output(self):
        rng1, rng2 = random.split(random.PRNGKey(0))
        att = Attention(256)
        x = jnp.zeros([1, 64, 256])
        params = att.init({"params": rng1, "dropout": rng2}, x, False)["params"]
        out = att.apply({"params": params}, x, False, rngs={"dropout": rng2})
        self.assertEqual(out.shape, x.shape)
