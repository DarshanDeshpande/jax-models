import unittest
from jax import random
import jax.numpy as jnp

from jax_models.models.poolformer import (
    PoolFormer_S12,
    PoolFormer_S24,
    PoolFormer_S36,
    PoolFormer_M36,
    PoolFormer_M48,
)


class PoolformerTests(unittest.TestCase):
    def test_s12_init(self):
        rng, drop = random.split(random.PRNGKey(0), 2)
        s12 = PoolFormer_S12()
        x = jnp.ones([1, 256, 256, 3])
        params = s12.init({"params": rng, "dropout": drop}, x, False)["params"]
        sample_out = s12.apply({"params": params}, x, False, rngs={"dropout": drop})
        self.assertEqual(sample_out.shape, (1, 8, 8, 512))

    def test_s24_init(self):
        rng, drop = random.split(random.PRNGKey(0), 2)
        s24 = PoolFormer_S24()
        x = jnp.ones([1, 256, 256, 3])
        params = s24.init({"params": rng, "dropout": drop}, x, False)["params"]
        sample_out = s24.apply({"params": params}, x, False, rngs={"dropout": drop})
        self.assertEqual(sample_out.shape, (1, 8, 8, 512))

    def test_s36_init(self):
        rng, drop = random.split(random.PRNGKey(0), 2)
        s36 = PoolFormer_S36()
        x = jnp.ones([1, 256, 256, 3])
        params = s36.init({"params": rng, "dropout": drop}, x, False)["params"]
        sample_out = s36.apply({"params": params}, x, False, rngs={"dropout": drop})
        self.assertEqual(sample_out.shape, (1, 8, 8, 512))

    def test_m36_init(self):
        rng, drop = random.split(random.PRNGKey(0), 2)
        m36 = PoolFormer_M36()
        x = jnp.ones([1, 256, 256, 3])
        params = m36.init({"params": rng, "dropout": drop}, x, False)["params"]
        sample_out = m36.apply({"params": params}, x, False, rngs={"dropout": drop})
        self.assertEqual(sample_out.shape, (1, 8, 8, 768))

    def test_m48_init(self):
        rng, drop = random.split(random.PRNGKey(0), 2)
        m48 = PoolFormer_M48()
        x = jnp.ones([1, 256, 256, 3])
        params = m48.init({"params": rng, "dropout": drop}, x, False)["params"]
        sample_out = m48.apply({"params": params}, x, False, rngs={"dropout": drop})
        self.assertEqual(sample_out.shape, (1, 8, 8, 768))
