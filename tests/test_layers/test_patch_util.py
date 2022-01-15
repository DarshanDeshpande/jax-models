import unittest
import jax.numpy as jnp
import jax.random as random

from jax_models.layers import ExtractPatches, MergePatches


class TestPatchLayers(unittest.TestCase):
    def test_patch_creation(self):
        x = jnp.zeros([1, 64, 64, 3])
        patch_size = 16
        layer = ExtractPatches()
        params = layer.init({"params": random.PRNGKey(0)}, x, patch_size)
        out = layer.apply({"params": params}, x, patch_size)
        self.assertEqual(out.shape, (1, 16, 768))

    def test_patch_merging(self):
        x = jnp.zeros([1, 16, 768])
        patch_size = 16
        layer = MergePatches()
        params = layer.init({"params": random.PRNGKey(0)}, x, patch_size)
        out = layer.apply({"params": params}, x, patch_size)
        self.assertEqual(out.shape, (1, 64, 64, 3))
